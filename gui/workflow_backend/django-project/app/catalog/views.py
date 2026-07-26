"""Authenticated proxy to the bm_mindsdb (mdb) dataset catalog.

The browser cannot call mdb directly: mdb sends no CORS headers, and its port is
deliberately not published because it has no authentication of its own and its
``POST /api/execute_sql`` runs arbitrary SQL. This module is the only browser-facing
route to the catalog, so it does two things mdb does not:

* requires a Keycloak JWT, and
* forwards only the read paths listed in ``ALLOWED_ROUTES`` plus the one write
  action (catalog sync). Anything else 404s here and never reaches mdb.

Kernel-side database nodes do not go through this proxy — they talk to mdb directly
over the compose network via ``neuroworkflow.utils.mdb_client``.
"""

import logging
import os

import httpx
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.auth.authentication import KeycloakAuthentication

logger = logging.getLogger(__name__)

#: Read-only mdb paths this proxy will forward, keyed by the name used in our URLconf.
#: Deliberately excluded: /api/execute_sql (arbitrary SQL), the /api/mindsdb_* server
#: lifecycle endpoints, and the ML/LLM endpoints — none are needed by the catalog UI.
ALLOWED_ROUTES = {
    "statistics": "/api/api_statistics",
    "datasets": "/api/api_datasets",
    "search": "/api/search_api_datasets",
    "lookup": "/api/catalog_lookup",
}

#: Query parameters forwarded per route. Anything else is dropped rather than
#: passed through, so a caller cannot smuggle parameters into mdb.
ALLOWED_PARAMS = {
    "statistics": (),
    "datasets": ("source", "limit"),
    "search": ("q", "source"),
    "lookup": ("id", "source", "table"),
    "local": ("site_code", "participant_id", "limit"),
}

#: Views of /api/local_catalog/<source>/<dataset_id>[/<view>].
LOCAL_VIEWS = ("index", "participants", "sessions", "sites")


def get_mdb_base_url() -> str:
    """Return the configured mdb base URL, or an empty string when unset.

    Read at call time rather than import time so tests and a restart-free config
    change both take effect, matching how LOCAL_RAG_* is handled in app/metadata.
    """
    return (os.environ.get("MDB_BASE_URL") or "").rstrip("/")


def get_mdb_timeout() -> float:
    try:
        return float(os.environ.get("MDB_TIMEOUT", "30"))
    except ValueError:
        return 30.0


def get_mdb_sync_timeout() -> float:
    """Sync walks four upstream APIs (DANDI alone has ~900 datasets), so it needs
    far longer than a catalog read."""
    try:
        return float(os.environ.get("MDB_SYNC_TIMEOUT", "600"))
    except ValueError:
        return 600.0


def _unavailable() -> Response:
    return Response(
        {
            "available": False,
            "error": (
                "MDB_BASE_URL is not configured on the backend. The dataset "
                "catalog is unavailable."
            ),
        },
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def _forward(
    method: str,
    path: str,
    params: dict | None = None,
    timeout: float | None = None,
) -> Response:
    """Send one request to mdb and translate the result into a DRF Response."""
    base = get_mdb_base_url()
    if not base:
        return _unavailable()

    url = f"{base}{path}"
    try:
        response = httpx.request(
            method, url, params=params, timeout=timeout or get_mdb_timeout()
        )
    except httpx.HTTPError as e:
        logger.error("catalog proxy: mdb unreachable (%s): %s", url, e)
        return Response(
            {"available": False, "error": f"mdb unreachable at {base}: {e}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    try:
        payload = response.json()
    except ValueError:
        logger.error("catalog proxy: non-JSON response from mdb (%s)", url)
        return Response(
            {"error": f"mdb sent a non-JSON response (HTTP {response.status_code})"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    # mdb's own status codes are meaningful (404 = not in catalog, 400 = bad
    # request), so pass them through instead of flattening everything to 200.
    return Response(payload, status=response.status_code)


def _picked_params(request, route: str) -> dict:
    allowed = ALLOWED_PARAMS.get(route, ())
    return {k: v for k, v in request.query_params.items() if k in allowed}


class CatalogReadView(APIView):
    """Forward one allow-listed read route to mdb.

    The route name comes from the URLconf, never from user input, so a caller
    cannot reach an mdb path that is not in ALLOWED_ROUTES.
    """

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]
    route = ""

    def get(self, request):
        path = ALLOWED_ROUTES.get(self.route)
        if not path:  # pragma: no cover - guards a URLconf wiring mistake
            return Response(
                {"error": f"unknown catalog route '{self.route}'"},
                status=status.HTTP_404_NOT_FOUND,
            )
        return _forward("GET", path, _picked_params(request, self.route))


class CatalogLocalView(APIView):
    """Forward the local BIDS catalog routes, validating the view segment."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, source: str, dataset_id: str, view: str = "index"):
        if view not in LOCAL_VIEWS:
            return Response(
                {"error": f"unknown view '{view}'"},
                status=status.HTTP_404_NOT_FOUND,
            )
        path = f"/api/local_catalog/{source}/{dataset_id}"
        if view != "index":
            path = f"{path}/{view}"
        params = _picked_params(request, "local") if view == "sessions" else {}
        return _forward("GET", path, params)


@method_decorator(csrf_exempt, name="dispatch")
class CatalogSyncView(APIView):
    """Trigger mdb's remote catalog sync (``POST /api/sync_apis``).

    The only write this proxy allows. mdb answers 200 even when individual
    sources fail, so the per-source results in the body are what matter — they
    are passed through unchanged.
    """

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        return _forward("POST", "/api/sync_apis", timeout=get_mdb_sync_timeout())
