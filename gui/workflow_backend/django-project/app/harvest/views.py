"""OAI-PMH harvest APIs backed by the local record store.

``manage.py harvest_oai`` (looped by the compose ``harvester`` service) keeps a
copy of the repository in ``harvested_records``; the views here only read that
table or stream file downloads. The repository address and API key live only
in the backend environment (``OAI_PMH_*`` in ``gui/workflow_backend/.env``).

There are two auth planes here: the records/download views take the kernel
service token (``X-Api-Key``, the same token the Anthropic proxy accepts),
while :class:`OAIPMHSearchView` serves the browser with the user's Keycloak
token. Keep them separate — the service token must never be handed to browsers.
"""

import logging
import os

import httpx
from app.auth.authentication import KeycloakAuthentication
from app.harvest import services
from app.harvest.models import HarvestedRecord
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

logger = logging.getLogger(__name__)

DEFAULT_KEY_HEADER = "X-MDRS-API-Key"
DEFAULT_TIMEOUT = 60.0

DEFAULT_SEARCH_LIMIT = 25
MAX_SEARCH_LIMIT = 100
MAX_RECORDS_PER_REQUEST = 100


def _service_token_error(request):
    expected = os.environ.get("JUPYTERHUB_API_TOKEN", "")
    provided = request.headers.get("x-api-key")
    if not expected or provided != expected:
        return JsonResponse({"error": "Invalid service token"}, status=401)
    return None


def _upstream_headers():
    headers = {
        "accept": "text/xml, application/xml, */*",
        "accept-encoding": "identity",
    }
    key = os.environ.get("OAI_PMH_API_KEY", "")
    if key:
        headers[os.environ.get("OAI_PMH_API_KEY_HEADER") or DEFAULT_KEY_HEADER] = key
    return headers


def _timeout():
    try:
        return float(os.environ.get("OAI_PMH_TIMEOUT", DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def _relay(url, params=None, forward_headers=(), default_content_type="text/xml"):
    """Stream an upstream GET back to the caller with the configured key attached."""
    client = httpx.Client(timeout=_timeout())
    try:
        upstream = client.send(
            client.build_request(
                "GET", url, params=params, headers=_upstream_headers()
            ),
            stream=True,
        )
    except httpx.HTTPError as e:
        client.close()
        logger.error("OAI-PMH proxy upstream error: %s", e)
        return JsonResponse({"error": f"Upstream error: {e}"}, status=502)

    def _stream():
        try:
            # identity encoding upstream, so the bytes match the headers we forward.
            yield from upstream.iter_bytes()
        finally:
            upstream.close()
            client.close()

    response = StreamingHttpResponse(
        _stream(),
        status=upstream.status_code,
        content_type=upstream.headers.get("content-type", default_content_type),
    )
    for name in ("retry-after", *forward_headers):
        if name in upstream.headers:
            response[name] = upstream.headers[name]
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


class OAIPMHFileDownloadView(View):
    """``GET /api/harvest/oai/files/<uuid>/download/`` — stream one repository file."""

    def get(self, request, file_id):
        denied = _service_token_error(request)
        if denied:
            return denied
        template = os.environ.get("OAI_PMH_FILE_DOWNLOAD_URL", "")
        if not template:
            return JsonResponse(
                {"error": "OAI_PMH_FILE_DOWNLOAD_URL is not configured on the backend"},
                status=500,
            )
        return _relay(
            template.format(file_id=file_id),
            forward_headers=("content-length", "content-disposition"),
            default_content_type="application/octet-stream",
        )


class OAIPMHRecordsView(View):
    """``GET /api/harvest/records/?identifiers=a,b`` — harvested records for kernels.

    Kernel plane (service token). Serves the local record store; identifiers
    unknown to it are listed under ``missing`` and deleted records are returned
    with ``deleted: true`` (the download node skips them).
    """

    def get(self, request):
        denied = _service_token_error(request)
        if denied:
            return denied
        identifiers = []
        for part in request.GET.get("identifiers", "").split(","):
            ident = part.strip()
            if ident and ident not in identifiers:
                identifiers.append(ident)
        if not identifiers:
            return JsonResponse({"error": "identifiers is required"}, status=400)
        if len(identifiers) > MAX_RECORDS_PER_REQUEST:
            return JsonResponse(
                {
                    "error": (
                        f"At most {MAX_RECORDS_PER_REQUEST} identifiers per request"
                    )
                },
                status=400,
            )
        rows = {
            row.oai_identifier: row
            for row in HarvestedRecord.objects.filter(oai_identifier__in=identifiers)
        }
        return JsonResponse(
            {
                "status": "success",
                "records": [
                    services.record_payload(rows[i]) for i in identifiers if i in rows
                ],
                "count": len(rows),
                "missing": [i for i in identifiers if i not in rows],
            }
        )


def _record_metadata(record):
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _summary(record):
    """Project one parsed record onto the fields the search UI displays."""
    metadata = _record_metadata(record)
    return {
        "identifier": record.get("identifier", ""),
        "name": str(metadata.get("name", "")),
        "description": str(metadata.get("description", "")),
        "laboratory_name": str(metadata.get("laboratory_name", "")),
        "datestamp": record.get("datestamp", ""),
        "set_specs": record.get("set_specs", []),
        "file_count": len(record.get("files", [])),
        "size": metadata.get("size", ""),
    }


class OAIPMHSearchView(APIView):
    """``GET /api/harvest/oai/search/?q=...`` — keyword search for the GUI.

    Browser plane: authenticated with the user's Keycloak token, unlike the
    kernel-plane views above. Queries the harvested copy in the database (each
    term must appear in the record's ``search_text`` haystack), so results are
    at most one harvest interval stale — ``harvested_at`` reports how stale.
    """

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            limit = int(request.GET.get("limit", DEFAULT_SEARCH_LIMIT))
        except ValueError:
            return JsonResponse({"error": "limit must be an integer"}, status=400)
        limit = min(max(limit, 1), MAX_SEARCH_LIMIT)
        query = request.GET.get("q", "").strip()
        set_spec = request.GET.get("set", "").strip()

        qs = HarvestedRecord.objects.filter(deleted=False)
        if set_spec:
            qs = qs.filter(set_specs__contains=[set_spec])
        for term in query.lower().split():
            qs = qs.filter(search_text__contains=term)
        rows = list(qs[: limit + 1])
        run = services.latest_success_run()
        return JsonResponse(
            {
                "status": "success",
                "query": query,
                "set": set_spec,
                "count": min(len(rows), limit),
                "scanned": HarvestedRecord.objects.filter(deleted=False).count(),
                "truncated": len(rows) > limit,
                "results": [
                    _summary(services.record_payload(row)) for row in rows[:limit]
                ],
                "harvested_at": run.finished_at.isoformat() if run else None,
            }
        )
