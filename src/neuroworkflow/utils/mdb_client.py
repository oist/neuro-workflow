#!/usr/bin/env python3
"""HTTP client for the bm_mindsdb (mdb) dataset metadata catalog.

mdb (https://github.com/oist/bm_mindsdb) aggregates dataset metadata from DANDI,
CBS, Brain/MINDS and BMB Human into a local SQLite catalog, and indexes on-disk
BIDS trees as a "local catalog". This client is a thin wrapper over its REST API.

It complements :mod:`neuroworkflow.utils.remote_catalogs`, which talks to those
upstream APIs directly. The two differ in what they can do:

* ``remote_catalogs`` — live queries against the upstream API. Always current,
  but slow, one source at a time, and subject to upstream outages.
* this module — queries mdb's synced catalog. Fast, searchable across all
  sources at once, reproducible between runs, and the only way to reach the
  local BIDS catalog. Reflects the catalog as of the last sync.

Envelope contract (matching ``remote_catalogs.clients``): no method raises.
Every call returns ``{status, count, total, datasets, ...}`` on success and
``{status: "error", error, ...}`` on failure, so nodes never abort a workflow
on a network problem.

``POST /api/execute_sql`` is deliberately not wrapped: mdb runs it without an
allow-list or a read-only guard, and nothing here needs raw SQL.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

try:  # advertise a real User-Agent instead of the default requests one
    from importlib.metadata import version as _pkg_version

    _NW_VERSION = _pkg_version("neuroworkflow")
except Exception:  # pragma: no cover - fallback when metadata is unavailable
    _NW_VERSION = "0.1.0"
USER_AGENT = f"neuroworkflow/{_NW_VERSION} (+https://github.com/oist/neuro-workflow)"

DEFAULT_TIMEOUT = 30
#: Compose service name. Resolves from the backend and from spawned kernels,
#: both of which share a network with the ``mdb`` service.
DEFAULT_BASE_URL = "http://mdb:8004"

#: Source keys mdb recognises. ``aws`` is the local BIDS catalog (SRPBS_TS);
#: the rest are remote catalogs populated by ``POST /api/sync_apis``.
KNOWN_SOURCES = ("dandi", "cbs", "brainminds", "bmb_human", "aws")

#: Tables ``GET /api/catalog_lookup`` accepts.
LOOKUP_TABLES = ("api_datasets", "local_catalog_datasets", "metadata_entries")

#: Views ``GET /api/local_catalog/<source>/<dataset_id>/...`` exposes.
#: ``index`` is the bare dataset endpoint with no trailing segment.
LOCAL_CATALOG_VIEWS = ("index", "participants", "sessions", "sites")


def resolve_base_url(base_url: Optional[str] = None) -> str:
    """Resolve the mdb base URL: explicit argument > ``MDB_BASE_URL`` > default.

    The environment variable is what compose sets for the backend and what the
    JupyterHub spawner injects into each kernel, so nodes normally need no
    configuration. Passing ``base_url`` lets a standalone (pip-installed) user
    point at their own mdb.
    """
    return (base_url or os.environ.get("MDB_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


class MDBClient:
    """Read-only client for the mdb catalog REST API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.base_url = resolve_base_url(base_url)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": USER_AGENT}
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _error(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": message,
            "count": 0,
            "total": 0,
            "datasets": [],
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
        }

    def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """GET ``path`` and return the decoded JSON body, or an error envelope.

        mdb reports failures three different ways, all handled here: a 4xx/5xx
        with a JSON ``error`` body (missing id, dataset not ingested), a 200
        with an ``error`` key, and a transport failure.
        """
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ mdb request failed ({url}): {e}")
            return self._error(f"mdb unreachable at {self.base_url}: {e}")

        try:
            payload = response.json()
        except ValueError:
            if not response.ok:
                return self._error(
                    f"mdb returned HTTP {response.status_code} for {path}"
                )
            logger.error(f"❌ mdb returned a non-JSON response ({url})")
            return self._error(f"mdb returned a non-JSON response for {path}")

        if not isinstance(payload, dict):
            return self._error(
                f"mdb returned an unexpected payload type: {type(payload).__name__}"
            )
        if not response.ok:
            # mdb puts a human-readable reason in the body; surface that rather
            # than the bare status code.
            return self._error(
                str(payload.get("error") or f"mdb returned HTTP {response.status_code}")
            )
        if payload.get("error"):
            return self._error(str(payload["error"]))
        return payload

    def _envelope(
        self, payload: Dict[str, Any], records_key: str, **extra: Any
    ) -> Dict[str, Any]:
        """Normalise an mdb response into the shared node envelope."""
        if payload.get("status") == "error":
            return payload

        records = payload.get(records_key) or []
        envelope = {
            "status": "success",
            "count": payload.get("count", len(records)),
            "total": payload.get("total", payload.get("count", len(records))),
            "datasets": records,
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
        }
        envelope.update(extra)
        return envelope

    # ------------------------------------------------------------------
    # catalog
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """``GET /api/api_statistics`` — per-source counts and sync status.

        Doubles as a health check: a ``success`` status means mdb is reachable.
        """
        payload = self._get("/api/api_statistics")
        if payload.get("status") == "error":
            return payload
        return {
            "status": "success",
            "statistics": payload,
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
        }

    def list_datasets(self, source: str = "", limit: int = 50) -> Dict[str, Any]:
        """``GET /api/api_datasets`` — list synced datasets, newest sync first."""
        params: Dict[str, Any] = {"limit": limit}
        if source:
            params["source"] = source
        return self._envelope(
            self._get("/api/api_datasets", params), "datasets", source=source or "all"
        )

    def search_datasets(
        self, query: str, source: str = "", limit: int = 50
    ) -> Dict[str, Any]:
        """``GET /api/search_api_datasets`` — full-text search across the catalog.

        An empty ``query`` falls back to :meth:`list_datasets`, since mdb's
        search endpoint requires a term.
        """
        if not query:
            return self.list_datasets(source=source, limit=limit)

        params: Dict[str, Any] = {"q": query}
        if source:
            params["source"] = source
        envelope = self._envelope(
            self._get("/api/search_api_datasets", params),
            "datasets",
            query=query,
            source=source or "all",
        )
        # mdb's search endpoint has no limit parameter, so cap client-side to
        # keep the node's `limit` meaningful.
        if envelope.get("status") == "success" and limit:
            datasets = envelope["datasets"][:limit]
            envelope["datasets"] = datasets
            envelope["count"] = len(datasets)
        return envelope

    def lookup(
        self,
        dataset_id: str,
        source: str = "dandi",
        table: str = "api_datasets",
    ) -> Dict[str, Any]:
        """``GET /api/catalog_lookup`` — fetch one dataset record by ID.

        mdb normalises the ID per source (e.g. a bare DANDI number), so the ID
        does not have to match the stored form exactly.
        """
        if not dataset_id:
            return self._error("dataset_id is required")

        payload = self._get(
            "/api/catalog_lookup",
            {"id": dataset_id, "source": source, "table": table},
        )
        if payload.get("status") == "error":
            return payload

        record = payload.get("record")
        return {
            "status": "success",
            "count": 1 if record else 0,
            "total": 1 if record else 0,
            "record": record,
            "requested_id": payload.get("requested_id", dataset_id),
            "normalized_id": payload.get("normalized_id"),
            "source": payload.get("source", source),
            "table": payload.get("table", table),
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
        }

    def local_catalog(
        self,
        source: str = "aws",
        dataset_id: str = "srpbs-ts",
        view: str = "participants",
        site_code: str = "",
        participant_id: str = "",
        limit: int = 500,
    ) -> Dict[str, Any]:
        """``GET /api/local_catalog/<source>/<dataset_id>[/<view>]``.

        Reads the normalised BIDS index mdb builds from an on-disk dataset.
        ``view`` selects the collection: ``participants``, ``sessions``,
        ``sites``, or ``index`` for the dataset record itself. ``site_code`` and
        ``participant_id`` filter the ``sessions`` view only.
        """
        if view not in LOCAL_CATALOG_VIEWS:
            return self._error(
                f"unknown view '{view}' (expected one of {', '.join(LOCAL_CATALOG_VIEWS)})"
            )

        base = f"/api/local_catalog/{quote(str(source))}/{quote(str(dataset_id))}"
        if view == "index":
            payload = self._get(base)
            if payload.get("status") == "error":
                return payload
            index = payload.get("index") or {}
            return {
                "status": "success",
                "count": 1 if index else 0,
                "total": 1 if index else 0,
                "index": index,
                "source": source,
                "dataset_id": dataset_id,
                "base_url": self.base_url,
                "timestamp": datetime.now().isoformat(),
            }

        params: Dict[str, Any] = {}
        if view == "sessions":
            params["limit"] = limit
            if site_code:
                params["site_code"] = site_code
            if participant_id:
                params["participant_id"] = participant_id

        # For these views mdb names the record list after the view itself.
        return self._envelope(
            self._get(f"{base}/{view}", params or None),
            view,
            view=view,
            source=source,
            dataset_id=dataset_id,
        )
