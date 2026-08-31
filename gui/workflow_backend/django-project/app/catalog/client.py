import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

ALLOWED_SOURCES = frozenset({"dandi", "cbs", "brainminds", "bmb_human", "aws"})
ALLOWED_LOOKUP_TABLES = frozenset({"api_datasets"})
DEFAULT_LOOKUP_TABLE = "api_datasets"
LIMIT_MIN = 1
LIMIT_MAX = 200
LIMIT_DEFAULT = 50

MDB_PATH_STATISTICS = "/api/api_statistics"
MDB_PATH_SEARCH = "/api/catalog_search"
MDB_PATH_LOOKUP = "/api/catalog_lookup"
MDB_PATH_DATASETS = "/api/api_datasets"
ALLOWED_MDB_PATHS = frozenset(
    {
        MDB_PATH_STATISTICS,
        MDB_PATH_SEARCH,
        MDB_PATH_LOOKUP,
        MDB_PATH_DATASETS,
    }
)

_DOCKER_ENV_PATH = Path("/django-app/.env")
_DEFAULT_TIMEOUT = 15


class CatalogError(Exception):
    def __init__(self, status_code, code, error, payload=None):
        self.status_code = status_code
        self.code = code
        self.error = error
        self.payload = payload
        super().__init__(error)


def _read_mdb_env():
    base_url = (os.environ.get("MDB_BASE_URL") or "").strip().rstrip("/")
    token = (os.environ.get("MDB_API_TOKEN") or "").strip()
    timeout_raw = (os.environ.get("MDB_TIMEOUT") or "").strip()
    try:
        timeout = int(timeout_raw) if timeout_raw else _DEFAULT_TIMEOUT
    except ValueError:
        timeout = _DEFAULT_TIMEOUT
    return base_url, token, timeout


def _workflow_backend_env_path():
    for parent in Path(__file__).resolve().parents:
        if parent.name == "workflow_backend":
            return parent / ".env"
    return None


def _load_mdb_dotenv():
    from dotenv import load_dotenv

    if _DOCKER_ENV_PATH.is_file():
        load_dotenv(_DOCKER_ENV_PATH, override=False)

    backend_env = _workflow_backend_env_path()
    if backend_env is not None and backend_env.is_file():
        load_dotenv(backend_env, override=False)


def get_mdb_config():
    base_url, token, timeout = _read_mdb_env()
    if not base_url or not token:
        _load_mdb_dotenv()
        base_url, token, timeout = _read_mdb_env()
    return base_url, token, timeout


def clamp_limit(value):
    if value is None or value == "":
        return LIMIT_DEFAULT
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CatalogError(
            400,
            "invalid_limit",
            "limit must be an integer",
        ) from exc
    return max(LIMIT_MIN, min(LIMIT_MAX, parsed))


def validate_source(source):
    if source is None:
        return None
    if isinstance(source, str):
        source = source.strip()
    if source == "":
        return None
    if source not in ALLOWED_SOURCES:
        raise CatalogError(400, "invalid_source", "Unknown source")
    return source


def validate_lookup_table(table):
    if table is None:
        return DEFAULT_LOOKUP_TABLE
    if isinstance(table, str):
        table = table.strip()
    if table == "":
        return DEFAULT_LOOKUP_TABLE
    if table not in ALLOWED_LOOKUP_TABLES:
        raise CatalogError(400, "invalid_table", "Unknown lookup table")
    return table


def _parse_json_body(response):
    try:
        return response.json()
    except ValueError:
        return None


def _mdb_error_text(body, fallback):
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, str) and error.strip():
            return error
    return fallback


def mdb_request(method, path, *, params=None, json_body=None):
    if path not in ALLOWED_MDB_PATHS:
        raise CatalogError(500, "catalog_internal", "Invalid catalog path")

    base_url, token, timeout = get_mdb_config()
    if not base_url or not token:
        raise CatalogError(
            503,
            "catalog_unconfigured",
            "Catalog service is not configured",
        )

    url = f"{base_url}{path}"
    headers = {"Authorization": f"Bearer {token}"}
    method = method.upper()
    request_kwargs = {"headers": headers, "params": params}
    if method == "POST":
        headers["Content-Type"] = "application/json"
        request_kwargs["json"] = json_body if json_body is not None else {}

    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.request(method, url, **request_kwargs)
    except httpx.RequestError:
        logger.warning("mdb request failed: %s %s (network)", method, path)
        raise CatalogError(
            503,
            "catalog_unavailable",
            "Catalog service is unavailable",
        ) from None

    status_code = response.status_code
    if 200 <= status_code < 300:
        body = _parse_json_body(response)
        if body is None:
            raise CatalogError(
                502,
                "catalog_unavailable",
                "Catalog service is unavailable",
            )
        return body

    body = _parse_json_body(response)
    payload = body if isinstance(body, dict) else None

    if status_code in (401, 403):
        raise CatalogError(
            502,
            "catalog_auth",
            "Catalog authentication failed",
            payload=payload,
        )
    if status_code == 503:
        raise CatalogError(
            503,
            "catalog_unavailable",
            "Catalog service is unavailable",
            payload=payload,
        )
    if status_code in (400, 404):
        fallback = "Bad request" if status_code == 400 else "Not found"
        code = "catalog_bad_request" if status_code == 400 else "catalog_not_found"
        raise CatalogError(
            status_code,
            code,
            _mdb_error_text(body, fallback),
            payload=payload,
        )
    if status_code >= 500:
        raise CatalogError(
            502,
            "catalog_unavailable",
            "Catalog service is unavailable",
            payload=payload,
        )

    fallback = "Catalog request failed"
    raise CatalogError(
        status_code,
        "catalog_error",
        _mdb_error_text(body, fallback),
        payload=payload,
    )
