"""OAI-PMH client used by the ``database`` nodes and the backend harvester.

Stdlib only (``urllib`` + ``xml.etree``) so it runs unchanged inside the
JupyterHub kernel image without extra dependencies.

Two access paths:

* **Direct mode** (backend harvester / CLI): when ``OAI_PMH_BASE_URL`` is set
  in the process environment :class:`OAIPMHClient` calls the repository
  directly, with the optional ``OAI_PMH_API_KEY`` sent in the
  ``OAI_PMH_API_KEY_HEADER`` header.
* **Kernel mode**: kernels never talk to the repository. Records come from the
  backend's harvested copy via :func:`fetch_backend_records`, and files are
  streamed through the backend download proxy
  (``/api/harvest/oai/files/{id}/download/``) with the shared service token.

Responses are parsed into plain dicts; the ``mdrs`` payload of the RIKEN MDRS
repository is normalised into a folder dict with a ``files`` list, any other
payload (e.g. ``oai_dc``) becomes ``{local_tag: [values]}``.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

OAI_NS = "http://www.openarchives.org/OAI/2.0/"
MDRS_NS = "https://www.ni.riken.jp/oai/mdrs/"

DEFAULT_KEY_HEADER = "X-MDRS-API-Key"
DEFAULT_BACKEND_URL = "http://backend:3000"
USER_AGENT = "neuroworkflow-oai-pmh (+https://github.com/oist/neuro-workflow)"

_CHUNK_SIZE = 1024 * 1024
_DEFAULT_RETRY_AFTER = 5
_MAX_RETRY_AFTER = 60

_MDRS_FOLDER_FIELDS = (
    "id",
    "name",
    "description",
    "access_level",
    "laboratory_name",
    "created_at",
    "updated_at",
    "path",
    "size",
)
_MDRS_FILE_FIELDS = (
    "id",
    "name",
    "description",
    "type",
    "mime_type",
    "size",
    "created_at",
    "updated_at",
)


class OAIPMHError(Exception):
    """Protocol (``<error>``), HTTP, or transport failure.

    ``code`` is the OAI-PMH ``errorCode`` (e.g. ``badAuthentication``,
    ``noRecordsMatch``), ``http_<status>``, ``bad_response`` or ``transport``.
    """

    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


def resolve_endpoint() -> Tuple[str, str, Dict[str, str]]:
    """Return ``(oai_url, file_url_template, headers)`` for this process.

    Direct mode when ``OAI_PMH_BASE_URL`` is set; otherwise the backend is the
    peer, where only ``file_url_template`` is served — kernels fetch records
    with :func:`fetch_backend_records` instead of OAI-PMH verbs.
    """
    base = os.environ.get("OAI_PMH_BASE_URL", "").rstrip("/")
    if base:
        headers: Dict[str, str] = {}
        key = os.environ.get("OAI_PMH_API_KEY", "")
        if key:
            headers[os.environ.get("OAI_PMH_API_KEY_HEADER") or DEFAULT_KEY_HEADER] = (
                key
            )
        return base + "/", os.environ.get("OAI_PMH_FILE_DOWNLOAD_URL", ""), headers

    backend = os.environ.get("NEUROWORKFLOW_BACKEND_URL", DEFAULT_BACKEND_URL)
    backend = backend.rstrip("/")
    token = os.environ.get("NEUROWORKFLOW_SERVICE_TOKEN") or os.environ.get(
        "JUPYTERHUB_API_TOKEN", ""
    )
    return (
        f"{backend}/api/harvest/oai/",
        f"{backend}/api/harvest/oai/files/{{file_id}}/download/",
        {"X-Api-Key": token},
    )


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


def _int_or_str(value: str) -> Any:
    return int(value) if value.isdigit() else value


def parse_response(xml_bytes: bytes) -> ET.Element:
    """Parse an OAI-PMH response body, raising :class:`OAIPMHError` on ``<error>``.

    Repositories may return protocol errors with HTTP 200, so the error element
    is the authoritative signal. A body that is not XML (e.g. a JSON error from
    the proxy) raises ``bad_response`` with the start of the body in the message.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        snippet = xml_bytes[:200].decode("utf-8", "replace").strip()
        raise OAIPMHError(
            f"Unparseable OAI-PMH response: {snippet}", "bad_response"
        ) from e
    error = root.find(f"{{{OAI_NS}}}error")
    if error is not None:
        # The spec uses ``code``; the RIKEN MDRS repository uses ``errorCode``.
        code = error.get("code") or error.get("errorCode") or "error"
        raise OAIPMHError(_text(error) or code, code)
    return root


def _generic_metadata(payload: ET.Element) -> Dict[str, List[str]]:
    """Flatten a metadata payload to ``{local_tag: [text, ...]}`` (oai_dc etc.)."""
    out: Dict[str, List[str]] = {}
    for child in payload:
        out.setdefault(_local(child.tag), []).append(_text(child))
    return out


def _mdrs_metadata(payload: ET.Element) -> Dict[str, Any]:
    """Normalise an ``<mdrs><folder>`` payload into a folder dict with ``files``."""
    ns = f"{{{MDRS_NS}}}"
    folder = payload.find(ns + "folder")
    if folder is None:
        return _generic_metadata(payload)

    out: Dict[str, Any] = {f: _text(folder.find(ns + f)) for f in _MDRS_FOLDER_FIELDS}
    out["size"] = _int_or_str(out["size"])
    parent = folder.find(ns + "parent")
    out["parent"] = (
        {"id": _text(parent.find(ns + "id")), "name": _text(parent.find(ns + "name"))}
        if parent is not None
        else None
    )
    raw = _text(folder.find(ns + "metadata"))
    try:
        out["metadata"] = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        out["metadata"] = raw
    files = []
    for file_el in folder.findall(f"{ns}files/{ns}file"):
        entry = {f: _text(file_el.find(ns + f)) for f in _MDRS_FILE_FIELDS}
        entry["size"] = _int_or_str(entry["size"])
        files.append(entry)
    out["files"] = files
    return out


def parse_record(el: ET.Element, metadata_prefix: str = "") -> Dict[str, Any]:
    """Convert one ``<record>`` element into a plain dict."""
    ns = f"{{{OAI_NS}}}"
    header = el.find(ns + "header")
    record: Dict[str, Any] = {
        "identifier": (
            _text(header.find(ns + "identifier")) if header is not None else ""
        ),
        "datestamp": _text(header.find(ns + "datestamp")) if header is not None else "",
        "set_specs": (
            [_text(s) for s in header.findall(ns + "setSpec")]
            if header is not None
            else []
        ),
        "deleted": header is not None and header.get("status") == "deleted",
        "metadata_prefix": metadata_prefix,
        "metadata": None,
        "files": [],
    }
    metadata = el.find(ns + "metadata")
    payload = next(iter(metadata), None) if metadata is not None else None
    if payload is None:
        return record
    if payload.tag == f"{{{MDRS_NS}}}mdrs":
        record["metadata"] = _mdrs_metadata(payload)
        record["files"] = [
            {
                "id": f["id"],
                "name": f["name"],
                "mime_type": f["mime_type"],
                "size": f["size"],
            }
            for f in record["metadata"].get("files", [])
        ]
    else:
        record["metadata"] = _generic_metadata(payload)
    return record


def parse_records(
    root: ET.Element, metadata_prefix: str = ""
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[int]]:
    """Return ``(records, resumption_token, complete_list_size)`` for a response."""
    ns = f"{{{OAI_NS}}}"
    records = [parse_record(el, metadata_prefix) for el in root.iter(ns + "record")]
    token_el = root.find(f".//{ns}resumptionToken")
    token = _text(token_el) or None
    total = None
    if token_el is not None:
        size = token_el.get("completeListSize") or ""
        total = int(size) if size.isdigit() else None
    return records, token, total


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _envelope(
    records: List[Dict[str, Any]],
    total: Optional[int],
    error: Optional[OAIPMHError] = None,
) -> Dict[str, Any]:
    return {
        "status": "error" if error else "success",
        "records": records,
        "count": len(records),
        "total": total,
        "error": str(error) if error else None,
        "error_code": error.code if error else None,
    }


def _retry_after(value: Optional[str]) -> float:
    try:
        return min(max(float(value or ""), 0.0), _MAX_RETRY_AFTER)
    except ValueError:
        return _DEFAULT_RETRY_AFTER


class OAIPMHClient:
    """Minimal OAI-PMH harvester; list/get methods never raise (envelope API)."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.oai_url, self.file_url_template, self.headers = resolve_endpoint()

    def _open(self, url: str):
        headers = {
            **self.headers,
            "Accept": "text/xml, application/xml, */*",
            "User-Agent": USER_AGENT,
        }
        req = urllib.request.Request(url, headers=headers)
        for attempt in range(self.max_retries + 1):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                detail = e.read(200).decode("utf-8", "replace").strip()
                error = OAIPMHError(
                    f"HTTP {e.code}: {detail or e.reason}", f"http_{e.code}"
                )
                if e.code != 503 or attempt >= self.max_retries:
                    raise error from e
                time.sleep(_retry_after(e.headers.get("Retry-After")))
            except OSError as e:  # URLError, socket timeouts
                reason = getattr(e, "reason", e)
                error = OAIPMHError(f"Connection failed: {reason}", "transport")
                if attempt >= self.max_retries:
                    raise error from e
                time.sleep(_DEFAULT_RETRY_AFTER)
        raise OAIPMHError("retries exhausted", "transport")  # pragma: no cover

    def request(self, verb: str, **args: str) -> ET.Element:
        """Issue one verb (empty argument values are omitted) and parse the reply."""
        params = {"verb": verb}
        params.update({k: v for k, v in args.items() if v})
        url = self.oai_url + "?" + urllib.parse.urlencode(params)
        with self._open(url) as resp:
            try:
                body = resp.read()
            except OSError as e:  # stalled body read
                raise OAIPMHError(f"Read failed: {e}", "transport") from e
        return parse_response(body)

    def list_records(
        self,
        metadata_prefix: str = "mdrs",
        set_spec: str = "",
        from_date: str = "",
        until_date: str = "",
        max_records: int = 100,
    ) -> Dict[str, Any]:
        """``ListRecords`` following resumption tokens until ``max_records``."""
        records: List[Dict[str, Any]] = []
        total: Optional[int] = None
        args: Dict[str, str] = {
            "metadataPrefix": metadata_prefix,
            "set": set_spec,
            "from": from_date,
            "until": until_date,
        }
        try:
            while True:
                root = self.request("ListRecords", **args)
                page, token, page_total = parse_records(root, metadata_prefix)
                records.extend(page)
                total = page_total if page_total is not None else total
                if not token or len(records) >= max_records:
                    break
                args = {"resumptionToken": token}
        except OAIPMHError as e:
            if e.code != "noRecordsMatch":
                return _envelope(records[:max_records], total, error=e)
        return _envelope(records[:max_records], total)

    def get_record(
        self, identifier: str, metadata_prefix: str = "mdrs"
    ) -> Dict[str, Any]:
        """``GetRecord`` for one identifier (envelope with zero or one record)."""
        try:
            root = self.request(
                "GetRecord", identifier=identifier, metadataPrefix=metadata_prefix
            )
        except OAIPMHError as e:
            return _envelope([], None, error=e)
        records, _, _ = parse_records(root, metadata_prefix)
        return _envelope(records, len(records))

    def download_file(self, file_id: str, dest_path: str) -> str:
        """Stream one repository file to ``dest_path`` (via ``.part``); raises on failure."""
        if not self.file_url_template:
            raise OAIPMHError(
                "File download URL is not configured (OAI_PMH_FILE_DOWNLOAD_URL)",
                "not_configured",
            )
        url = self.file_url_template.format(
            file_id=urllib.parse.quote(str(file_id), safe="")
        )
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        part = dest_path + ".part"
        try:
            with self._open(url) as resp, open(part, "wb") as fh:
                for chunk in iter(lambda: resp.read(_CHUNK_SIZE), b""):
                    fh.write(chunk)
            os.replace(part, dest_path)
        except OSError as e:
            raise OAIPMHError(f"Download failed for {file_id}: {e}", "transport") from e
        finally:
            if os.path.exists(part):
                os.remove(part)
        return dest_path


# ---------------------------------------------------------------------------
# Backend record store (kernel side)
# ---------------------------------------------------------------------------


def fetch_backend_records(
    identifiers: List[str], timeout: float = 30.0
) -> Dict[str, Any]:
    """Fetch harvested records from the backend by identifier (envelope API).

    The backend keeps a local copy of the repository (``manage.py
    harvest_oai``); kernels read it through ``/api/harvest/records/`` with the
    shared service token. ``total`` is the number of requested identifiers;
    identifiers missing from the copy are reported in ``error`` with code
    ``not_found`` while the found records are still returned.
    """
    backend = os.environ.get("NEUROWORKFLOW_BACKEND_URL", DEFAULT_BACKEND_URL)
    token = os.environ.get("NEUROWORKFLOW_SERVICE_TOKEN") or os.environ.get(
        "JUPYTERHUB_API_TOKEN", ""
    )
    url = (
        backend.rstrip("/")
        + "/api/harvest/records/?"
        + urllib.parse.urlencode({"identifiers": ",".join(identifiers)})
    )
    req = urllib.request.Request(
        url,
        headers={
            "X-Api-Key": token,
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read(200).decode("utf-8", "replace").strip()
        error = OAIPMHError(f"HTTP {e.code}: {detail or e.reason}", f"http_{e.code}")
        return _envelope([], len(identifiers), error)
    except OSError as e:  # URLError, socket timeouts
        reason = getattr(e, "reason", e)
        error = OAIPMHError(f"Connection failed: {reason}", "transport")
        return _envelope([], len(identifiers), error)
    except ValueError as e:  # JSON decode
        error = OAIPMHError(f"Bad backend response: {e}", "bad_response")
        return _envelope([], len(identifiers), error)
    records = payload.get("records") or []
    missing = payload.get("missing") or []
    error = (
        OAIPMHError("not found: " + ", ".join(missing), "not_found")
        if missing
        else None
    )
    return _envelope(records, len(identifiers), error)
