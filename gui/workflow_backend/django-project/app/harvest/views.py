"""OAI-PMH proxy: workflow kernel -> backend -> external repository.

The repository address and API key live only in the backend environment
(``OAI_PMH_*`` in ``gui/workflow_backend/.env``). Kernels authenticate with the
shared service token (``X-Api-Key``, the same token the Anthropic proxy accepts)
and can only issue allowlisted OAI-PMH verbs/arguments or download a file by id.

This is deliberately not a generic reverse proxy: nothing from the client
request reaches the repository except the validated query arguments, and the
upstream body is relayed unchanged (OAI-PMH ``<error>`` elements arrive with
HTTP 200 and are interpreted by the client, ``neuroworkflow.utils.oai_pmh``).
"""

import logging
import os

import httpx
from django.http import JsonResponse, StreamingHttpResponse
from django.views import View

logger = logging.getLogger(__name__)

ALLOWED_VERBS = {
    "Identify",
    "ListMetadataFormats",
    "ListSets",
    "ListIdentifiers",
    "ListRecords",
    "GetRecord",
}
ALLOWED_ARGS = {
    "metadataPrefix",
    "set",
    "from",
    "until",
    "identifier",
    "resumptionToken",
}
DEFAULT_KEY_HEADER = "X-MDRS-API-Key"
DEFAULT_TIMEOUT = 60.0


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


class OAIPMHProxyView(View):
    """``GET /api/harvest/oai/?verb=...`` — allowlisted OAI-PMH passthrough."""

    def get(self, request):
        denied = _service_token_error(request)
        if denied:
            return denied
        base = os.environ.get("OAI_PMH_BASE_URL", "").rstrip("/")
        if not base:
            return JsonResponse(
                {"error": "OAI_PMH_BASE_URL is not configured on the backend"},
                status=500,
            )
        verb = request.GET.get("verb")
        if verb not in ALLOWED_VERBS:
            return JsonResponse({"error": "Unsupported or missing verb"}, status=400)
        extra = set(request.GET) - ALLOWED_ARGS - {"verb"}
        if extra:
            return JsonResponse(
                {"error": f"Unsupported arguments: {sorted(extra)}"}, status=400
            )
        params = {"verb": verb}
        params.update({k: request.GET[k] for k in ALLOWED_ARGS if k in request.GET})
        return _relay(f"{base}/", params=params)


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
