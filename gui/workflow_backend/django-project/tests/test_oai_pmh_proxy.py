"""Tests for the OAI-PMH file download proxy (workflow kernel -> backend -> repository).

Kernels authenticate with the shared service token (``X-Api-Key``); the backend
attaches the repository API key from its own environment and streams the file
back unchanged. (Record metadata is no longer proxied — kernels read it from the
harvested store, see ``test_harvest_records_api.py``.)
"""

import pytest
from django.urls import reverse

FILE_ID = "ea77ccb8-8414-4a74-85c5-7b8b8f7bf4d6"


def _file_url():
    return reverse("harvest-oai-file-download", kwargs={"file_id": FILE_ID})


class _FakeUpstream:
    def __init__(self, body=b"<OAI-PMH/>", status_code=200, headers=None):
        self.body = body
        self.status_code = status_code
        self.headers = (
            headers
            if headers is not None
            else {"content-type": "text/xml; charset=UTF-8"}
        )

    def iter_bytes(self):
        yield self.body

    def close(self):
        pass


class _FakeHttpxClient:
    """Records the outgoing request and returns a canned streaming response."""

    captured: dict = {}
    upstream = None

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, headers=None, params=None, content=None):
        _FakeHttpxClient.captured = {
            "method": method,
            "url": url,
            "headers": headers,
            "params": params,
        }
        return ("request", url)

    def send(self, request, stream=False):
        return _FakeHttpxClient.upstream or _FakeUpstream()

    def close(self):
        pass


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "svc-token")
    monkeypatch.setenv("OAI_PMH_API_KEY", "secret-key")
    monkeypatch.setenv(
        "OAI_PMH_FILE_DOWNLOAD_URL",
        "https://repo.example/api/v3/files/{file_id}/download/",
    )
    monkeypatch.delenv("OAI_PMH_API_KEY_HEADER", raising=False)
    monkeypatch.setattr("app.harvest.views.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.captured = {}
    _FakeHttpxClient.upstream = None
    return "svc-token"


def test_download_streams_file_with_key(client, configured):
    _FakeHttpxClient.upstream = _FakeUpstream(
        body=b"data-bytes",
        headers={"content-type": "application/octet-stream", "content-length": "10"},
    )

    resp = client.get(_file_url(), HTTP_X_API_KEY="svc-token")

    assert resp.status_code == 200
    captured = _FakeHttpxClient.captured
    assert captured["url"] == f"https://repo.example/api/v3/files/{FILE_ID}/download/"
    # Repository key attached; the kernel's service token is never forwarded.
    assert captured["headers"]["X-MDRS-API-Key"] == "secret-key"
    assert "x-api-key" not in {k.lower() for k in captured["headers"]}
    assert b"".join(resp.streaming_content) == b"data-bytes"
    assert resp["Content-Length"] == "10"


def test_key_header_name_is_configurable(client, configured, monkeypatch):
    monkeypatch.setenv("OAI_PMH_API_KEY_HEADER", "X-Other-Key")
    resp = client.get(_file_url(), HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 200
    assert _FakeHttpxClient.captured["headers"]["X-Other-Key"] == "secret-key"


def test_key_omitted_when_not_configured(client, configured, monkeypatch):
    monkeypatch.delenv("OAI_PMH_API_KEY")
    resp = client.get(_file_url(), HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 200
    assert "X-MDRS-API-Key" not in _FakeHttpxClient.captured["headers"]


def test_download_relays_upstream_status_and_retry_after(client, configured):
    _FakeHttpxClient.upstream = _FakeUpstream(
        body=b"busy",
        status_code=503,
        headers={"content-type": "text/plain", "retry-after": "7"},
    )
    resp = client.get(_file_url(), HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 503
    assert resp["Retry-After"] == "7"


def test_download_defaults_to_octet_stream_without_upstream_type(client, configured):
    _FakeHttpxClient.upstream = _FakeUpstream(body=b"raw", headers={})
    resp = client.get(_file_url(), HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 200
    assert resp["Content-Type"] == "application/octet-stream"


def test_download_requires_token(client, configured):
    assert client.get(_file_url()).status_code == 401
    assert _FakeHttpxClient.captured == {}


def test_download_rejects_wrong_token(client, configured):
    resp = client.get(_file_url(), HTTP_X_API_KEY="nope")
    assert resp.status_code == 401


def test_download_errors_when_template_missing(client, configured, monkeypatch):
    monkeypatch.delenv("OAI_PMH_FILE_DOWNLOAD_URL")
    resp = client.get(_file_url(), HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 500


def test_download_rejects_non_uuid_id(client, configured):
    resp = client.get(
        "/api/harvest/oai/files/not-a-uuid/download/", HTTP_X_API_KEY="svc-token"
    )
    assert resp.status_code == 404
    assert _FakeHttpxClient.captured == {}
