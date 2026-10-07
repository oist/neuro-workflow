"""Tests for the Anthropic passthrough proxy (in-kernel Claude agent).

The proxy lets the notebook kernel reach Anthropic without holding the real key:
the kernel authenticates with the shared service token (as ``x-api-key``) and the
backend swaps in the real ``ANTHROPIC_API_KEY`` before forwarding upstream.
"""
import pytest
from django.urls import reverse


def _url(subpath="v1/messages"):
    return reverse("chat-anthropic-proxy", kwargs={"subpath": subpath})


class _FakeUpstream:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def iter_bytes(self):
        yield b"event: message_start\ndata: {}\n\n"

    def close(self):
        pass


class _FakeHttpxClient:
    """Records the outgoing request and returns a canned streaming response."""

    captured: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def build_request(self, method, url, headers=None, content=None):
        _FakeHttpxClient.captured = {
            "method": method,
            "url": url,
            "headers": headers,
            "content": content,
        }
        return ("request", headers)

    def send(self, request, stream=False):
        return _FakeUpstream()

    def close(self):
        pass


@pytest.fixture
def service_token(monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "svc-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real-key")
    return "svc-token"


def test_proxy_rejects_missing_token(client, service_token):
    resp = client.post(_url(), data="{}", content_type="application/json")
    assert resp.status_code == 401


def test_proxy_rejects_wrong_token(client, service_token):
    resp = client.post(
        _url(), data="{}", content_type="application/json", HTTP_X_API_KEY="nope"
    )
    assert resp.status_code == 401


def test_proxy_injects_real_key_and_streams(client, service_token, monkeypatch):
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)

    resp = client.post(
        _url(),
        data='{"model":"claude"}',
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )

    assert resp.status_code == 200
    # The service token is swapped for the real Anthropic key upstream.
    assert _FakeHttpxClient.captured["headers"]["x-api-key"] == "sk-real-key"
    assert _FakeHttpxClient.captured["url"].endswith("/v1/messages")
    body = b"".join(resp.streaming_content)
    assert b"message_start" in body


def _minimax_url(subpath="v1/messages"):
    return reverse("chat-minimax-proxy", kwargs={"subpath": subpath})


def test_minimax_route_injects_minimax_key_as_bearer(
    client, service_token, monkeypatch
):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    monkeypatch.delenv("MINIMAX_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)

    resp = client.post(
        _minimax_url("v1/messages/count_tokens"),
        data='{"model":"MiniMax-M3"}',
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )

    assert resp.status_code == 200
    captured = _FakeHttpxClient.captured
    assert captured["url"] == (
        "https://api.minimax.io/anthropic/v1/messages/count_tokens"
    )
    assert captured["headers"]["authorization"] == "Bearer mm-real-key"
    # Neither the service token nor the Anthropic key goes to MiniMax.
    assert "x-api-key" not in {k.lower() for k in captured["headers"]}
    assert b"message_start" in b"".join(resp.streaming_content)


def test_minimax_route_only_serves_text_generation(client, service_token, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    resp = client.post(
        "/api/chat/minimax/v1/files",
        data="{}",
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )
    assert resp.status_code == 404


def test_minimax_route_errors_when_backend_key_missing(
    client, service_token, monkeypatch
):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    resp = client.post(
        _minimax_url(),
        data="{}",
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )
    assert resp.status_code == 500
    assert "MINIMAX_API_KEY" in resp.json()["error"]


def test_proxy_errors_when_backend_key_missing(client, monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "svc-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post(
        _url(), data="{}", content_type="application/json", HTTP_X_API_KEY="svc-token"
    )
    assert resp.status_code == 500
