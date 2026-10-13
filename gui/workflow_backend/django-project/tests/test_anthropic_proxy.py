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


def _llm_url(subpath="v1/messages"):
    return reverse("chat-llm-proxy", kwargs={"subpath": subpath})


def _post_llm(client, model, subpath="v1/messages"):
    return client.post(
        _llm_url(subpath),
        data='{"model":"%s"}' % model,
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )


def test_llm_route_picks_the_provider_from_the_model(
    client, service_token, monkeypatch
):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    monkeypatch.delenv("MINIMAX_ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)

    resp = _post_llm(client, "MiniMax-M3", "v1/messages/count_tokens")

    assert resp.status_code == 200
    captured = _FakeHttpxClient.captured
    assert captured["url"] == (
        "https://api.minimax.io/anthropic/v1/messages/count_tokens"
    )
    assert captured["headers"]["authorization"] == "Bearer mm-real-key"
    # Neither the service token nor the Anthropic key goes to the provider.
    assert "x-api-key" not in {k.lower() for k in captured["headers"]}
    assert b"message_start" in b"".join(resp.streaming_content)


def test_former_minimax_route_still_serves_running_kernels(
    client, service_token, monkeypatch
):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)

    resp = client.post(
        "/api/chat/minimax/v1/messages",
        data='{"model":"MiniMax-M3"}',
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )

    assert resp.status_code == 200
    assert _FakeHttpxClient.captured["url"] == (
        "https://api.minimax.io/anthropic/v1/messages"
    )
    assert _FakeHttpxClient.captured["headers"]["authorization"] == (
        "Bearer mm-real-key"
    )


def test_llm_route_uses_a_configured_provider(
    client, service_token, monkeypatch, tmp_path
):
    providers_file = tmp_path / "llm_providers.json"
    providers_file.write_text(
        """{"providers": [
            {"name": "deepseek", "base_url": "https://ds.example/v1",
             "api_key_env": "DS_KEY", "models": ["ds-chat"],
             "anthropic_base_url": "https://ds.example/anthropic/"},
            {"name": "openonly", "base_url": "https://oo.example/v1",
             "api_key_env": "OO_KEY", "models": ["oo-chat"]}
        ]}"""
    )
    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(providers_file))
    monkeypatch.setenv("DS_KEY", "ds-real-key")
    monkeypatch.setenv("OO_KEY", "oo-real-key")
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)

    resp = _post_llm(client, "ds-chat")
    assert resp.status_code == 200
    captured = _FakeHttpxClient.captured
    assert captured["url"] == "https://ds.example/anthropic/v1/messages"
    assert captured["headers"]["authorization"] == "Bearer ds-real-key"

    # A provider without an Anthropic-compatible API is not reachable.
    assert _post_llm(client, "oo-chat").status_code == 400


@pytest.mark.parametrize("body", ['{"model":"claude-opus"}', "{}", "not json"])
def test_llm_route_rejects_models_not_on_offer(
    client, service_token, monkeypatch, body
):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)
    _FakeHttpxClient.captured = {}

    resp = client.post(
        _llm_url(),
        data=body,
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )

    assert resp.status_code == 400
    assert _FakeHttpxClient.captured == {}


def test_llm_route_requires_the_service_token(client, service_token, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    resp = client.post(
        _llm_url(), data='{"model":"MiniMax-M3"}', content_type="application/json"
    )
    assert resp.status_code == 401


def test_llm_route_only_serves_text_generation(client, service_token, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    resp = client.post(
        "/api/chat/llm/v1/files",
        data="{}",
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
    )
    assert resp.status_code == 404


def test_llm_route_rejects_provider_without_key(client, service_token, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    assert _post_llm(client, "MiniMax-M3").status_code == 400


def test_proxy_errors_when_backend_key_missing(client, monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "svc-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post(
        _url(), data="{}", content_type="application/json", HTTP_X_API_KEY="svc-token"
    )
    assert resp.status_code == 500
