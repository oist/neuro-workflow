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
    providers_file.write_text("""{"providers": [
            {"name": "deepseek", "base_url": "https://ds.example/v1",
             "api_key_env": "DS_KEY", "models": ["ds-chat"],
             "anthropic_base_url": "https://ds.example/anthropic/"},
            {"name": "openonly", "base_url": "https://oo.example/v1",
             "api_key_env": "OO_KEY", "models": ["oo-chat"]}
        ]}""")
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


def _saved_keys(user):
    from app.chat.models import PersonalAgentCredential
    from app.chat.personal_agent import encrypt_key

    PersonalAgentCredential.objects.create(
        user=user,
        anthropic_key=encrypt_key("sk-ant-personal"),
        openai_key=encrypt_key("sk-codex-personal"),
    )


def _open_notebook(user, hub_user="internal"):
    from datetime import timedelta

    from app.chat.models import NotebookToken
    from app.workflow.models import FlowProject
    from django.utils import timezone

    project = FlowProject.objects.create(name="Open notebook", owner=user)
    NotebookToken.objects.create(
        project=project,
        user=user,
        hub_user=hub_user,
        access_token="relayed",
        expires_at=timezone.now() + timedelta(minutes=5),
    )
    return project


def _post_anthropic(client, project_id=None, hub_token=None):
    headers = {"HTTP_X_API_KEY": "svc-token"}
    if project_id is not None:
        headers["HTTP_X_NW_PROJECT_ID"] = str(project_id)
    if hub_token is not None:
        headers["HTTP_X_JUPYTERHUB_TOKEN"] = hub_token
    return client.post(
        _url(),
        data='{"model":"claude"}',
        content_type="application/json",
        **headers,
    )


def _upstream_names(captured):
    return {key.lower() for key in captured["headers"]}


@pytest.mark.django_db
def test_proxy_uses_the_personal_claude_key_for_the_open_notebook(
    client, service_token, monkeypatch, user_alice
):
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)
    monkeypatch.setattr("app.chat.views._kernel_hub_user", lambda request: "internal")
    project = _open_notebook(user_alice)
    _saved_keys(user_alice)

    resp = _post_anthropic(client, project.id, "hub-token")

    assert resp.status_code == 200
    captured = _FakeHttpxClient.captured
    assert captured["headers"]["x-api-key"] == "sk-ant-personal"
    names = _upstream_names(captured)
    assert "x-nw-project-id" not in names
    assert "x-jupyterhub-token" not in names


@pytest.mark.django_db
def test_proxy_keeps_the_shared_key_without_a_matching_notebook(
    client, service_token, monkeypatch, user_alice
):
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)
    monkeypatch.setattr("app.chat.views._kernel_hub_user", lambda request: "external")
    project = _open_notebook(user_alice, hub_user="internal")
    _saved_keys(user_alice)

    missing = _post_anthropic(client)
    assert missing.status_code == 200
    assert _FakeHttpxClient.captured["headers"]["x-api-key"] == "sk-real-key"

    other_space = _post_anthropic(client, project.id, "hub-token")
    assert other_space.status_code == 200
    assert _FakeHttpxClient.captured["headers"]["x-api-key"] == "sk-real-key"


@pytest.mark.django_db
def test_llm_route_uses_the_personal_codex_key_for_an_openai_provider(
    client, service_token, monkeypatch, user_alice, tmp_path
):
    providers_file = tmp_path / "llm_providers.json"
    providers_file.write_text("""{"providers": [
            {"name": "openai", "base_url": "https://api.openai.com/v1",
             "api_key_env": "OPENAI_LAB_KEY", "models": ["gpt-codex"],
             "anthropic_base_url": "https://api.openai.com/anthropic"},
            {"name": "minimax", "base_url": "https://api.minimax.io/v1",
             "api_key_env": "MINIMAX_API_KEY", "models": ["MiniMax-M3"],
             "anthropic_base_url": "https://api.minimax.io/anthropic"}
        ]}""")
    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(providers_file))
    monkeypatch.setenv("OPENAI_LAB_KEY", "shared-openai")
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-real-key")
    monkeypatch.setattr("app.chat.views.httpx.Client", _FakeHttpxClient)
    monkeypatch.setattr("app.chat.views._kernel_hub_user", lambda request: "internal")
    project = _open_notebook(user_alice)
    _saved_keys(user_alice)

    codex = client.post(
        _llm_url(),
        data='{"model":"gpt-codex"}',
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
        HTTP_X_NW_PROJECT_ID=str(project.id),
        HTTP_X_JUPYTERHUB_TOKEN="hub-token",
    )
    assert codex.status_code == 200
    captured = _FakeHttpxClient.captured
    assert captured["headers"]["authorization"] == "Bearer sk-codex-personal"
    names = _upstream_names(captured)
    assert "x-nw-project-id" not in names
    assert "x-jupyterhub-token" not in names

    minimax = client.post(
        _llm_url(),
        data='{"model":"MiniMax-M3"}',
        content_type="application/json",
        HTTP_X_API_KEY="svc-token",
        HTTP_X_NW_PROJECT_ID=str(project.id),
        HTTP_X_JUPYTERHUB_TOKEN="hub-token",
    )
    assert minimax.status_code == 200
    assert _FakeHttpxClient.captured["headers"]["authorization"] == "Bearer mm-real-key"
