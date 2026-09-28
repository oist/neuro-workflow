"""Tests for the notebook token relay (in-kernel Claude agent workflow tools).

The browser relays its Keycloak access token per project; the kernel-side MCP
proxies authenticate with the service token, the kernel's own JupyterHub token
(verified with the hub to learn its Jupyter space) and ``project_id``, and the
backend forwards the relayed token to the MCP server. The kernel never sees it.
"""

import time
from datetime import timedelta

import jwt
import pytest
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from app.chat.models import NotebookToken
from app.workflow.models import FlowProject

pytestmark = pytest.mark.django_db

RELAY_URL = reverse("chat-notebook-token")
TOOLS_URL = reverse("chat-notebook-mcp-tools")
CALL_URL = reverse("chat-notebook-mcp-call")


def _jwt(exp_delta=300, sub="alice-sub-uuid"):
    return jwt.encode(
        {"exp": int(time.time()) + exp_delta, "sub": sub}, "k", algorithm="HS256"
    )


def _project(owner, *, visibility="private"):
    return FlowProject.objects.create(name="P", owner=owner, visibility=visibility)


def _relay(client, project, token):
    return client.post(
        RELAY_URL,
        {"project_id": str(project.id)},
        format="json",
        HTTP_AUTHORIZATION=f"Bearer {token}",
    )


class _FakeMCP:
    """Records the auth token the proxy forwards to the MCP server."""

    tokens: list = []

    def __init__(self, auth_token=None):
        _FakeMCP.tokens.append(auth_token)

    async def initialize(self):
        return {}

    async def list_tools(self):
        return [
            {
                "name": "get_flow",
                "description": "Get flow",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

    async def call_tool(self, name, arguments):
        return f"called {name} with {arguments}"


@pytest.fixture
def fake_mcp(monkeypatch):
    _FakeMCP.tokens = []
    monkeypatch.setattr("app.chat.views.MCPClient", _FakeMCP)
    return _FakeMCP


class _FakeHubResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


# Per-server hub tokens the fake JupyterHub recognises -> hub user.
_HUB_TOKENS = {"hub-internal": "internal", "hub-hackathon": "hackathon"}


@pytest.fixture
def service_token(monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "svc-token")

    def fake_hub_get(url, headers=None, timeout=None):
        assert url.endswith("/hub/api/user")
        token = (headers or {}).get("Authorization", "").removeprefix("token ")
        if token in _HUB_TOKENS:
            return _FakeHubResponse(200, {"name": _HUB_TOKENS[token]})
        return _FakeHubResponse(403, {"message": "Forbidden"})

    monkeypatch.setattr("app.chat.views.httpx.get", fake_hub_get)
    return "svc-token"


# Headers a kernel in the project space sends on the MCP proxies.
KERNEL = {"HTTP_X_API_KEY": "svc-token", "HTTP_X_JUPYTERHUB_TOKEN": "hub-internal"}


# ---------------------------------------------------------------------------
# POST /api/chat/notebook-token/ (browser)
# ---------------------------------------------------------------------------


def test_relay_rejects_anonymous(user_alice):
    project = _project(user_alice)
    resp = APIClient().post(RELAY_URL, {"project_id": str(project.id)}, format="json")
    assert resp.status_code == 401


def test_relay_requires_project_id(auth_client, user_alice):
    resp = auth_client(user_alice).post(
        RELAY_URL, {}, format="json", HTTP_AUTHORIZATION=f"Bearer {_jwt()}"
    )
    assert resp.status_code == 400


def test_relay_stores_own_token_for_own_project(auth_client, user_alice):
    project = _project(user_alice)
    token = _jwt(exp_delta=300)

    resp = _relay(auth_client(user_alice), project, token)

    assert resp.status_code == 200
    stored = NotebookToken.objects.get(project=project)
    assert stored.access_token == token
    assert stored.user == user_alice
    assert stored.hub_user == "internal"  # default (project) tenant
    assert (
        int(stored.expires_at.timestamp())
        == jwt.decode(token, options={"verify_signature": False})["exp"]
    )


def test_relay_respects_project_visibility(auth_client, user_alice, user_bob):
    private = _project(user_alice, visibility="private")
    public = _project(user_alice, visibility="public")
    client = auth_client(user_bob)

    assert _relay(client, private, _jwt(sub="bob")).status_code == 404
    assert _relay(client, public, _jwt(sub="bob")).status_code == 200
    assert NotebookToken.objects.get(project=public).user == user_bob


def test_relay_upserts_per_project(auth_client, user_alice):
    project = _project(user_alice)
    client = auth_client(user_alice)
    first, second = _jwt(exp_delta=100), _jwt(exp_delta=200)

    _relay(client, project, first)
    _relay(client, project, second)

    assert NotebookToken.objects.filter(project=project).count() == 1
    assert NotebookToken.objects.get(project=project).access_token == second


def test_relay_purges_expired_rows(auth_client, user_alice):
    stale = _project(user_alice)
    NotebookToken.objects.create(
        project=stale,
        user=user_alice,
        hub_user="internal",
        access_token="old",
        expires_at=timezone.now() - timedelta(minutes=1),
    )
    fresh = _project(user_alice)

    _relay(auth_client(user_alice), fresh, _jwt())

    assert not NotebookToken.objects.filter(project=stale).exists()
    assert NotebookToken.objects.filter(project=fresh).exists()


# ---------------------------------------------------------------------------
# Kernel path on the MCP proxies (service token + project_id)
# ---------------------------------------------------------------------------


def test_kernel_tools_forwards_relayed_token(
    auth_client, user_alice, service_token, fake_mcp
):
    project = _project(user_alice)
    token = _jwt()
    _relay(auth_client(user_alice), project, token)

    resp = APIClient().get(TOOLS_URL, {"project_id": str(project.id)}, **KERNEL)

    assert resp.status_code == 200
    assert resp.json()["tools"][0]["function"]["name"] == "get_flow"
    assert fake_mcp.tokens == [token]


def test_kernel_tools_rejects_wrong_service_token(user_alice, service_token, fake_mcp):
    project = _project(user_alice)
    resp = APIClient().get(
        TOOLS_URL,
        {"project_id": str(project.id)},
        HTTP_X_API_KEY="nope",
        HTTP_X_JUPYTERHUB_TOKEN="hub-internal",
    )
    assert resp.status_code == 401
    assert fake_mcp.tokens == []


def test_kernel_tools_requires_valid_hub_token(
    auth_client, user_alice, service_token, fake_mcp
):
    project = _project(user_alice)
    _relay(auth_client(user_alice), project, _jwt())
    params = {"project_id": str(project.id)}

    missing = APIClient().get(TOOLS_URL, params, HTTP_X_API_KEY="svc-token")
    assert missing.status_code == 401
    rejected = APIClient().get(
        TOOLS_URL, params, HTTP_X_API_KEY="svc-token", HTTP_X_JUPYTERHUB_TOKEN="nope"
    )
    assert rejected.status_code == 401
    assert fake_mcp.tokens == []


def test_kernel_in_other_space_cannot_use_token(
    auth_client, user_alice, service_token, fake_mcp
):
    """A community-space kernel must not use a token relayed for the project space."""
    project = _project(user_alice)
    _relay(auth_client(user_alice), project, _jwt())

    resp = APIClient().get(
        TOOLS_URL,
        {"project_id": str(project.id)},
        HTTP_X_API_KEY="svc-token",
        HTTP_X_JUPYTERHUB_TOKEN="hub-hackathon",
    )
    assert resp.status_code == 401
    assert fake_mcp.tokens == []


def test_kernel_tools_requires_project_id(service_token, fake_mcp):
    resp = APIClient().get(TOOLS_URL, **KERNEL)
    assert resp.status_code == 400


def test_kernel_tools_401_without_relayed_or_expired_token(
    user_alice, service_token, fake_mcp
):
    project = _project(user_alice)
    client = APIClient()
    params = {"project_id": str(project.id)}

    assert client.get(TOOLS_URL, params, **KERNEL).status_code == 401

    NotebookToken.objects.create(
        project=project,
        user=user_alice,
        hub_user="internal",
        access_token="old",
        expires_at=timezone.now() - timedelta(seconds=1),
    )
    assert client.get(TOOLS_URL, params, **KERNEL).status_code == 401
    assert fake_mcp.tokens == []


def _kernel_call(project, tool_name, arguments):
    return APIClient().post(
        CALL_URL,
        {
            "tool_name": tool_name,
            "arguments": arguments,
            "project_id": str(project.id),
        },
        format="json",
        **KERNEL,
    )


def test_kernel_call_scoped_to_relayed_project(
    auth_client, user_alice, service_token, fake_mcp
):
    project = _project(user_alice)
    other = _project(user_alice)
    token = _jwt()
    _relay(auth_client(user_alice), project, token)

    ok = _kernel_call(project, "get_flow", {"workflow_id": str(project.id)})
    assert ok.status_code == 200
    assert "called get_flow" in ok.json()["result"]
    assert fake_mcp.tokens == [token]

    denied = _kernel_call(project, "get_flow", {"workflow_id": str(other.id)})
    assert denied.status_code == 403
    assert fake_mcp.tokens == [token]  # never reached MCP

    unscoped = _kernel_call(project, "list_projects", {})
    assert unscoped.status_code == 200


def test_user_path_unchanged(auth_client, user_alice, service_token, fake_mcp):
    """A browser/user caller still forwards its own bearer token."""
    resp = auth_client(user_alice).get(TOOLS_URL, HTTP_AUTHORIZATION="Bearer user-jwt")
    assert resp.status_code == 200
    assert fake_mcp.tokens == ["user-jwt"]
