"""Tests for the notebook agent's admin tool policy (Issue #123).

Staff set ceilings on the notebook agent's tool permissions; the kernel reads
them with the service token. A deny ceiling on a workflow (MCP) tool is also
enforced by the notebook MCP proxies.
"""

import pytest
from django.urls import reverse
from rest_framework.test import APIClient

from app.chat.models import NotebookAgentPolicy
from tests.test_notebook_token import (  # noqa: F401
    KERNEL,
    TOOLS_URL,
    _FakeMCP,
    _jwt,
    _kernel_call,
    _project,
    _relay,
    fake_mcp,
    service_token,
)

pytestmark = pytest.mark.django_db

POLICY_URL = reverse("chat-notebook-policy")


class _TwoToolMCP(_FakeMCP):
    async def list_tools(self):
        return [
            {"name": name, "description": "", "inputSchema": {"type": "object"}}
            for name in ("get_flow", "add_node")
        ]


@pytest.fixture
def two_tool_mcp(monkeypatch):
    _FakeMCP.tokens = []
    monkeypatch.setattr("app.chat.views.MCPClient", _TwoToolMCP)
    return _TwoToolMCP


def _set_policy(**fields):
    policy = NotebookAgentPolicy.load()
    for key, value in fields.items():
        setattr(policy, key, value)
    policy.save()


# ---------------------------------------------------------------------------
# GET / PUT /api/chat/notebook-policy/
# ---------------------------------------------------------------------------


def test_get_rejects_anonymous():
    assert APIClient().get(POLICY_URL).status_code == 401


def test_get_with_service_token(service_token):
    resp = APIClient().get(POLICY_URL, HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["category_ceilings"] == {}
    assert body["tool_ceilings"] == {}
    assert "code-exec" in body["categories"]
    assert "get_flow" in body["workflow_read_tools"]
    assert "add_node" not in body["workflow_read_tools"]


def test_get_as_user(auth_client, user_alice):
    resp = auth_client(user_alice).get(POLICY_URL)
    assert resp.status_code == 200


def test_put_requires_staff(auth_client, user_alice, service_token):
    body = {"category_ceilings": {"code-exec": "ask"}}
    resp = auth_client(user_alice).put(POLICY_URL, body, format="json")
    assert resp.status_code == 403
    resp = APIClient().put(POLICY_URL, body, format="json", HTTP_X_API_KEY="svc-token")
    assert resp.status_code in (401, 403)
    assert NotebookAgentPolicy.load().category_ceilings == {}


@pytest.mark.parametrize(
    "body",
    [
        {"category_ceilings": {"nope": "ask"}},
        {"category_ceilings": {"code-exec": "maybe"}},
        {"tool_ceilings": {"delete_node": "never"}},
        {"tool_ceilings": {"": "deny"}},
        {"tool_ceilings": ["delete_node"]},
    ],
)
def test_put_rejects_bad_values(auth_client, user_admin, body):
    resp = auth_client(user_admin).put(POLICY_URL, body, format="json")
    assert resp.status_code == 400


def test_put_saves_partial_update(auth_client, user_admin):
    client = auth_client(user_admin)
    resp = client.put(
        POLICY_URL, {"category_ceilings": {"code-exec": "ask"}}, format="json"
    )
    assert resp.status_code == 200
    resp = client.put(
        POLICY_URL, {"tool_ceilings": {"delete_node": "deny"}}, format="json"
    )
    assert resp.status_code == 200
    assert resp.json()["category_ceilings"] == {"code-exec": "ask"}

    assert NotebookAgentPolicy.objects.count() == 1
    policy = NotebookAgentPolicy.load()
    assert policy.tool_ceilings == {"delete_node": "deny"}
    assert policy.updated_by == user_admin


# ---------------------------------------------------------------------------
# Enforcement on the notebook MCP proxies
# ---------------------------------------------------------------------------


def test_kernel_tool_list_drops_denied_tools(
    auth_client, user_alice, service_token, two_tool_mcp
):
    project = _project(user_alice)
    _relay(auth_client(user_alice), project, _jwt())
    _set_policy(category_ceilings={"workflow-edit": "deny"})

    resp = APIClient().get(TOOLS_URL, {"project_id": str(project.id)}, **KERNEL)

    assert resp.status_code == 200
    assert [t["function"]["name"] for t in resp.json()["tools"]] == ["get_flow"]


def test_user_tool_list_is_unfiltered(auth_client, user_alice, two_tool_mcp):
    """The Chat Profile editor lists every tool through the user path."""
    _set_policy(category_ceilings={"workflow-edit": "deny"})

    resp = auth_client(user_alice).get(TOOLS_URL, HTTP_AUTHORIZATION="Bearer user-jwt")

    assert [t["function"]["name"] for t in resp.json()["tools"]] == [
        "get_flow",
        "add_node",
    ]


def test_call_refuses_denied_tool(auth_client, user_alice, service_token, fake_mcp):
    project = _project(user_alice)
    token = _jwt()
    _relay(auth_client(user_alice), project, token)
    _set_policy(category_ceilings={"workflow-edit": "deny"})

    denied = _kernel_call(project, "add_node", {"workflow_id": str(project.id)})
    assert denied.status_code == 403
    assert "disabled for the notebook agent" in denied.json()["error"]
    assert fake_mcp.tokens == []  # never reached MCP

    allowed = _kernel_call(project, "get_flow", {"workflow_id": str(project.id)})
    assert allowed.status_code == 200


def test_call_refuses_denied_tool_on_user_path(auth_client, user_alice, fake_mcp):
    _set_policy(tool_ceilings={"get_flow": "deny"})

    resp = auth_client(user_alice).post(
        reverse("chat-notebook-mcp-call"),
        {"tool_name": "get_flow", "arguments": {}},
        format="json",
        HTTP_AUTHORIZATION="Bearer user-jwt",
    )

    assert resp.status_code == 403
    assert fake_mcp.tokens == []


def test_tool_ceiling_overrides_category(
    auth_client, user_alice, service_token, fake_mcp
):
    project = _project(user_alice)
    _relay(auth_client(user_alice), project, _jwt())
    _set_policy(
        category_ceilings={"workflow-edit": "deny"},
        tool_ceilings={"add_node": "ask", "get_flow": "deny"},
    )

    assert _kernel_call(project, "add_node", {}).status_code == 200
    assert _kernel_call(project, "get_flow", {}).status_code == 403
    assert _kernel_call(project, "delete_node", {}).status_code == 403
