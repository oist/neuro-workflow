"""Kernel side of the notebook token relay (neuroworkflow.agent).

The kernel never holds the user's Keycloak token: it derives ``project_id``
from the notebook folder and calls the backend MCP proxies with the service
token, and the backend forwards the token the browser relayed for that project.
Requires ``claude_agent_sdk`` (run inside the nest kernel image).
"""

import pytest

from neuroworkflow.agent.client import BackendClient
from neuroworkflow.agent.config import AgentConfig, _project_id_from_cwd, get_config

UUID = "4b5023b0-8f1e-4dfc-87f0-1579c1a9bf00"
ROOT = "/home/jovyan/codes"


def _config(**overrides) -> AgentConfig:
    base = dict(
        backend_url="http://backend:3000",
        service_token="svc-token",
        user_token=None,
        skills_dir="/skills",
        project_id=UUID,
        anthropic_base_url="http://backend:3000/api/chat/anthropic",
        anthropic_model=None,
        workspace_root=ROOT,
    )
    base.update(overrides)
    return AgentConfig(**base)


@pytest.mark.parametrize(
    "cwd, expected",
    [
        (f"{ROOT}/projects/{UUID}", UUID),
        (f"{ROOT}/projects/{UUID}/sub", UUID),
        (f"{ROOT}/projects/legacy-name", None),
        (f"{ROOT}/nodes/analysis", None),
        ("/home/jovyan", None),
    ],
)
def test_project_id_from_cwd(monkeypatch, cwd, expected):
    monkeypatch.setattr("os.getcwd", lambda: cwd)
    assert _project_id_from_cwd(ROOT) == expected


def test_get_config_derives_project_id_from_cwd(monkeypatch):
    monkeypatch.delenv("NEUROWORKFLOW_PROJECT_ID", raising=False)
    monkeypatch.delenv("NEUROWORKFLOW_USER_TOKEN", raising=False)
    monkeypatch.setenv("NEUROWORKFLOW_SERVICE_TOKEN", "svc-token")
    monkeypatch.setattr("os.getcwd", lambda: f"{ROOT}/projects/{UUID}")

    config = get_config()

    assert config.project_id == UUID
    assert config.user_token is None
    assert config.has_mcp is True
    assert get_config(project_id="explicit").project_id == "explicit"


def test_has_mcp_rules():
    assert _config(user_token="eyJ", project_id=None).has_mcp is True
    assert _config(user_token=None, project_id=UUID).has_mcp is True
    assert _config(user_token=None, project_id=None).has_mcp is False
    assert _config(user_token=None, service_token="").has_mcp is False


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHttpx:
    """Records the last request the client made."""

    calls: list = []

    class Client:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None, params=None):
            _FakeHttpx.calls.append(("GET", url, headers, params))
            return _FakeResponse({"tools": [{"function": {"name": "get_flow"}}]})

        def post(self, url, json=None, headers=None):
            _FakeHttpx.calls.append(("POST", url, headers, json))
            return _FakeResponse({"result": "ok"})


@pytest.fixture
def fake_httpx(monkeypatch):
    _FakeHttpx.calls = []
    monkeypatch.setattr(BackendClient, "_httpx", lambda self: _FakeHttpx)
    return _FakeHttpx


def test_kernel_path_sends_service_token_and_project_id(fake_httpx):
    client = BackendClient(_config())

    assert client.list_mcp_tools()[0]["function"]["name"] == "get_flow"
    assert client.call_mcp_tool("get_flow", {"workflow_id": UUID}) == "ok"

    method, url, headers, params = fake_httpx.calls[0]
    assert (method, url) == ("GET", "http://backend:3000/api/chat/mcp-tools/")
    assert headers == {"x-api-key": "svc-token"}
    assert params == {"project_id": UUID}

    method, url, headers, payload = fake_httpx.calls[1]
    assert (method, url) == ("POST", "http://backend:3000/api/chat/mcp-call/")
    assert headers == {"x-api-key": "svc-token"}
    assert payload == {
        "tool_name": "get_flow",
        "arguments": {"workflow_id": UUID},
        "project_id": UUID,
    }


def test_explicit_user_token_is_forwarded_directly(fake_httpx):
    client = BackendClient(_config(user_token="eyJ.user", project_id=None))

    client.list_mcp_tools()
    client.call_mcp_tool("list_projects", {})

    for call in fake_httpx.calls:
        assert call[2] == {"Authorization": "Bearer eyJ.user"}
    assert fake_httpx.calls[0][3] == {}
    assert "project_id" not in fake_httpx.calls[1][3]


def test_no_mcp_config_skips_backend(fake_httpx):
    client = BackendClient(_config(user_token=None, project_id=None))
    assert client.list_mcp_tools() == []
    assert fake_httpx.calls == []
