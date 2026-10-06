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
        hub_token="hub-internal",
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
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "hub-internal")
    monkeypatch.setattr("os.getcwd", lambda: f"{ROOT}/projects/{UUID}")

    config = get_config()

    assert config.project_id == UUID
    assert config.hub_token == "hub-internal"
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
            if url.endswith("/api/chat/models/"):
                return _FakeResponse(
                    {
                        "models": [
                            {"id": "gpt-test", "provider": "openai"},
                            {"id": "MiniMax-M3", "provider": "minimax"},
                        ]
                    }
                )
            return _FakeResponse({"tools": [{"function": {"name": "get_flow"}}]})

        def post(self, url, json=None, headers=None):
            _FakeHttpx.calls.append(("POST", url, headers, json))
            return _FakeResponse({"result": "ok"})


@pytest.fixture
def fake_httpx(monkeypatch):
    _FakeHttpx.calls = []
    monkeypatch.setattr(BackendClient, "_httpx", lambda self: _FakeHttpx)
    return _FakeHttpx


def test_kernel_path_sends_service_and_hub_tokens_and_project_id(fake_httpx):
    client = BackendClient(_config())
    kernel_headers = {"x-api-key": "svc-token", "x-jupyterhub-token": "hub-internal"}

    assert client.list_mcp_tools()[0]["function"]["name"] == "get_flow"
    assert client.call_mcp_tool("get_flow", {"workflow_id": UUID}) == "ok"

    method, url, headers, params = fake_httpx.calls[0]
    assert (method, url) == ("GET", "http://backend:3000/api/chat/mcp-tools/")
    assert headers == kernel_headers
    assert params == {"project_id": UUID}

    method, url, headers, payload = fake_httpx.calls[1]
    assert (method, url) == ("POST", "http://backend:3000/api/chat/mcp-call/")
    assert headers == kernel_headers
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


def test_default_model_uses_the_anthropic_proxy():
    config = _config(anthropic_model="claude-test")

    assert config.model == "claude-test"
    assert config.cli_env() == {
        "ANTHROPIC_BASE_URL": "http://backend:3000/api/chat/anthropic",
        "ANTHROPIC_API_KEY": "svc-token",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ANTHROPIC_MODEL": "claude-test",
    }


def test_minimax_model_uses_the_minimax_proxy_for_every_model_alias():
    config = _config(anthropic_model="claude-test", minimax_model="MiniMax-M3")
    env = config.cli_env()

    assert config.model == "MiniMax-M3"
    assert env["ANTHROPIC_BASE_URL"] == "http://backend:3000/api/chat/minimax"
    # The kernel still only holds the service token, never the MiniMax key.
    assert env["ANTHROPIC_API_KEY"] == "svc-token"
    for name in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        assert env[name] == "MiniMax-M3"


def test_get_config_model_selects_minimax():
    assert get_config(model="MiniMax-M3").minimax_model == "MiniMax-M3"
    assert get_config(model="").minimax_model is None
    assert get_config().minimax_model is None


def test_list_minimax_models_uses_the_service_token(fake_httpx):
    assert BackendClient(_config()).list_minimax_models() == ["MiniMax-M3"]

    method, url, headers, _ = fake_httpx.calls[0]
    assert (method, url) == ("GET", "http://backend:3000/api/chat/models/")
    assert headers == {"x-api-key": "svc-token"}


def test_get_agent_rebuilds_when_the_model_changes(monkeypatch):
    import neuroworkflow.agent as agent_pkg

    monkeypatch.setattr(agent_pkg, "_agent", None)
    monkeypatch.setattr(agent_pkg.Agent, "_build_tools", lambda self: None)

    claude = agent_pkg.get_agent()
    assert agent_pkg.get_agent() is claude
    assert agent_pkg.get_agent(model="") is claude

    minimax = agent_pkg.get_agent(model="MiniMax-M3")
    assert minimax is not claude
    assert minimax._config.minimax_model == "MiniMax-M3"
    # No model given: keep the current one.
    assert agent_pkg.get_agent() is minimax
    assert agent_pkg.get_agent(model="") is not minimax


def test_chat_magic_splits_the_model_option():
    from neuroworkflow.agent.magic import _split_model

    assert _split_model("--model MiniMax-M3 hello there") == (
        "MiniMax-M3",
        "hello there",
    )
    assert _split_model("--model MiniMax-M3") == ("MiniMax-M3", "")
    assert _split_model("hello --model x") == (None, "hello --model x")
