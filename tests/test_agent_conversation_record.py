"""Kernel side of saved notebook agent conversations (Issue #122).

Every turn is posted to the backend as OpenAI-shaped messages, with tool
arguments/results clipped and the kernel's tokens redacted; a recording
failure never breaks the chat. Requires ``claude_agent_sdk`` (run inside the
nest kernel image).
"""

import asyncio
import threading

import pytest

from neuroworkflow.agent.client import BackendClient, BackendError
from neuroworkflow.agent.config import AgentConfig
from neuroworkflow.agent.loop import Agent, RunHandle, _Run
from neuroworkflow.agent.permissions import Permissions
from neuroworkflow.agent.record import (
    TurnRecorder,
    clip,
    redact,
    tool_result_text,
)

UUID = "4b5023b0-8f1e-4dfc-87f0-1579c1a9bf00"


def _config(**overrides) -> AgentConfig:
    base = dict(
        backend_url="http://backend:3000",
        service_token="svc-token-123",
        hub_token="hub-internal-456",
        user_token=None,
        skills_dir="/skills",
        project_id=UUID,
        anthropic_base_url="http://backend:3000/api/chat/anthropic",
        anthropic_model=None,
        workspace_root="/home/jovyan/codes",
    )
    base.update(overrides)
    return AgentConfig(**base)


# ---------------------------------------------------------------------------
# record.py helpers
# ---------------------------------------------------------------------------


def test_clip():
    assert clip("x" * 4000) == "x" * 4000
    assert clip("x" * 4005) == "x" * 4000 + "…(5 chars truncated)"


def test_redact_nested_and_ignores_short_secrets():
    value = {"a": ["key=svc-token-123", {"b": "hub-internal-456!"}], "n": 3}
    assert redact(value, ["svc-token-123", "hub-internal-456", "", None, "short"]) == {
        "a": ["key=[REDACTED]", {"b": "[REDACTED]!"}],
        "n": 3,
    }
    assert redact("short text", ["short"]) == "short text"


def test_tool_result_text():
    assert tool_result_text(None) == ""
    assert tool_result_text("plain") == "plain"
    blocks = [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]
    assert tool_result_text(blocks) == "a\n[image]\nb"


def test_turn_recorder_builds_openai_messages():
    rec = TurnRecorder("hi")
    rec.add_text_delta("Let me")
    rec.add_text("Let me check.")
    rec.add_tool_use("t1", "run_code", {"code": "x" * 5000})
    rec.add_tool_result("t1", [{"type": "text", "text": "boom"}], True)
    rec.add_text("Fixed.")

    messages = rec.finish()

    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assistant = messages[1]
    assert assistant["content"] == "Let me check."
    call = assistant["tool_calls"][0]
    assert call["id"] == "t1" and call["function"]["name"] == "run_code"
    assert call["function"]["arguments"].endswith("chars truncated)")
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "t1",
        "tool_name": "run_code",
        "content": "boom",
        "is_error": True,
    }
    assert messages[3]["content"] == "Fixed."


def test_turn_recorder_redacts_before_clipping():
    # A token straddling the clip boundary must not leave a fragment behind.
    secret = "svc-token-0123456789abcdef0123456"  # 32 characters
    padding = "x" * 3969
    rec = TurnRecorder("hi", [secret])
    rec.add_tool_use("t1", "run_code", {"code": padding + secret})
    rec.add_tool_result("t1", padding + secret, False)

    arguments = rec.messages[1]["tool_calls"][0]["function"]["arguments"]
    result = rec.messages[2]["content"]
    for text in (arguments, result):
        assert secret[:8] not in text
    assert result == padding + "[REDACTED]"
    assert "[REDACTED]" in arguments


def test_turn_recorder_marks_interrupted_and_failed_turns():
    rec = TurnRecorder("hi")
    rec.add_text_delta("Half a sen")

    messages = rec.finish(RuntimeError("CLI died"))

    assert messages[-1]["content"] == (
        "Half a sen …(interrupted)\n[error] RuntimeError: CLI died"
    )


# ---------------------------------------------------------------------------
# BackendClient.record_turn
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeHttpx:
    calls: list = []
    response = _FakeResponse(201, {"id": "conv-1"})

    class Client:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, headers=None):
            _FakeHttpx.calls.append((url, headers, json))
            return _FakeHttpx.response


@pytest.fixture
def fake_httpx(monkeypatch):
    _FakeHttpx.calls = []
    _FakeHttpx.response = _FakeResponse(201, {"id": "conv-1"})
    monkeypatch.setattr(BackendClient, "_httpx", lambda self: _FakeHttpx)
    return _FakeHttpx


def test_record_turn_kernel_and_user_paths(fake_httpx):
    payload = {"project_id": UUID, "messages": []}

    assert BackendClient(_config()).record_turn(payload) == "conv-1"
    url, headers, body = fake_httpx.calls[-1]
    assert url == "http://backend:3000/api/chat/notebook-conversations/"
    assert headers == {
        "x-api-key": "svc-token-123",
        "x-jupyterhub-token": "hub-internal-456",
    }
    assert body == payload

    BackendClient(_config(user_token="eyJ-user")).record_turn(payload)
    assert fake_httpx.calls[-1][1] == {"Authorization": "Bearer eyJ-user"}


def test_record_turn_raises_on_error(fake_httpx):
    fake_httpx.response = _FakeResponse(404, {"detail": "Project not found."})
    with pytest.raises(BackendError):
        BackendClient(_config()).record_turn({"project_id": UUID})


# ---------------------------------------------------------------------------
# Agent: recording from the SDK message stream
# ---------------------------------------------------------------------------


class _RecordingClient:
    def __init__(self, error=None):
        self.payloads = []
        self.error = error

    def record_turn(self, payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return "conv-1"


def _agent(config=None, client=None) -> Agent:
    # Skip __init__: building the MCP servers would call the backend.
    agent = Agent.__new__(Agent)
    agent._config = config or _config()
    agent._client = client or _RecordingClient()
    agent._ipython = None
    agent._session_id = None
    agent._conversation_id = None
    agent._record_warned = False
    agent._servers = {}
    agent._append_prompt = ""
    agent.permissions = Permissions()
    agent.on_decision = None
    agent._busy = threading.Lock()
    agent._run = None
    return agent


def _inline_run():
    return _Run(lambda job: job(), RunHandle(), None, None, None, None, None)


def test_record_skips_without_project():
    agent = _agent(_config(project_id=None))
    agent._record([{"role": "user", "content": "hi"}])
    assert agent._client.payloads == []


def test_record_redacts_and_keeps_conversation_id():
    agent = _agent()
    agent._session_id = "sess-1"

    agent._record([{"role": "user", "content": "token is svc-token-123"}], "m-1")
    agent._record([{"role": "user", "content": "again"}])

    first, second = agent._client.payloads
    assert first["conversation_id"] is None
    assert first["model"] == "m-1"
    assert first["session_id"] == "sess-1"
    assert first["messages"][0]["content"] == "token is [REDACTED]"
    assert second["conversation_id"] == "conv-1"


def test_record_failure_warns_once_and_never_raises(capsys):
    agent = _agent(client=_RecordingClient(error=BackendError("down")))

    agent._record([{"role": "user", "content": "a"}])
    agent._record([{"role": "user", "content": "b"}])

    err = capsys.readouterr().err
    assert err.count("conversation not saved: down") == 1
    assert agent._conversation_id is None


def _fake_query(messages, raise_after=None):
    async def query(prompt=None, options=None):
        async for _ in prompt:
            pass
        for message in messages:
            yield message
        if raise_after:
            raise raise_after

    return query


def _sdk_turn():
    from claude_agent_sdk import (
        AssistantMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    return [
        AssistantMessage(
            content=[
                TextBlock(text="Running."),
                ToolUseBlock(
                    id="t1", name="mcp__notebook__run_code", input={"code": "1/0"}
                ),
            ],
            model="claude-test",
        ),
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="t1", content="ZeroDivisionError", is_error=True
                )
            ]
        ),
        # A subagent's internals are not recorded.
        AssistantMessage(
            content=[TextBlock(text="inside a subagent")],
            model="claude-test",
            parent_tool_use_id="t-task",
        ),
        AssistantMessage(content=[TextBlock(text="It fails.")], model="claude-test"),
    ]


def test_arun_records_the_turn(monkeypatch):
    monkeypatch.setattr("claude_agent_sdk.query", _fake_query(_sdk_turn()))
    agent = _agent()

    asyncio.run(agent._arun("divide", _inline_run()))

    (payload,) = agent._client.payloads
    assert payload["model"] == "claude-test"
    messages = payload["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[0]["content"] == "divide"
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "run_code",
        "arguments": '{"code": "1/0"}',
    }
    assert messages[2]["is_error"] is True
    assert messages[2]["tool_name"] == "run_code"
    assert messages[3]["content"] == "It fails."
    assert agent._conversation_id == "conv-1"


def test_arun_records_a_failed_turn(monkeypatch):
    turn = _sdk_turn()[:1]
    monkeypatch.setattr(
        "claude_agent_sdk.query", _fake_query(turn, raise_after=RuntimeError("cli"))
    )
    agent = _agent()

    with pytest.raises(RuntimeError):
        asyncio.run(agent._arun("divide", _inline_run()))

    (payload,) = agent._client.payloads
    assert payload["messages"][-1]["content"].endswith("[error] RuntimeError: cli")


def test_arun_redacts_tool_results_before_clipping(monkeypatch):
    from claude_agent_sdk import ToolResultBlock, UserMessage

    turn = _sdk_turn()[:1] + [
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="t1", content="x" * 3995 + "svc-token-123", is_error=False
                )
            ]
        )
    ]
    monkeypatch.setattr("claude_agent_sdk.query", _fake_query(turn))
    agent = _agent()

    asyncio.run(agent._arun("divide", _inline_run()))

    (payload,) = agent._client.payloads
    result = payload["messages"][2]["content"]
    assert "svc-t" not in result
    assert result.startswith("x" * 3995 + "[REDA")
