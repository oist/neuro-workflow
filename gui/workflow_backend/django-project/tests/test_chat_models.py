"""Tests for model selection in the browser chat (MiniMax next to OpenAI).

Covered:
- The catalogue offers a provider only when its API key is set.
- /api/chat/stream/ accepts only models on offer and defaults to the first.
- The MiniMax (Chat Completions) stream is parsed into the orchestrator's
  chunks, and a truncated or empty stream is an error.
- The model's reasoning is stored with its tool calls and sent back.
"""

import json
from contextlib import asynccontextmanager

import pytest
from asgiref.sync import async_to_sync
from django.urls import reverse
from rest_framework.test import APIClient

from app.chat import views as chat_views
from app.chat.models import Conversation
from app.chat.services import chat_orchestrator, openai_client
from app.chat.services.chat_orchestrator import orchestrate_chat
from app.chat.services.llm_providers import chat_models

pytestmark = pytest.mark.django_db


@pytest.fixture
def providers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
    monkeypatch.setenv("MINIMAX_MODELS", "MiniMax-M3, MiniMax-M2.7")
    monkeypatch.delenv("MINIMAX_OPENAI_BASE_URL", raising=False)


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------


def test_catalogue_lists_providers_with_a_key(providers, monkeypatch):
    assert chat_models() == [
        {"id": "gpt-test", "provider": "openai"},
        {"id": "MiniMax-M3", "provider": "minimax"},
        {"id": "MiniMax-M2.7", "provider": "minimax"},
    ]

    monkeypatch.delenv("MINIMAX_MODELS")
    assert [m["id"] for m in chat_models()] == ["gpt-test", "MiniMax-M3"]

    monkeypatch.delenv("MINIMAX_API_KEY")
    assert [m["id"] for m in chat_models()] == ["gpt-test"]

    monkeypatch.delenv("OPENAI_API_KEY")
    assert chat_models() == []


def test_models_endpoint(providers, auth_client, user_alice):
    assert APIClient().get(reverse("chat-models")).status_code == 401

    resp = auth_client(user_alice).get(reverse("chat-models"))
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["models"]] == [
        "gpt-test",
        "MiniMax-M3",
        "MiniMax-M2.7",
    ]


# --------------------------------------------------------------------------
# Model resolution on /api/chat/stream/
# --------------------------------------------------------------------------


def _capture_stream_model(monkeypatch):
    seen = {}

    async def fake_orchestrate_chat(conversation, user_message, **kwargs):
        seen["model"] = kwargs.get("model")
        yield {"type": "done", "data": {}}

    monkeypatch.setattr(chat_views, "orchestrate_chat", fake_orchestrate_chat)
    return seen


def _stream(client, **payload):
    resp = client.post(
        reverse("chat-stream"), {"message": "hi", **payload}, format="json"
    )
    if resp.status_code == 200:
        b"".join(resp.streaming_content)  # drive the SSE generator
    return resp


def test_stream_rejects_model_not_on_offer(providers, auth_client, user_alice):
    resp = _stream(auth_client(user_alice), model="not-a-model")
    assert resp.status_code == 400
    assert Conversation.objects.count() == 0


def test_stream_uses_chosen_or_default_model(
    providers, auth_client, user_alice, monkeypatch
):
    seen = _capture_stream_model(monkeypatch)

    assert _stream(auth_client(user_alice), model="MiniMax-M2.7").status_code == 200
    assert seen["model"] == "MiniMax-M2.7"

    assert _stream(auth_client(user_alice)).status_code == 200
    assert seen["model"] == "gpt-test"


# --------------------------------------------------------------------------
# MiniMax (Chat Completions) stream
# --------------------------------------------------------------------------


def _sse(*events, done=True):
    lines = [f"data: {json.dumps(e)}" for e in events]
    return lines + (["data: [DONE]"] if done else [])


def _delta(delta, finish_reason=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


class _FakeResponse:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return "\n".join(self._lines).encode()


def _fake_minimax(monkeypatch, lines, status_code=200):
    """Serve ``lines`` as the MiniMax response; return the recorded request."""
    request = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        @asynccontextmanager
        async def stream(self, method, url, json=None, headers=None):
            request.update(url=url, json=json, headers=headers)
            yield _FakeResponse(lines, status_code)

    monkeypatch.setattr(openai_client.httpx, "AsyncClient", _FakeAsyncClient)
    return request


def _chunks(messages, tools=None, model="MiniMax-M3"):
    async def _collect():
        return [
            c
            async for c in openai_client.stream_chat_completion(
                messages, tools, model=model
            )
        ]

    return async_to_sync(_collect)()


_TOOL = {
    "type": "function",
    "function": {"name": "add_node", "parameters": {"type": "object"}},
}


def test_minimax_request_and_text_stream(providers, monkeypatch):
    request = _fake_minimax(
        monkeypatch,
        _sse(
            _delta({"reasoning_content": "thinking"}),
            _delta({"content": "Hel"}),
            _delta({"content": "lo"}, finish_reason="stop"),
            {"choices": [], "usage": {"total_tokens": 3}},
        ),
    )
    messages = [
        {"role": "system", "content": "PROMPT"},
        {"role": "system", "content": "VIEWER"},
        {"role": "user", "content": "hi"},
    ]

    chunks = _chunks(messages, [_TOOL])

    assert chunks == [
        {"type": "reasoning_delta", "content": "thinking"},
        {"type": "content_delta", "content": "Hel"},
        {"type": "content_delta", "content": "lo"},
        {"type": "done"},
    ]
    assert request["url"] == "https://api.minimax.io/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer mm-key"
    body = request["json"]
    assert body["model"] == "MiniMax-M3"
    assert body["reasoning_split"] is True
    assert body["tools"] == [_TOOL]
    # The two leading system messages are sent as one.
    assert body["messages"] == [
        {"role": "system", "content": "PROMPT\n\nVIEWER"},
        {"role": "user", "content": "hi"},
    ]


def test_minimax_tool_call_stream_names_each_call_once(providers, monkeypatch):
    def call(arguments, **extra):
        function = {"name": "add_node", "arguments": arguments}
        return _delta({"tool_calls": [{"index": 0, "function": function, **extra}]})

    _fake_minimax(
        monkeypatch,
        _sse(
            call("", id="call_1"),
            call('{"a":'),
            call("1}"),
            _delta({}, finish_reason="tool_calls"),
        ),
    )

    chunks = _chunks([{"role": "user", "content": "hi"}], [_TOOL])

    assert chunks[-1] == {"type": "tool_calls_complete"}
    deltas = [c for c in chunks if c["type"] == "tool_call_delta"]
    assert [c["function_name"] for c in deltas] == ["add_node", None, None]
    assert deltas[0]["id"] == "call_1"
    assert "".join(c["arguments_delta"] for c in deltas) == '{"a":1}'


@pytest.mark.parametrize(
    "lines, status_code, expected",
    [
        # Output cut off by the token limit: never run or save it.
        (
            _sse(_delta({"content": "x"}, finish_reason="length")),
            200,
            "max output tokens",
        ),
        # A 200 that is not an event stream (e.g. a JSON error body).
        (['{"base_resp": {"status_msg": "bad"}}'], 200, "without a result"),
        (['{"error": "unauthorized"}'], 401, "MiniMax API error 401"),
    ],
)
def test_minimax_stream_errors(providers, monkeypatch, lines, status_code, expected):
    _fake_minimax(monkeypatch, lines, status_code)

    last = _chunks([{"role": "user", "content": "hi"}])[-1]

    assert last["type"] == "error"
    assert expected in last["message"]


def test_other_models_use_the_openai_path(providers, monkeypatch):
    monkeypatch.setattr(openai_client, "OPENAI_API_KEY", "")

    chunks = _chunks([{"role": "user", "content": "hi"}], model="gpt-test")

    assert chunks == [{"type": "error", "message": "OpenAI API key is not configured"}]


# --------------------------------------------------------------------------
# Reasoning is kept with the tool calls
# --------------------------------------------------------------------------


class _FakeMCP:
    def __init__(self, auth_token=None):
        pass

    async def initialize(self):
        return {}

    async def list_tools(self):
        return [{"name": "add_node", "description": "", "inputSchema": {}}]

    async def call_tool(self, name, arguments):
        return "added"


def test_reasoning_is_stored_and_sent_back_with_tool_calls(user_alice, monkeypatch):
    calls = []

    async def stream_chat_completion(messages, tools=None, model=None):
        calls.append({"messages": messages, "model": model})
        if len(calls) == 1:
            yield {"type": "reasoning_delta", "content": "I should add a node"}
            yield {
                "type": "tool_call_delta",
                "index": 0,
                "id": "call_1",
                "function_name": "add_node",
                "arguments_delta": "{}",
            }
            yield {"type": "tool_calls_complete"}
        else:
            yield {"type": "content_delta", "content": "done"}
            yield {"type": "done"}

    monkeypatch.setattr(chat_orchestrator, "MCPClient", _FakeMCP)
    monkeypatch.setattr(
        chat_orchestrator, "stream_chat_completion", stream_chat_completion
    )
    conv = Conversation.objects.create(user=user_alice)

    async def _collect():
        return [e async for e in orchestrate_chat(conv, "hi", model="MiniMax-M3")]

    events = async_to_sync(_collect)()

    assert [c["model"] for c in calls] == ["MiniMax-M3", "MiniMax-M3"]
    # The reasoning is not shown to the user...
    assert "I should add a node" not in json.dumps(events)
    # ...but goes back to the model with the tool calls it led to.
    assistant = [m for m in calls[1]["messages"] if m["role"] == "assistant"]
    assert assistant[0]["reasoning_content"] == "I should add a node"
    assert assistant[0]["tool_calls"][0]["id"] == "call_1"
