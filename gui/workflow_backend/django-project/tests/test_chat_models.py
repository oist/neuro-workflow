"""Tests for model selection in the browser chat (OpenAI-compatible providers,
such as MiniMax, next to OpenAI).

Covered:
- The catalogue offers a provider only when its API key is set, reads extra
  providers from LLM_PROVIDERS_FILE, and offers each model id once.
- /api/chat/stream/ accepts only models on offer and defaults to the first.
- The Chat Completions stream is parsed into the orchestrator's chunks with the
  provider's URL, key and extra body, and a truncated or empty stream is an
  error.
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
from app.chat.services.llm_providers import chat_models, provider_for_model

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
        {"id": "gpt-test", "provider": "openai", "notebook": False},
        {"id": "MiniMax-M3", "provider": "minimax", "notebook": True},
        {"id": "MiniMax-M2.7", "provider": "minimax", "notebook": True},
    ]

    monkeypatch.delenv("MINIMAX_MODELS")
    assert [m["id"] for m in chat_models()] == ["gpt-test", "MiniMax-M3"]

    monkeypatch.delenv("MINIMAX_API_KEY")
    assert [m["id"] for m in chat_models()] == ["gpt-test"]

    monkeypatch.delenv("OPENAI_API_KEY")
    assert chat_models() == []


def _providers_file(monkeypatch, tmp_path, content):
    path = tmp_path / "llm_providers.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(path))


_DEEPSEEK = {
    "name": "RIKEN-deepseek",
    "base_url": "https://ds.example/v1/",
    "api_key_env": "DS_KEY",
    "models": ["ds-chat", " ds-reasoner "],
    "anthropic_base_url": "https://ds.example/anthropic",
    "extra_body": {"thinking": {"type": "enabled"}},
}


def test_catalogue_reads_the_providers_file(providers, monkeypatch, tmp_path):
    openai_only = {
        "name": "groq",
        "base_url": "https://groq.example/v1",
        "api_key_env": "GROQ_KEY",
        "models": ["llama", "ds-chat", "MiniMax-M3"],
    }
    _providers_file(monkeypatch, tmp_path, {"providers": [_DEEPSEEK, openai_only]})

    # Without their keys the file's providers are not offered.
    assert [m["provider"] for m in chat_models()] == ["openai", "minimax", "minimax"]

    monkeypatch.setenv("DS_KEY", "ds-key")
    monkeypatch.setenv("GROQ_KEY", "groq-key")
    assert chat_models() == [
        {"id": "gpt-test", "provider": "openai", "notebook": False},
        {"id": "ds-chat", "provider": "RIKEN-deepseek", "notebook": True},
        {"id": "ds-reasoner", "provider": "RIKEN-deepseek", "notebook": True},
        # ds-chat is already deepseek's; MiniMax (from its variables) comes
        # after the file, so groq keeps MiniMax-M3.
        {"id": "llama", "provider": "groq", "notebook": False},
        {"id": "MiniMax-M3", "provider": "groq", "notebook": False},
        {"id": "MiniMax-M2.7", "provider": "minimax", "notebook": True},
    ]
    deepseek = provider_for_model("ds-reasoner")
    assert deepseek.base_url == "https://ds.example/v1"
    assert deepseek.api_key == "ds-key"
    assert provider_for_model("gpt-test") is None


def test_providers_file_can_define_minimax(providers, monkeypatch, tmp_path):
    _providers_file(
        monkeypatch,
        tmp_path,
        {
            "providers": [
                {
                    "name": "minimax",
                    "base_url": "https://mm.example/v1",
                    "api_key_env": "MM_FILE_KEY",
                    "models": ["MiniMax-M2"],
                }
            ]
        },
    )
    monkeypatch.setenv("MM_FILE_KEY", "mm-file-key")

    # The file's entry replaces the one built from MINIMAX_*.
    assert [m["id"] for m in chat_models()] == ["gpt-test", "MiniMax-M2"]
    assert provider_for_model("MiniMax-M2").extra_body == {}


_VALID = {
    "name": "p",
    "base_url": "https://p.example/v1",
    "api_key_env": "K",
    "models": ["p-chat"],
}


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        {"other": []},
        {"providers": {"name": "p"}},
        {"providers": [{**_VALID, "name": ""}]},
        {"providers": [{k: v for k, v in _VALID.items() if k != "base_url"}]},
        {"providers": [{**_VALID, "base_url": ""}]},
        {"providers": [{**_VALID, "api_key_env": None}]},
        # A string would otherwise become one model per character.
        {"providers": [{**_VALID, "models": "p-chat"}]},
        {"providers": [{**_VALID, "models": ["p-chat", 1]}]},
        {"providers": [{**_VALID, "anthropic_base_url": 1}]},
        {"providers": [{**_VALID, "extra_body": ["x"]}]},
    ],
)
def test_invalid_providers_file_is_ignored(providers, monkeypatch, tmp_path, content):
    _providers_file(monkeypatch, tmp_path, content)
    monkeypatch.setenv("K", "key")

    assert [m["id"] for m in chat_models()] == [
        "gpt-test",
        "MiniMax-M3",
        "MiniMax-M2.7",
    ]


def test_models_endpoint(providers, auth_client, user_alice):
    assert APIClient().get(reverse("chat-models")).status_code == 401

    resp = auth_client(user_alice).get(reverse("chat-models"))
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["models"]] == [
        "gpt-test",
        "MiniMax-M3",
        "MiniMax-M2.7",
    ]


def test_models_endpoint_accepts_the_kernel_service_token(providers, monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "svc-token")

    assert (
        APIClient().get(reverse("chat-models"), HTTP_X_API_KEY="no").status_code == 401
    )
    resp = APIClient().get(reverse("chat-models"), HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 200
    assert len(resp.json()["models"]) == 3


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
# OpenAI-compatible (Chat Completions) stream
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


def _fake_provider(monkeypatch, lines, status_code=200):
    """Serve ``lines`` as the provider's response; return the recorded request."""
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
    request = _fake_provider(
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

    _fake_provider(
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
        (['{"error": "unauthorized"}'], 401, "minimax API error 401"),
    ],
)
def test_minimax_stream_errors(providers, monkeypatch, lines, status_code, expected):
    _fake_provider(monkeypatch, lines, status_code)

    last = _chunks([{"role": "user", "content": "hi"}])[-1]

    assert last["type"] == "error"
    assert expected in last["message"]


def test_configured_provider_stream(providers, monkeypatch, tmp_path):
    _providers_file(monkeypatch, tmp_path, {"providers": [_DEEPSEEK]})
    monkeypatch.setenv("DS_KEY", "ds-key")
    request = _fake_provider(
        monkeypatch, _sse(_delta({"content": "hi"}, finish_reason="stop"), done=False)
    )

    chunks = _chunks([{"role": "user", "content": "hi"}], model="ds-chat")

    assert chunks == [{"type": "content_delta", "content": "hi"}, {"type": "done"}]
    assert request["url"] == "https://ds.example/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer ds-key"
    body = request["json"]
    assert body["model"] == "ds-chat"
    assert body["thinking"] == {"type": "enabled"}
    assert "reasoning_split" not in body
    assert "tools" not in body


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
