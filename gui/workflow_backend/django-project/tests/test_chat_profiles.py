"""Tests for browser chat profiles (admin-managed MCP tool allowlist + prompt).

Covered:
- Profiles are shared: everyone can read them, only staff can change them.
- Serializer validation (unique name, allowed_tools shape, single default).
- Non-staff users get the default profile when they send no profile_id.
- The allowlist filters the tools offered to OpenAI, and an empty allowlist
  skips MCP entirely.
- Off-allowlist tools/call is refused without calling mcp.call_tool.
- System prompt precedence: profile > conversation > default.
- Serializer caps: system_prompt length, empty tool names, unique-name 400.
"""

import uuid

import pytest
from asgiref.sync import async_to_sync
from django.urls import reverse
from rest_framework.test import APIClient

from app.chat import views as chat_views
from app.chat.models import ChatProfile, Conversation
from app.chat.services import chat_orchestrator
from app.chat.services.chat_orchestrator import (
    DEFAULT_SYSTEM_PROMPT,
    TOOLS_DISABLED_NOTE,
    _build_openai_messages,
    orchestrate_chat,
)
from app.chat.services.mcp_client import mcp_tools_to_openai_functions

pytestmark = pytest.mark.django_db


def _create(client, **overrides):
    payload = {
        "name": "Viewer only",
        "allowed_tools": ["viewer_get_region"],
        "system_prompt": "",
    }
    payload.update(overrides)
    return client.post(reverse("chat-profiles"), payload, format="json")


# --------------------------------------------------------------------------
# CRUD API
# --------------------------------------------------------------------------


def test_profiles_require_auth():
    assert APIClient().get(reverse("chat-profiles")).status_code == 401


def test_profiles_are_shared_and_admin_managed(auth_client, user_admin, user_alice):
    admin = auth_client(user_admin)
    alice = auth_client(user_alice)

    resp = _create(admin)
    assert resp.status_code == 201, resp.json()
    profile_id = resp.json()["id"]
    assert resp.json()["is_default"] is False
    assert ChatProfile.objects.get(id=profile_id).created_by == user_admin

    # Everyone signed in can read the shared profiles.
    detail = reverse("chat-profile-detail", args=[profile_id])
    names = [p["name"] for p in alice.get(reverse("chat-profiles")).json()]
    assert names == ["Viewer only"]
    assert alice.get(detail).status_code == 200

    # Only staff may change them.
    assert _create(alice, name="mine").status_code == 403
    assert alice.put(detail, {"name": "x"}, format="json").status_code == 403
    assert alice.delete(detail).status_code == 403
    assert ChatProfile.objects.get(id=profile_id).name == "Viewer only"

    resp = admin.put(
        detail, {"allowed_tools": ["add_node", "delete_node"]}, format="json"
    )
    assert resp.status_code == 200
    assert resp.json()["allowed_tools"] == ["add_node", "delete_node"]
    assert resp.json()["name"] == "Viewer only"  # partial update keeps name

    assert admin.delete(detail).status_code == 204
    assert not ChatProfile.objects.filter(id=profile_id).exists()


def test_duplicate_name_rejected(auth_client, user_admin):
    admin = auth_client(user_admin)
    assert _create(admin).status_code == 201
    resp = _create(admin)
    assert resp.status_code == 400
    assert "name" in resp.json()
    # Surrounding whitespace does not make a new name.
    assert _create(admin, name="  Viewer only ").status_code == 400


def test_allowed_tools_validation(auth_client, user_admin):
    admin = auth_client(user_admin)
    assert _create(admin, name="a", allowed_tools="add_node").status_code == 400
    resp = _create(admin, name="b", allowed_tools=[{"name": "add_node"}])
    assert resp.status_code == 400

    resp = _create(admin, name="c", allowed_tools=["add_node", "add_node"])
    assert resp.status_code == 201
    assert resp.json()["allowed_tools"] == ["add_node"]

    resp = _create(admin, name="d", allowed_tools=[])
    assert resp.status_code == 201
    assert resp.json()["allowed_tools"] == []

    resp = _create(admin, name="e", allowed_tools=["", "add_node"])
    assert resp.status_code == 400
    resp = _create(admin, name="f", allowed_tools=["   "])
    assert resp.status_code == 400
    resp = _create(admin, name="g", allowed_tools=["  add_node  ", "delete_node"])
    assert resp.status_code == 201
    assert resp.json()["allowed_tools"] == ["add_node", "delete_node"]


def test_system_prompt_capped(auth_client, user_admin):
    admin = auth_client(user_admin)
    too_long = _create(admin, name="long", system_prompt="x" * 16001)
    assert too_long.status_code == 400
    assert "system_prompt" in too_long.json()
    ok = _create(admin, name="ok", system_prompt="x" * 16000)
    assert ok.status_code == 201


def test_duplicate_name_integrity_error_returns_400(
    auth_client, user_admin, monkeypatch
):
    admin = auth_client(user_admin)
    assert _create(admin, name="Viewer only").status_code == 201

    from app.chat.serializers import ChatProfileSerializer

    monkeypatch.setattr(
        ChatProfileSerializer, "validate_name", lambda self, value: value.strip()
    )
    resp = _create(admin, name="Viewer only")
    assert resp.status_code == 400
    assert "name" in resp.json()
    # The failed insert must not poison the connection for later queries.
    assert ChatProfile.objects.count() == 1


def test_only_one_default_profile(auth_client, user_admin):
    admin = auth_client(user_admin)
    a = _create(admin, name="A", is_default=True).json()
    assert a["is_default"] is True

    b = _create(admin, name="B", is_default=True).json()
    assert b["is_default"] is True
    assert ChatProfile.objects.get(id=a["id"]).is_default is False

    detail_a = reverse("chat-profile-detail", args=[a["id"]])
    resp = admin.put(detail_a, {"is_default": True}, format="json")
    assert resp.status_code == 200 and resp.json()["is_default"] is True
    assert ChatProfile.objects.get(id=b["id"]).is_default is False

    resp = admin.put(detail_a, {"is_default": False}, format="json")
    assert resp.status_code == 200 and resp.json()["is_default"] is False
    assert not ChatProfile.objects.filter(is_default=True).exists()


# --------------------------------------------------------------------------
# Profile resolution on /api/chat/stream/
# --------------------------------------------------------------------------


def _capture_stream_profile(monkeypatch):
    """Replace the orchestrator with a stub that records the profile it got."""
    seen = {}

    async def fake_orchestrate_chat(conversation, user_message, **kwargs):
        seen["profile"] = kwargs.get("profile")
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


def test_stream_rejects_unknown_profile(auth_client, user_alice):
    resp = _stream(auth_client(user_alice), profile_id=str(uuid.uuid4()))
    assert resp.status_code == 404
    # A bad profile id must not leave an orphan conversation behind.
    assert Conversation.objects.count() == 0


def test_stream_applies_default_profile_to_non_staff(
    auth_client, user_alice, monkeypatch
):
    seen = _capture_stream_profile(monkeypatch)
    default = ChatProfile.objects.create(
        name="locked", allowed_tools=[], is_default=True
    )
    other = ChatProfile.objects.create(name="other", allowed_tools=["add_node"])

    assert _stream(auth_client(user_alice)).status_code == 200
    assert seen["profile"] == default

    # An explicit choice still wins.
    assert _stream(auth_client(user_alice), profile_id=str(other.id)).status_code == 200
    assert seen["profile"] == other


def test_stream_without_default_or_for_staff_uses_no_profile(
    auth_client, user_alice, user_admin, monkeypatch
):
    seen = _capture_stream_profile(monkeypatch)

    assert _stream(auth_client(user_alice)).status_code == 200
    assert seen["profile"] is None

    ChatProfile.objects.create(name="locked", allowed_tools=[], is_default=True)
    assert _stream(auth_client(user_admin)).status_code == 200
    assert seen["profile"] is None


# --------------------------------------------------------------------------
# Tool filtering
# --------------------------------------------------------------------------

_TOOLS = [
    {"name": "add_node", "description": "Add", "inputSchema": {"type": "object"}},
    {"name": "delete_node", "description": "Del", "inputSchema": {"type": "object"}},
]


def test_mcp_tools_to_openai_functions_allowlist():
    assert len(mcp_tools_to_openai_functions(_TOOLS)) == 2
    assert len(mcp_tools_to_openai_functions(_TOOLS, allowed=None)) == 2
    only = mcp_tools_to_openai_functions(_TOOLS, allowed={"add_node"})
    assert [f["function"]["name"] for f in only] == ["add_node"]
    assert mcp_tools_to_openai_functions(_TOOLS, allowed=set()) == []


class _ExplodingMCP:
    def __init__(self, *args, **kwargs):
        raise AssertionError("MCPClient must not be constructed when tools are off")


class _FakeMCP:
    def __init__(self, auth_token=None):
        self.auth_token = auth_token

    async def initialize(self):
        return {}

    async def list_tools(self):
        return _TOOLS

    async def call_tool(self, name, arguments):
        return f"called {name}"


def _fake_stream(recorder):
    async def stream_chat_completion(messages, tools=None):
        recorder["tools"] = tools
        recorder["messages"] = messages
        yield {"type": "content_delta", "content": "hi"}
        yield {"type": "done"}

    return stream_chat_completion


def _run(conversation, profile):
    async def _collect():
        return [
            e async for e in orchestrate_chat(conversation, "hello", profile=profile)
        ]

    return async_to_sync(_collect)()


def test_orchestrator_skips_mcp_when_tools_disabled(user_alice, monkeypatch):
    recorder = {}
    monkeypatch.setattr(chat_orchestrator, "MCPClient", _ExplodingMCP)
    monkeypatch.setattr(
        chat_orchestrator, "stream_chat_completion", _fake_stream(recorder)
    )
    conv = Conversation.objects.create(user=user_alice)
    profile = ChatProfile.objects.create(name="none", allowed_tools=[])

    events = _run(conv, profile)

    assert [e["type"] for e in events] == ["text_delta", "done"]
    assert recorder["tools"] is None
    system = recorder["messages"][0]
    assert system["role"] == "system"
    assert system["content"].endswith(TOOLS_DISABLED_NOTE)


def test_orchestrator_filters_tools_by_profile(user_alice, monkeypatch):
    recorder = {}
    monkeypatch.setattr(chat_orchestrator, "MCPClient", _FakeMCP)
    monkeypatch.setattr(
        chat_orchestrator, "stream_chat_completion", _fake_stream(recorder)
    )
    conv = Conversation.objects.create(user=user_alice)
    profile = ChatProfile.objects.create(name="add only", allowed_tools=["add_node"])

    _run(conv, profile)

    assert [t["function"]["name"] for t in recorder["tools"]] == ["add_node"]
    assert "Only these tools are enabled" in recorder["messages"][0]["content"]


def test_orchestrator_without_profile_is_unchanged(user_alice, monkeypatch):
    recorder = {}
    monkeypatch.setattr(chat_orchestrator, "MCPClient", _FakeMCP)
    monkeypatch.setattr(
        chat_orchestrator, "stream_chat_completion", _fake_stream(recorder)
    )
    conv = Conversation.objects.create(user=user_alice)

    _run(conv, None)

    names = [t["function"]["name"] for t in recorder["tools"]]
    assert names == ["add_node", "delete_node"]
    assert recorder["messages"][0]["content"] == DEFAULT_SYSTEM_PROMPT


class _RecordingMCP:
    def __init__(self, auth_token=None):
        self.auth_token = auth_token
        self.calls = []

    async def initialize(self):
        return {}

    async def list_tools(self):
        return _TOOLS

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        raise AssertionError(
            "mcp.call_tool must not be invoked for an off-allowlist tool"
        )


def _tool_call_then_done_stream(recorder, tool_name):
    call_count = {"n": 0}

    async def stream_chat_completion(messages, tools=None):
        recorder["tools"] = tools
        recorder["messages"] = messages
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield {
                "type": "tool_call_delta",
                "index": 0,
                "id": "call_denied",
                "function_name": tool_name,
            }
            yield {
                "type": "tool_call_delta",
                "index": 0,
                "arguments_delta": "{}",
            }
            yield {"type": "tool_calls_complete"}
        else:
            yield {"type": "content_delta", "content": "ok"}
            yield {"type": "done"}

    return stream_chat_completion


def test_orchestrator_refuses_off_allowlist_tool_call(user_alice, monkeypatch):
    recorder = {}
    mcp_holder = {}

    class _CaptureMCP(_RecordingMCP):
        def __init__(self, auth_token=None):
            super().__init__(auth_token)
            mcp_holder["client"] = self

    monkeypatch.setattr(chat_orchestrator, "MCPClient", _CaptureMCP)
    monkeypatch.setattr(
        chat_orchestrator,
        "stream_chat_completion",
        _tool_call_then_done_stream(recorder, "delete_node"),
    )
    conv = Conversation.objects.create(user=user_alice)
    profile = ChatProfile.objects.create(name="add only", allowed_tools=["add_node"])

    events = _run(conv, profile)

    assert mcp_holder["client"].calls == []
    refusals = [
        e
        for e in events
        if e["type"] == "tool_result"
        and "not enabled in the current chat profile" in e["data"]["result"]
    ]
    assert len(refusals) == 1
    assert refusals[0]["data"]["tool_name"] == "delete_node"


# --------------------------------------------------------------------------
# System prompt precedence
# --------------------------------------------------------------------------


def test_system_prompt_precedence(user_alice):
    conv = Conversation.objects.create(user=user_alice, system_prompt="CONV PROMPT")
    with_prompt = ChatProfile.objects.create(
        name="p", allowed_tools=["add_node"], system_prompt="PROFILE PROMPT"
    )
    without_prompt = ChatProfile.objects.create(name="q", allowed_tools=["add_node"])
    build = async_to_sync(_build_openai_messages)

    assert build(conv, None, with_prompt)[0]["content"].startswith("PROFILE PROMPT")
    assert build(conv, None, without_prompt)[0]["content"].startswith("CONV PROMPT")
    assert build(conv, None, None)[0]["content"] == "CONV PROMPT"

    plain = Conversation.objects.create(user=user_alice)
    assert build(plain, None, None)[0]["content"] == DEFAULT_SYSTEM_PROMPT
