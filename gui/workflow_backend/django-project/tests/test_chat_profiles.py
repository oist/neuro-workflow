"""Tests for browser chat profiles (per-user MCP tool allowlist + prompt).

Covered:
- Profile CRUD is scoped to the owner.
- Serializer validation (unique name per user, allowed_tools shape).
- The allowlist filters the tools offered to OpenAI, and an empty allowlist
  skips MCP entirely.
- System prompt precedence: profile > conversation > default.
"""

import pytest
from asgiref.sync import async_to_sync
from django.urls import reverse
from rest_framework.test import APIClient

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


def test_profile_crud_is_owner_scoped(auth_client, user_alice, user_bob):
    alice = auth_client(user_alice)
    bob = auth_client(user_bob)

    resp = _create(alice)
    assert resp.status_code == 201, resp.json()
    profile_id = resp.json()["id"]

    names = [p["name"] for p in alice.get(reverse("chat-profiles")).json()]
    assert names == ["Viewer only"]
    assert bob.get(reverse("chat-profiles")).json() == []

    detail = reverse("chat-profile-detail", args=[profile_id])
    assert bob.get(detail).status_code == 404
    assert bob.put(detail, {"name": "x"}, format="json").status_code == 404
    assert bob.delete(detail).status_code == 404
    assert ChatProfile.objects.filter(id=profile_id).exists()

    resp = alice.put(
        detail, {"allowed_tools": ["add_node", "delete_node"]}, format="json"
    )
    assert resp.status_code == 200
    assert resp.json()["allowed_tools"] == ["add_node", "delete_node"]
    assert resp.json()["name"] == "Viewer only"  # partial update keeps name

    assert alice.delete(detail).status_code == 204
    assert not ChatProfile.objects.filter(id=profile_id).exists()


def test_duplicate_name_rejected_per_user(auth_client, user_alice, user_bob):
    alice = auth_client(user_alice)
    assert _create(alice).status_code == 201
    resp = _create(alice)
    assert resp.status_code == 400
    assert "name" in resp.json()
    # The same name is fine for another user.
    assert _create(auth_client(user_bob)).status_code == 201


def test_allowed_tools_validation(auth_client, user_alice):
    alice = auth_client(user_alice)
    assert _create(alice, name="a", allowed_tools="add_node").status_code == 400
    resp = _create(alice, name="b", allowed_tools=[{"name": "add_node"}])
    assert resp.status_code == 400

    resp = _create(alice, name="c", allowed_tools=["add_node", "add_node"])
    assert resp.status_code == 201
    assert resp.json()["allowed_tools"] == ["add_node"]

    resp = _create(alice, name="d", allowed_tools=[])
    assert resp.status_code == 201
    assert resp.json()["allowed_tools"] == []


def test_stream_rejects_foreign_profile(auth_client, user_alice, user_bob):
    bobs = ChatProfile.objects.create(user=user_bob, name="bob", allowed_tools=[])
    alice = auth_client(user_alice)
    resp = alice.post(
        reverse("chat-stream"),
        {"message": "hi", "profile_id": str(bobs.id)},
        format="json",
    )
    assert resp.status_code == 404
    # A bad profile id must not leave an orphan conversation behind.
    assert Conversation.objects.count() == 0


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
    profile = ChatProfile.objects.create(user=user_alice, name="none", allowed_tools=[])

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
    profile = ChatProfile.objects.create(
        user=user_alice, name="add only", allowed_tools=["add_node"]
    )

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


# --------------------------------------------------------------------------
# System prompt precedence
# --------------------------------------------------------------------------


def test_system_prompt_precedence(user_alice):
    conv = Conversation.objects.create(user=user_alice, system_prompt="CONV PROMPT")
    with_prompt = ChatProfile.objects.create(
        user=user_alice,
        name="p",
        allowed_tools=["add_node"],
        system_prompt="PROFILE PROMPT",
    )
    without_prompt = ChatProfile.objects.create(
        user=user_alice, name="q", allowed_tools=["add_node"]
    )
    build = async_to_sync(_build_openai_messages)

    assert build(conv, None, with_prompt)[0]["content"].startswith("PROFILE PROMPT")
    assert build(conv, None, without_prompt)[0]["content"].startswith("CONV PROMPT")
    assert build(conv, None, None)[0]["content"] == "CONV PROMPT"

    plain = Conversation.objects.create(user=user_alice)
    assert build(plain, None, None)[0]["content"] == DEFAULT_SYSTEM_PROMPT
