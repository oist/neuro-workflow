"""Tests for saved notebook agent conversations (Issue #122).

The in-kernel agent records each turn with the service token and its own
JupyterHub token; anyone who can open the project reads them in the app. They
share the ``Conversation`` table with the browser chat but must never show up
in, or be continued from, the browser chat.
"""

from datetime import timedelta

import pytest
from django.urls import reverse
from django.utils import timezone
from rest_framework.test import APIClient

from app.chat.models import Conversation, NotebookToken
from app.workflow.models import FlowProject
from tests.test_notebook_token import KERNEL, service_token  # noqa: F401

pytestmark = pytest.mark.django_db

LIST_URL = reverse("chat-notebook-conversations")


def _detail_url(conversation_id):
    return reverse("chat-notebook-conversation-detail", args=[conversation_id])


def _project(owner, *, visibility="private", **kwargs):
    return FlowProject.objects.create(
        name="P", owner=owner, visibility=visibility, **kwargs
    )


def _relayed_by(user, project, *, expired=False):
    delta = timedelta(minutes=-5 if expired else 5)
    return NotebookToken.objects.create(
        project=project,
        user=user,
        hub_user="internal",
        access_token="jwt",
        expires_at=timezone.now() + delta,
    )


def _turn(project, conversation_id=None, **extra):
    return {
        "project_id": str(project.id),
        "conversation_id": conversation_id,
        "model": "claude-sonnet-5-5",
        "session_id": "sess-1",
        "messages": [
            {"role": "user", "content": "Plot the spike raster"},
            {
                "role": "assistant",
                "content": "Running it.",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "run_code", "arguments": '{"code": "1"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "tool_name": "run_code",
                "content": "NameError",
                "is_error": True,
            },
            {"role": "assistant", "content": "Done."},
        ],
        **extra,
    }


def _record(project, conversation_id=None, headers=KERNEL, **extra):
    return APIClient().post(
        LIST_URL, _turn(project, conversation_id, **extra), format="json", **headers
    )


# ---------------------------------------------------------------------------
# POST (kernel recording)
# ---------------------------------------------------------------------------


def test_kernel_records_a_turn(user_alice, service_token):  # noqa: F811
    project = _project(user_alice)
    _relayed_by(user_alice, project, expired=True)

    resp = _record(project)

    assert resp.status_code == 201
    conversation = Conversation.objects.get(id=resp.json()["id"])
    assert conversation.source == "notebook"
    assert conversation.project == project
    # The label comes from the relay row even after the token has expired.
    assert conversation.user == user_alice
    assert conversation.title == "Plot the spike raster"
    assert conversation.metadata == {
        "model": "claude-sonnet-5-5",
        "session_id": "sess-1",
        "hub_user": "internal",
    }
    roles = [m.role for m in conversation.messages.all()]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assistant = conversation.messages.get(role="assistant", content="Running it.")
    assert assistant.tool_calls[0]["function"]["name"] == "run_code"
    tool = conversation.messages.get(role="tool")
    assert tool.tool_call_id == "toolu_1"
    assert tool.raw_response == {"is_error": True}


def test_kernel_appends_to_its_conversation(user_alice, service_token):  # noqa: F811
    project = _project(user_alice)
    first = _record(project).json()["id"]

    resp = _record(project, first, session_id="sess-2")

    assert resp.status_code == 200
    assert resp.json()["id"] == first
    conversation = Conversation.objects.get(id=first)
    assert conversation.messages.count() == 8
    assert conversation.metadata["session_id"] == "sess-2"


def test_unknown_conversation_starts_a_new_one(
    user_alice, service_token  # noqa: F811
):
    project = _project(user_alice)
    other = _project(user_alice)
    browser = Conversation.objects.create(user=user_alice, project=project)
    elsewhere = _record(other).json()["id"]
    deleted = Conversation.objects.get(id=_record(project).json()["id"])
    deleted.is_active = False
    deleted.save()

    for conversation_id in (str(browser.id), elsewhere, str(deleted.id)):
        resp = _record(project, conversation_id)
        assert resp.status_code == 201
        assert resp.json()["id"] != conversation_id

    assert browser.messages.count() == 0
    assert Conversation.objects.get(id=elsewhere).messages.count() == 4


def test_kernel_records_without_relay_row(user_alice, service_token):  # noqa: F811
    project = _project(user_alice)

    resp = _record(project)

    assert resp.status_code == 201
    assert Conversation.objects.get(id=resp.json()["id"]).user is None


def test_kernel_recording_rejections(user_alice, service_token):  # noqa: F811
    project = _project(user_alice)
    community = _project(user_alice, tenant="community")
    inactive = _project(user_alice, is_active=False)

    assert _record(project, headers={}).status_code == 401
    wrong_key = {**KERNEL, "HTTP_X_API_KEY": "nope"}
    assert _record(project, headers=wrong_key).status_code == 401
    bad_hub = {**KERNEL, "HTTP_X_JUPYTERHUB_TOKEN": "forged"}
    assert _record(project, headers=bad_hub).status_code == 401
    # The project space's kernel cannot record into a community project.
    assert _record(community).status_code == 404
    assert _record(inactive).status_code == 404
    assert not Conversation.objects.exists()


def test_kernel_turn_needs_messages(user_alice, service_token):  # noqa: F811
    project = _project(user_alice)

    resp = _record(project, messages=[])

    assert resp.status_code == 400


def test_user_path_records_with_project_write_access(
    auth_client, user_alice, user_bob
):
    own = _project(user_alice)
    bobs = _project(user_bob)
    client = auth_client(user_alice)

    ok = client.post(LIST_URL, _turn(own), format="json")
    assert ok.status_code == 201
    assert Conversation.objects.get(id=ok.json()["id"]).user == user_alice
    assert client.post(LIST_URL, _turn(bobs), format="json").status_code == 404


# ---------------------------------------------------------------------------
# GET list / detail, DELETE (browser)
# ---------------------------------------------------------------------------


def test_list_is_project_scoped(
    auth_client, user_alice, user_bob, service_token  # noqa: F811
):
    user_alice.first_name, user_alice.last_name = "Alice", "A"
    user_alice.save()
    private = _project(user_alice)
    public = _project(user_alice, visibility="public")
    _relayed_by(user_alice, private)
    recorded = _record(private).json()["id"]
    _record(public)
    Conversation.objects.create(user=user_alice, project=private)  # browser chat
    gone = Conversation.objects.get(id=_record(private).json()["id"])
    gone.is_active = False
    gone.save()

    alice = auth_client(user_alice)
    assert alice.get(LIST_URL).status_code == 400
    rows = alice.get(LIST_URL, {"project_id": str(private.id)}).json()
    assert [r["id"] for r in rows] == [recorded]
    assert rows[0]["user_label"] == "Alice A"
    assert rows[0]["model"] == "claude-sonnet-5-5"
    assert rows[0]["message_count"] == 4
    assert rows[0]["can_delete"] is True

    bob = auth_client(user_bob)
    assert bob.get(LIST_URL, {"project_id": str(private.id)}).status_code == 404
    public_rows = bob.get(LIST_URL, {"project_id": str(public.id)}).json()
    assert len(public_rows) == 1
    assert public_rows[0]["can_delete"] is False


def test_list_is_newest_first(auth_client, user_alice, service_token):  # noqa: F811
    project = _project(user_alice)
    older = _record(project).json()["id"]
    newer = _record(project).json()["id"]

    def listed():
        rows = auth_client(user_alice).get(LIST_URL, {"project_id": str(project.id)})
        return [r["id"] for r in rows.json()]

    assert listed() == [newer, older]
    _record(project, older)  # a new turn moves it to the top
    assert listed() == [older, newer]


def test_detail_returns_messages(
    auth_client, user_alice, user_bob, service_token  # noqa: F811
):
    project = _project(user_alice)
    conversation_id = _record(project).json()["id"]

    resp = auth_client(user_alice).get(_detail_url(conversation_id))

    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert [m["is_error"] for m in messages] == [False, False, True, False]
    assert auth_client(user_bob).get(_detail_url(conversation_id)).status_code == 404


def test_delete_rights(
    auth_client, user_alice, user_bob, user_admin, service_token  # noqa: F811
):
    project = _project(user_alice, visibility="public")
    _relayed_by(user_bob, project)
    by_owner = _record(project).json()["id"]
    by_recorder = _record(project).json()["id"]
    untouched = _record(project).json()["id"]

    # Same tenant, can read the public project, but neither owner nor recorder.
    assert auth_client(user_admin).delete(_detail_url(untouched)).status_code == 403
    assert auth_client(user_alice).delete(_detail_url(by_owner)).status_code == 200
    assert auth_client(user_bob).delete(_detail_url(by_recorder)).status_code == 200

    # Soft delete: the rows stay, hidden from the list and detail.
    assert Conversation.objects.filter(id=by_owner, is_active=False).exists()
    assert auth_client(user_alice).get(_detail_url(by_owner)).status_code == 404
    rows = auth_client(user_alice).get(LIST_URL, {"project_id": str(project.id)})
    assert [r["id"] for r in rows.json()] == [untouched]


# ---------------------------------------------------------------------------
# Isolation from the browser chat
# ---------------------------------------------------------------------------


def test_browser_chat_never_sees_notebook_conversations(
    auth_client, user_alice, service_token  # noqa: F811
):
    project = _project(user_alice)
    _relayed_by(user_alice, project)
    conversation_id = _record(project).json()["id"]
    client = auth_client(user_alice)

    assert client.get(reverse("chat-conversations")).json() == []
    detail = reverse("chat-conversation-detail", args=[conversation_id])
    assert client.get(detail).status_code == 404
    assert client.delete(detail).status_code == 404
    stream = client.post(
        reverse("chat-stream"),
        {"message": "continue", "conversation_id": conversation_id},
        format="json",
    )
    assert stream.status_code == 404
    assert Conversation.objects.get(id=conversation_id).is_active
