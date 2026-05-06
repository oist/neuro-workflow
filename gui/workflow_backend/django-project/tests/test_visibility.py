"""Tests for FlowProject Public/Private visibility and permissions."""
import pytest
from django.urls import reverse

from app.workflow.models import FlowProject, FlowNode, WorkflowRun

pytestmark = pytest.mark.django_db


def _make_project(owner, *, visibility="private", name="P"):
    return FlowProject.objects.create(name=name, owner=owner, visibility=visibility)


# ---------------------------------------------------------------------------
# Migration / default behavior
# ---------------------------------------------------------------------------

def test_default_visibility_is_private(user_alice):
    project = FlowProject.objects.create(name="X", owner=user_alice)
    assert project.visibility == FlowProject.Visibility.PRIVATE


# ---------------------------------------------------------------------------
# (a) Owner can manage own private project
# ---------------------------------------------------------------------------

def test_owner_can_get_patch_delete_own_private(auth_client, user_alice):
    project = _make_project(user_alice, visibility="private")
    client = auth_client(user_alice)
    detail_url = reverse("workflow:workflow-detail", args=[project.id])

    resp = client.get(detail_url)
    assert resp.status_code == 200
    assert resp.json()["visibility"] == "private"
    assert resp.json()["is_owned_by_me"] is True

    resp = client.patch(detail_url, {"description": "edited"}, format="json")
    assert resp.status_code == 200

    resp = client.delete(detail_url)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# (b) Stranger cannot see/access others' private
# ---------------------------------------------------------------------------

def test_stranger_cannot_see_others_private(auth_client, user_alice, user_bob):
    project = _make_project(user_alice, visibility="private")
    client = auth_client(user_bob)

    list_url = reverse("workflow:workflow-list-create")
    resp = client.get(list_url)
    assert resp.status_code == 200
    ids = [p["id"] for p in resp.json()]
    assert str(project.id) not in ids

    detail_url = reverse("workflow:workflow-detail", args=[project.id])
    assert client.get(detail_url).status_code == 404
    assert client.patch(detail_url, {"description": "x"}, format="json").status_code == 404
    assert client.delete(detail_url).status_code == 404


# ---------------------------------------------------------------------------
# (c) Stranger can read+edit but not delete or change-visibility on public
# ---------------------------------------------------------------------------

def test_stranger_on_public(auth_client, user_alice, user_bob):
    project = _make_project(user_alice, visibility="public")
    client = auth_client(user_bob)
    detail_url = reverse("workflow:workflow-detail", args=[project.id])

    resp = client.get(detail_url)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_owned_by_me"] is False
    assert body["can_edit"] is True
    assert body["can_delete"] is False
    assert body["can_change_visibility"] is False

    resp = client.patch(detail_url, {"description": "edited by bob"}, format="json")
    assert resp.status_code == 200
    project.refresh_from_db()
    assert project.description == "edited by bob"

    resp = client.patch(detail_url, {"visibility": "private"}, format="json")
    assert resp.status_code == 403
    project.refresh_from_db()
    assert project.visibility == "public"

    resp = client.delete(detail_url)
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# (d) Unauthenticated requests are rejected
# ---------------------------------------------------------------------------

def test_unauthenticated_is_rejected(auth_client):
    client = auth_client()
    list_url = reverse("workflow:workflow-list-create")
    assert client.get(list_url).status_code == 401
    assert client.post(list_url, {"name": "X"}, format="json").status_code == 401


# ---------------------------------------------------------------------------
# (e) Child resources (FlowNode) inherit parent visibility
# ---------------------------------------------------------------------------

def test_node_creation_respects_parent_visibility(auth_client, user_alice, user_bob):
    private_proj = _make_project(user_alice, visibility="private")
    public_proj = _make_project(user_alice, visibility="public", name="Q")
    client = auth_client(user_bob)

    private_nodes_url = reverse(
        "workflow:workflow-node-list-create", args=[private_proj.id]
    )
    payload = {
        "id": "n1",
        "position": {"x": 0, "y": 0},
        "type": "default",
        "data": {"nodeType": "analysis"},
    }
    resp = client.post(private_nodes_url, payload, format="json")
    assert resp.status_code == 404

    public_nodes_url = reverse(
        "workflow:workflow-node-list-create", args=[public_proj.id]
    )
    resp = client.post(public_nodes_url, payload, format="json")
    assert resp.status_code in (200, 201)
    assert FlowNode.objects.filter(project=public_proj, id="n1").exists()


# ---------------------------------------------------------------------------
# (g)/(h) WorkflowRun ownership = executor; list shows own + project-owner runs
# ---------------------------------------------------------------------------

def test_workflowrun_ownership_and_listing(auth_client, user_alice, user_bob):
    project = _make_project(user_alice, visibility="public")

    # Bob's run on Alice's public project
    run_bob = WorkflowRun.objects.create(workflow=project, user=user_bob)

    # Alice's own run
    run_alice = WorkflowRun.objects.create(workflow=project, user=user_alice)

    # A third user's run that neither alice nor bob should see
    third = user_alice.__class__.objects.create_user(
        username="carol-sub", email="carol@example.com"
    )
    _ = WorkflowRun.objects.create(workflow=project, user=third)

    list_url = reverse("workflow:workflow-run-list", args=[project.id])

    # Bob (executor) sees own run only
    resp = auth_client(user_bob).get(list_url)
    assert resp.status_code == 200
    ids = {r["id"] for r in resp.json()}
    assert str(run_bob.id) in ids
    assert str(run_alice.id) not in ids

    # Alice (project owner) sees all runs on her project
    resp = auth_client(user_alice).get(list_url)
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_workflowrun_detail_access(auth_client, user_alice, user_bob):
    project = _make_project(user_alice, visibility="public")
    run = WorkflowRun.objects.create(workflow=project, user=user_bob)

    detail_url = reverse("workflow:workflow-run-detail", args=[project.id, run.id])

    # Executor sees it
    assert auth_client(user_bob).get(detail_url).status_code == 200
    # Project owner sees it
    assert auth_client(user_alice).get(detail_url).status_code == 200

    # Unrelated user does not see it
    third = user_alice.__class__.objects.create_user(
        username="dave-sub", email="dave@example.com"
    )
    resp = auth_client(third).get(detail_url)
    # dave can access the public project but not someone else's run record
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Visibility flip is owner-only and works in both directions
# ---------------------------------------------------------------------------

def test_owner_can_toggle_visibility_both_directions(auth_client, user_alice):
    project = _make_project(user_alice, visibility="private")
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-detail", args=[project.id])

    resp = client.patch(url, {"visibility": "public"}, format="json")
    assert resp.status_code == 200
    project.refresh_from_db()
    assert project.visibility == "public"

    resp = client.patch(url, {"visibility": "private"}, format="json")
    assert resp.status_code == 200
    project.refresh_from_db()
    assert project.visibility == "private"


# ---------------------------------------------------------------------------
# create endpoint always assigns request.user as owner regardless of payload
# ---------------------------------------------------------------------------

def test_create_assigns_owner_to_request_user(auth_client, user_alice, user_bob):
    client = auth_client(user_bob)
    list_url = reverse("workflow:workflow-list-create")
    resp = client.post(
        list_url,
        {"name": "MyProj", "owner": user_alice.id},  # owner in payload should be ignored
        format="json",
    )
    assert resp.status_code == 201
    project = FlowProject.objects.get(id=resp.json()["id"])
    assert project.owner_id == user_bob.id
    assert project.visibility == "private"
