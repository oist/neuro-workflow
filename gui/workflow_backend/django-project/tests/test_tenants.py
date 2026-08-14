"""Tenant isolation for FlowProject list/detail and Jupyter visible-paths."""
import pytest
from django.urls import reverse

from app.tenants import (
    TENANT_HACKATHON,
    TENANT_INTERNAL,
    get_user_tenant,
    hub_username_for_tenant,
    set_user_tenant,
)
from app.workflow.models import FlowProject
from app.workflow.viewer_tokens import mint_viewer_token

pytestmark = pytest.mark.django_db


def _make_project(owner, *, visibility="private", name="P", tenant=None):
    if tenant is None:
        tenant = get_user_tenant(owner)
    return FlowProject.objects.create(
        name=name, owner=owner, visibility=visibility, tenant=tenant
    )


@pytest.fixture
def user_guest(db, django_user_model):
    user = django_user_model.objects.create_user(
        username="guest-sub-uuid", email="guest@example.com"
    )
    set_user_tenant(user, TENANT_HACKATHON)
    return user


def test_default_tenant_is_internal(user_alice):
    project = FlowProject.objects.create(name="X", owner=user_alice)
    assert project.tenant == TENANT_INTERNAL
    assert get_user_tenant(user_alice) == TENANT_INTERNAL


def test_create_assigns_caller_tenant(auth_client, user_guest):
    client = auth_client(user_guest)
    list_url = reverse("workflow:workflow-list-create")
    resp = client.post(list_url, {"name": "GuestProj", "tenant": "internal"}, format="json")
    assert resp.status_code == 201
    project = FlowProject.objects.get(id=resp.json()["id"])
    assert project.tenant == TENANT_HACKATHON
    assert resp.json()["tenant"] == TENANT_HACKATHON


def test_internal_user_cannot_see_hackathon_public(
    auth_client, user_alice, user_guest
):
    project = _make_project(
        user_guest, visibility="public", name="GuestPublic", tenant=TENANT_HACKATHON
    )
    client = auth_client(user_alice)

    list_url = reverse("workflow:workflow-list-create")
    resp = client.get(list_url)
    assert resp.status_code == 200
    ids = [p["id"] for p in resp.json()]
    assert str(project.id) not in ids

    detail_url = reverse("workflow:workflow-detail", args=[project.id])
    assert client.get(detail_url).status_code == 404
    assert client.patch(detail_url, {"description": "x"}, format="json").status_code == 404


def test_hackathon_user_cannot_see_internal_public(
    auth_client, user_alice, user_guest
):
    project = _make_project(
        user_alice, visibility="public", name="InternalPublic", tenant=TENANT_INTERNAL
    )
    client = auth_client(user_guest)
    detail_url = reverse("workflow:workflow-detail", args=[project.id])
    assert client.get(detail_url).status_code == 404

    list_url = reverse("workflow:workflow-list-create")
    resp = client.get(list_url)
    ids = [p["id"] for p in resp.json()]
    assert str(project.id) not in ids


def test_same_tenant_public_still_visible(auth_client, user_alice, user_bob):
    project = _make_project(user_alice, visibility="public")
    resp = auth_client(user_bob).get(
        reverse("workflow:workflow-detail", args=[project.id])
    )
    assert resp.status_code == 200


def test_jupyter_session_and_visible_paths(auth_client, user_alice, user_bob, user_guest):
    own = _make_project(user_alice, visibility="private", name="AlicePrivate")
    pub = _make_project(user_alice, visibility="public", name="AlicePublic")
    bob_private = _make_project(user_bob, visibility="private", name="BobPrivate")
    guest_pub = _make_project(
        user_guest, visibility="public", name="GuestPublic", tenant=TENANT_HACKATHON
    )

    session = auth_client(user_alice).get(reverse("workflow:jupyter-session"))
    assert session.status_code == 200
    body = session.json()
    assert body["tenant"] == TENANT_INTERNAL
    assert body["hub_user"] == hub_username_for_tenant(TENANT_INTERNAL)
    assert body["viewer_token"]

    token = body["viewer_token"]
    client = auth_client()
    resp = client.get(
        reverse("workflow:jupyter-visible-paths"),
        HTTP_AUTHORIZATION=f"Viewer {token}",
    )
    assert resp.status_code == 200
    ids = set(resp.json()["project_ids"])
    assert str(own.id) in ids
    assert str(pub.id) in ids
    assert str(bob_private.id) not in ids
    assert str(guest_pub.id) not in ids


def test_minted_token_matches_user(user_alice):
    token = mint_viewer_token(user_alice)
    from app.workflow.viewer_tokens import user_from_viewer_token

    user, payload = user_from_viewer_token(token)
    assert user.id == user_alice.id
    assert payload["tenant"] == TENANT_INTERNAL
