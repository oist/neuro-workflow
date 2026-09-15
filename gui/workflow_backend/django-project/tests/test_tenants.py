"""Tenant isolation for FlowProject list/detail and Jupyter visible-paths."""

import pytest
from django.urls import reverse

from app.tenants import (
    TENANT_COMMUNITY,
    TENANT_PROJECT,
    get_user_tenant,
    hub_username_for_tenant,
    normalize_tenant,
    set_user_tenant,
    tenant_from_claims,
)
from app.workflow.models import FlowProject
from app.workflow.viewer_tokens import mint_viewer_token, unsign_viewer_token

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
    set_user_tenant(user, TENANT_COMMUNITY)
    return user


def test_default_tenant_is_project(user_alice):
    project = FlowProject.objects.create(name="X", owner=user_alice)
    assert project.tenant == TENANT_PROJECT
    assert get_user_tenant(user_alice) == TENANT_PROJECT


def test_normalize_tenant_maps_legacy_and_keeps_project():
    assert normalize_tenant("internal") == TENANT_PROJECT
    assert normalize_tenant("project") == TENANT_PROJECT
    assert normalize_tenant("hackathon") == TENANT_COMMUNITY
    assert normalize_tenant("community") == TENANT_COMMUNITY


def test_create_assigns_caller_tenant(auth_client, user_guest):
    client = auth_client(user_guest)
    list_url = reverse("workflow:workflow-list-create")
    resp = client.post(
        list_url, {"name": "GuestProj", "tenant": "internal"}, format="json"
    )
    assert resp.status_code == 201
    project = FlowProject.objects.get(id=resp.json()["id"])
    assert project.tenant == TENANT_COMMUNITY
    assert resp.json()["tenant"] == TENANT_COMMUNITY


def test_post_legacy_internal_slug_stores_project(auth_client, user_alice):
    resp = auth_client(user_alice).post(
        reverse("workflow:workflow-list-create"),
        {"name": "Proj", "tenant": "internal"},
        format="json",
    )
    assert resp.status_code == 201
    assert resp.json()["tenant"] == TENANT_PROJECT
    project = FlowProject.objects.get(id=resp.json()["id"])
    assert project.tenant == TENANT_PROJECT


def test_project_user_cannot_see_community_public(auth_client, user_alice, user_guest):
    project = _make_project(
        user_guest, visibility="public", name="GuestPublic", tenant=TENANT_COMMUNITY
    )
    client = auth_client(user_alice)

    list_url = reverse("workflow:workflow-list-create")
    resp = client.get(list_url)
    assert resp.status_code == 200
    ids = [p["id"] for p in resp.json()]
    assert str(project.id) not in ids

    detail_url = reverse("workflow:workflow-detail", args=[project.id])
    assert client.get(detail_url).status_code == 404
    assert (
        client.patch(detail_url, {"description": "x"}, format="json").status_code == 404
    )


def test_community_user_cannot_see_project_public(auth_client, user_alice, user_guest):
    project = _make_project(
        user_alice, visibility="public", name="ProjectPublic", tenant=TENANT_PROJECT
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


def test_jupyter_session_and_visible_paths(
    auth_client, user_alice, user_bob, user_guest
):
    own = _make_project(user_alice, visibility="private", name="AlicePrivate")
    pub = _make_project(user_alice, visibility="public", name="AlicePublic")
    bob_private = _make_project(user_bob, visibility="private", name="BobPrivate")
    guest_pub = _make_project(
        user_guest, visibility="public", name="GuestPublic", tenant=TENANT_COMMUNITY
    )

    session = auth_client(user_alice).get(reverse("workflow:jupyter-session"))
    assert session.status_code == 200
    body = session.json()
    assert body["tenant"] == TENANT_PROJECT
    assert body["hub_user"] == hub_username_for_tenant(TENANT_PROJECT)
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
    assert payload["tenant"] == TENANT_PROJECT


def test_legacy_viewer_token_tenant_still_matches(user_alice):
    token = mint_viewer_token(user_alice, tenant="internal")
    payload = unsign_viewer_token(token)
    assert payload["tenant"] == TENANT_PROJECT


def test_tenant_claims_match_exact_group_names():
    assert tenant_from_claims({"groups": ["/nw-internal"]}) == TENANT_PROJECT
    assert tenant_from_claims({"groups": ["nw-hackathon"]}) == TENANT_COMMUNITY
    assert tenant_from_claims({"groups": ["nw-project"]}) == TENANT_PROJECT
    assert tenant_from_claims({"groups": ["nw-community"]}) == TENANT_COMMUNITY
    assert tenant_from_claims({"groups": ["/teams/nw-internal-mentees"]}) is None
    assert tenant_from_claims({"groups": ["nw-internal-readonly"]}) is None
    assert (
        tenant_from_claims({"realm_access": {"roles": ["nw-internal-readonly"]}})
        is None
    )


def test_owner_still_lists_own_project_after_tenant_move(
    auth_client, user_alice, user_guest
):
    project = _make_project(user_alice, visibility="private", name="My Project")
    set_user_tenant(user_alice, TENANT_COMMUNITY)
    resp = auth_client(user_alice).get(reverse("workflow:workflow-list-create"))
    ids = [p["id"] for p in resp.json()]
    assert str(project.id) in ids
    guest_ids = [
        p["id"]
        for p in auth_client(user_guest)
        .get(reverse("workflow:workflow-list-create"))
        .json()
    ]
    assert str(project.id) not in guest_ids


def test_community_root_uses_live_hackathon_dir(tmp_path, settings):
    from app.workflow.path_utils import community_codes_root

    settings.BASE_DIR = tmp_path
    (tmp_path / "codes-hackathon").mkdir()
    assert community_codes_root() == tmp_path / "codes-hackathon"
    (tmp_path / "codes-community").mkdir()
    assert community_codes_root() == tmp_path / "codes-community"


def test_visible_paths_legacy_name_matches_disk(auth_client, user_alice):
    from app.workflow.path_utils import legacy_project_dir

    project = _make_project(user_alice, visibility="private", name="My Project")
    token = mint_viewer_token(user_alice)
    resp = auth_client().get(
        reverse("workflow:jupyter-visible-paths"),
        HTTP_AUTHORIZATION=f"Viewer {token}",
    )
    assert resp.status_code == 200
    names = set(resp.json()["legacy_names"])
    assert legacy_project_dir(project).name in names
