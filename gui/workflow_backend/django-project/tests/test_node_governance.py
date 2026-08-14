"""Node governance pipeline is tenant-scoped."""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from app.box.models import NodeAuditLog, PythonFile
from app.tenants import (
    GROUP_NODE_REVIEWERS,
    TENANT_HACKATHON,
    TENANT_INTERNAL,
    ensure_tenant_groups,
    set_user_tenant,
)

pytestmark = pytest.mark.django_db


@pytest.fixture
def reviewer(db, django_user_model):
    user = django_user_model.objects.create_user(
        username="reviewer-sub", email="reviewer@example.com"
    )
    groups = ensure_tenant_groups()
    user.groups.add(groups[GROUP_NODE_REVIEWERS])
    set_user_tenant(user, TENANT_INTERNAL)
    return user


@pytest.fixture
def guest(db, django_user_model):
    user = django_user_model.objects.create_user(
        username="guest-gov", email="guest-gov@example.com"
    )
    set_user_tenant(user, TENANT_HACKATHON)
    return user


def _make_node(owner, *, tenant=TENANT_INTERNAL, status=PythonFile.Status.PRIVATE, name="n.py"):
    return PythonFile.objects.create(
        name=name,
        category="analysis",
        file_content="class Foo:\n    pass\n",
        file_hash=f"hash-{owner.id}-{name}-{tenant}",
        uploaded_by=owner,
        tenant=tenant,
        status=status,
        is_analyzed=True,
        node_classes={"Foo": {"description": "", "inputs": {}, "outputs": {}, "parameters": {}, "methods": {}}},
    )


def test_owner_submit_reviewer_approve_public(auth_client, user_alice, reviewer):
    node = _make_node(user_alice)
    submit_url = reverse("box:node-submit", args=[node.id])
    resp = auth_client(user_alice).post(submit_url, {}, format="json")
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.SUBMITTED

    approve_url = reverse("box:node-approve", args=[node.id])
    resp = auth_client(reviewer).post(
        approve_url, {"make_public": True, "comment": "ok"}, format="json"
    )
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.PUBLIC
    assert NodeAuditLog.objects.filter(python_file=node, action="published").exists()


def test_stranger_cannot_submit(auth_client, user_alice, user_bob):
    node = _make_node(user_alice)
    url = reverse("box:node-submit", args=[node.id])
    resp = auth_client(user_bob).post(url, {}, format="json")
    assert resp.status_code in (403, 404)


def test_non_reviewer_cannot_approve(auth_client, user_alice):
    node = _make_node(user_alice, status=PythonFile.Status.SUBMITTED)
    url = reverse("box:node-approve", args=[node.id])
    resp = auth_client(user_alice).post(url, {"make_public": True}, format="json")
    assert resp.status_code == 403


def test_palette_hides_other_private_and_other_tenant(
    auth_client, user_alice, user_bob, guest
):
    own = _make_node(user_alice, name="alice.py")
    bob_private = _make_node(user_bob, name="bob.py")
    catalog = PythonFile.objects.create(
        name="catalog.py",
        category="analysis",
        file_content="class Cat:\n    pass\n",
        file_hash="hash-catalog-internal",
        uploaded_by=None,
        tenant=TENANT_INTERNAL,
        status=PythonFile.Status.PUBLIC,
        is_analyzed=True,
        node_classes={"Cat": {"description": "", "inputs": {}, "outputs": {}, "parameters": {}, "methods": {}}},
    )
    guest_public = _make_node(
        guest,
        tenant=TENANT_HACKATHON,
        status=PythonFile.Status.PUBLIC,
        name="guest.py",
    )

    resp = auth_client(user_alice).get(reverse("box:uploaded-nodes"))
    assert resp.status_code == 200
    names = {n["file_name"] for n in resp.json()["nodes"]}
    assert own.name in names
    assert catalog.name in names
    assert bob_private.name not in names
    assert guest_public.name not in names


def test_approve_in_one_tenant_does_not_publish_to_the_other(
    auth_client, user_alice, reviewer, guest
):
    node = _make_node(user_alice, status=PythonFile.Status.SUBMITTED)
    resp = auth_client(reviewer).post(
        reverse("box:node-approve", args=[node.id]),
        {"make_public": True},
        format="json",
    )
    assert resp.status_code == 200
    guest_resp = auth_client(guest).get(reverse("box:uploaded-nodes"))
    names = {n["file_name"] for n in guest_resp.json()["nodes"]}
    assert node.name not in names
