"""Node governance: owner opening + review labels, tenant-scoped."""

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from app.box.models import NodeAuditLog, PythonFile
from app.tenants import (
    GROUP_NODE_REVIEWERS,
    TENANT_COMMUNITY,
    TENANT_PROJECT,
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
    set_user_tenant(user, TENANT_PROJECT)
    return user


@pytest.fixture
def guest(db, django_user_model):
    user = django_user_model.objects.create_user(
        username="guest-gov", email="guest-gov@example.com"
    )
    set_user_tenant(user, TENANT_COMMUNITY)
    return user


def _make_node(
    owner,
    *,
    tenant=TENANT_PROJECT,
    status=PythonFile.Status.PRIVATE,
    review_status=PythonFile.ReviewStatus.UNREVIEWED,
    name="n.py",
):
    return PythonFile.objects.create(
        name=name,
        category="analysis",
        file_content="class Foo:\n    pass\n",
        file_hash=f"hash-{owner.id}-{name}-{tenant}",
        uploaded_by=owner,
        tenant=tenant,
        status=status,
        review_status=review_status,
        is_analyzed=True,
        node_classes={
            "Foo": {
                "description": "",
                "inputs": {},
                "outputs": {},
                "parameters": {},
                "methods": {},
            }
        },
    )


def test_owner_submit_reviewer_approve_does_not_open(auth_client, user_alice, reviewer):
    node = _make_node(user_alice)
    submit_url = reverse("box:node-submit", args=[node.id])
    resp = auth_client(user_alice).post(submit_url, {}, format="json")
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.SUBMITTED
    assert node.review_status == PythonFile.ReviewStatus.IN_REVIEW

    approve_url = reverse("box:node-approve", args=[node.id])
    resp = auth_client(reviewer).post(approve_url, {"comment": "ok"}, format="json")
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.APPROVED
    assert node.review_status == PythonFile.ReviewStatus.REVIEWED
    assert NodeAuditLog.objects.filter(python_file=node, action="approved").exists()


def test_owner_can_publish_unreviewed(auth_client, user_alice, guest):
    node = _make_node(user_alice)
    resp = auth_client(user_alice).post(
        reverse("box:node-publish", args=[node.id]), {}, format="json"
    )
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.PUBLIC
    assert node.review_status == PythonFile.ReviewStatus.UNREVIEWED
    assert node.reviewed_by_id is None

    names = {
        n["file_name"]
        for n in auth_client(user_alice)
        .get(reverse("box:uploaded-nodes"))
        .json()["nodes"]
    }
    assert node.name in names
    guest_names = {
        n["file_name"]
        for n in auth_client(guest).get(reverse("box:uploaded-nodes")).json()["nodes"]
    }
    assert node.name not in guest_names


def test_owner_unpublish(auth_client, user_alice):
    node = _make_node(user_alice, status=PythonFile.Status.PUBLIC)
    resp = auth_client(user_alice).post(
        reverse("box:node-unpublish", args=[node.id]), {}, format="json"
    )
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.PRIVATE
    assert node.review_status == PythonFile.ReviewStatus.UNREVIEWED


def test_flag_on_blocks_unreviewed_owner_publish(
    auth_client, user_alice, reviewer, settings
):
    settings.NODE_PUBLISH_REQUIRES_REVIEW = True
    node = _make_node(user_alice)
    resp = auth_client(user_alice).post(
        reverse("box:node-publish", args=[node.id]), {}, format="json"
    )
    assert resp.status_code == 400
    node.review_status = PythonFile.ReviewStatus.REVIEWED
    node.status = PythonFile.Status.APPROVED
    node.save(update_fields=["review_status", "status"])
    resp = auth_client(user_alice).post(
        reverse("box:node-publish", args=[node.id]), {}, format="json"
    )
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.PUBLIC


def test_stranger_cannot_submit(auth_client, user_alice, user_bob):
    node = _make_node(user_alice)
    url = reverse("box:node-submit", args=[node.id])
    resp = auth_client(user_bob).post(url, {}, format="json")
    assert resp.status_code in (403, 404)


def test_non_reviewer_cannot_approve(auth_client, user_alice):
    node = _make_node(
        user_alice,
        status=PythonFile.Status.SUBMITTED,
        review_status=PythonFile.ReviewStatus.IN_REVIEW,
    )
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
        file_hash="hash-catalog-project",
        uploaded_by=None,
        tenant=TENANT_PROJECT,
        status=PythonFile.Status.PUBLIC,
        review_status=PythonFile.ReviewStatus.REVIEWED,
        is_analyzed=True,
        node_classes={
            "Cat": {
                "description": "",
                "inputs": {},
                "outputs": {},
                "parameters": {},
                "methods": {},
            }
        },
    )
    guest_public = _make_node(
        guest,
        tenant=TENANT_COMMUNITY,
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
    node = _make_node(
        user_alice,
        status=PythonFile.Status.SUBMITTED,
        review_status=PythonFile.ReviewStatus.IN_REVIEW,
    )
    resp = auth_client(reviewer).post(
        reverse("box:node-approve", args=[node.id]),
        {"make_public": True},
        format="json",
    )
    assert resp.status_code == 200
    guest_resp = auth_client(guest).get(reverse("box:uploaded-nodes"))
    names = {n["file_name"] for n in guest_resp.json()["nodes"]}
    assert node.name not in names


def test_owner_keeps_own_node_after_tenant_move(auth_client, user_alice, guest):
    node = _make_node(user_alice, name="moved.py")
    set_user_tenant(user_alice, TENANT_COMMUNITY)
    resp = auth_client(user_alice).get(reverse("box:uploaded-nodes"))
    names = {n["file_name"] for n in resp.json()["nodes"]}
    assert node.name in names
    guest_names = {
        n["file_name"]
        for n in auth_client(guest).get(reverse("box:uploaded-nodes")).json()["nodes"]
    }
    assert node.name not in guest_names


def test_approve_without_make_public_does_not_open(auth_client, user_alice, reviewer):
    node = _make_node(
        user_alice,
        status=PythonFile.Status.SUBMITTED,
        review_status=PythonFile.ReviewStatus.IN_REVIEW,
    )
    resp = auth_client(reviewer).post(
        reverse("box:node-approve", args=[node.id]), {}, format="json"
    )
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.APPROVED
    assert node.review_status == PythonFile.ReviewStatus.REVIEWED

    # After approve the node leaves the review queue, so the reviewer cannot
    # open it. Opening is the owner's action.
    resp = auth_client(reviewer).post(
        reverse("box:node-publish", args=[node.id]), {}, format="json"
    )
    assert resp.status_code == 404
    node.refresh_from_db()
    assert node.status == PythonFile.Status.APPROVED

    resp = auth_client(user_alice).post(
        reverse("box:node-publish", args=[node.id]), {}, format="json"
    )
    assert resp.status_code == 200, resp.content
    node.refresh_from_db()
    assert node.status == PythonFile.Status.PUBLIC


def test_reviewer_cannot_approve_own_node(auth_client, reviewer):
    node = _make_node(
        reviewer,
        status=PythonFile.Status.SUBMITTED,
        review_status=PythonFile.ReviewStatus.IN_REVIEW,
        name="self.py",
    )
    resp = auth_client(reviewer).post(
        reverse("box:node-approve", args=[node.id]),
        {"make_public": True},
        format="json",
    )
    assert resp.status_code == 403
    node.refresh_from_db()
    assert node.review_status == PythonFile.ReviewStatus.IN_REVIEW


def test_queue_lists_public_in_review(auth_client, user_alice, reviewer):
    node = _make_node(
        user_alice,
        status=PythonFile.Status.PUBLIC,
        review_status=PythonFile.ReviewStatus.IN_REVIEW,
        name="open-review.py",
    )
    resp = auth_client(reviewer).get(reverse("box:node-review-queue"))
    assert resp.status_code == 200
    names = {n["name"] for n in resp.json()["nodes"]}
    assert node.name in names


def test_copy_keeps_caller_tenant(auth_client, user_alice, guest, tmp_path, settings):
    settings.MEDIA_ROOT = str(tmp_path)
    (tmp_path / "analysis").mkdir()
    node = _make_node(user_alice, name="orig.py")
    set_user_tenant(user_alice, TENANT_COMMUNITY)
    resp = auth_client(user_alice).post(
        reverse("box:python-file-copy"),
        {"file_ids": [str(node.id)]},
        format="json",
    )
    assert resp.status_code == 201, resp.content
    copied = resp.json()["copied_files"][0]
    copied_id = copied["id"]
    copied_row = PythonFile.objects.get(pk=copied_id)
    assert copied_row.tenant == TENANT_COMMUNITY
    assert copied_row.status == PythonFile.Status.PRIVATE
    assert copied_row.review_status == PythonFile.ReviewStatus.UNREVIEWED


def test_identical_hash_does_not_steal_ownership(
    user_alice, user_bob, tmp_path, settings
):
    from app.box.services.python_file_service import PythonFileService

    settings.MEDIA_ROOT = str(tmp_path)
    (tmp_path / "analysis").mkdir()
    src = "class Foo:\n    pass\n"
    node = _make_node(user_alice, name="orig.py")
    node.file_hash = "will-be-replaced"
    node.file_content = src
    node.save()
    import hashlib

    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
    node.file_hash = digest
    node.save(update_fields=["file_hash"])

    service = PythonFileService()
    uploaded = SimpleUploadedFile(
        "copy.py", src.encode("utf-8"), content_type="text/x-python"
    )
    try:
        service.create_python_file(
            uploaded,
            user=user_bob,
            name="copy.py",
            category="analysis",
            tenant=TENANT_PROJECT,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
    node.refresh_from_db()
    assert node.uploaded_by_id == user_alice.id
