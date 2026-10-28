"""An owner cannot keep two active projects with the same name."""
import pytest
from app.workflow.models import FlowProject
from django.urls import reverse

pytestmark = pytest.mark.django_db


def test_duplicate_active_name_is_rejected(auth_client, user_alice):
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-list-create")
    first = client.post(url, {"name": "Same"}, format="json")
    assert first.status_code == 201, first.content

    second = client.post(url, {"name": "Same"}, format="json")
    assert second.status_code == 400
    assert second.json()["name"] == [
        "You already have an active project with this name."
    ]
    assert (
        FlowProject.objects.filter(owner=user_alice, name="Same", is_active=True).count()
        == 1
    )


def test_another_owner_may_reuse_the_name(auth_client, user_alice, user_bob):
    url = reverse("workflow:workflow-list-create")
    first = auth_client(user_alice).post(url, {"name": "Same"}, format="json")
    assert first.status_code == 201, first.content

    second = auth_client(user_bob).post(url, {"name": "Same"}, format="json")
    assert second.status_code == 201, second.content
    assert FlowProject.objects.filter(name="Same", is_active=True).count() == 2


def test_inactive_project_does_not_block_the_name(auth_client, user_alice):
    FlowProject.objects.create(name="Same", owner=user_alice, is_active=False)
    created = auth_client(user_alice).post(
        reverse("workflow:workflow-list-create"), {"name": "Same"}, format="json"
    )
    assert created.status_code == 201, created.content
    assert (
        FlowProject.objects.filter(owner=user_alice, name="Same", is_active=True).count()
        == 1
    )
