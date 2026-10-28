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
