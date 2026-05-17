"""Tests for FlowProject user-defined metadata (Issue #31)."""
import pytest
from django.urls import reverse

from app.workflow.models import FlowProject

pytestmark = pytest.mark.django_db


def test_default_metadata_is_empty_dict(user_alice):
    project = FlowProject.objects.create(name="X", owner=user_alice)
    assert project.metadata == {}


def test_create_with_metadata(auth_client, user_alice):
    client = auth_client(user_alice)
    list_url = reverse("workflow:workflow-list-create")
    resp = client.post(
        list_url,
        {
            "name": "WithMeta",
            "metadata": {"Affiliation": "OIST", "paper DOI": "10.1234/x"},
        },
        format="json",
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["metadata"] == {"Affiliation": "OIST", "paper DOI": "10.1234/x"}

    project = FlowProject.objects.get(id=body["id"])
    assert project.metadata == {"Affiliation": "OIST", "paper DOI": "10.1234/x"}


def test_patch_metadata(auth_client, user_alice):
    project = FlowProject.objects.create(name="P", owner=user_alice)
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-detail", args=[project.id])

    resp = client.patch(
        url, {"metadata": {"Funding": "Brain/MINDS 2.0"}}, format="json"
    )
    assert resp.status_code == 200
    assert resp.json()["metadata"] == {"Funding": "Brain/MINDS 2.0"}

    project.refresh_from_db()
    assert project.metadata == {"Funding": "Brain/MINDS 2.0"}


def test_metadata_replaces_on_patch(auth_client, user_alice):
    """PATCH replaces the metadata dict wholesale (no deep-merge)."""
    project = FlowProject.objects.create(
        name="P",
        owner=user_alice,
        metadata={"Affiliation": "OIST", "Collaborators": "A, B"},
    )
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-detail", args=[project.id])

    resp = client.patch(
        url, {"metadata": {"paper DOI": "10.9999/z"}}, format="json"
    )
    assert resp.status_code == 200

    project.refresh_from_db()
    assert project.metadata == {"paper DOI": "10.9999/z"}
    assert "Affiliation" not in project.metadata
    assert "Collaborators" not in project.metadata


def test_metadata_rejects_non_object(auth_client, user_alice):
    project = FlowProject.objects.create(name="P", owner=user_alice)
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-detail", args=[project.id])

    resp = client.patch(url, {"metadata": ["a", "b"]}, format="json")
    assert resp.status_code == 400
    assert "metadata" in resp.json()


def test_metadata_rejects_non_string_value(auth_client, user_alice):
    project = FlowProject.objects.create(name="P", owner=user_alice)
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-detail", args=[project.id])

    resp = client.patch(url, {"metadata": {"cpus": 4}}, format="json")
    assert resp.status_code == 400
    assert "metadata" in resp.json()

    project.refresh_from_db()
    assert project.metadata == {}
