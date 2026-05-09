"""Regression tests guarding DRF DEFAULT_PERMISSION_CLASSES = IsAuthenticated.

If a future change reverts the default to AllowAny, or drops permission_classes
from a protected view, these tests fail.
"""
import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db


def test_drf_default_permission_is_isauthenticated(settings):
    perms = settings.REST_FRAMEWORK.get("DEFAULT_PERMISSION_CLASSES", [])
    assert "rest_framework.permissions.IsAuthenticated" in perms
    assert "rest_framework.permissions.AllowAny" not in perms


def test_health_check_allows_anonymous(auth_client):
    client = auth_client()
    resp = client.get(reverse("health_check"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_public_box_categories_allows_anonymous(auth_client):
    client = auth_client()
    resp = client.get(reverse("box:node-categories"))
    # The view may hit the filesystem and fail; the contract here is only
    # "not 401" — the explicit AllowAny on NodeCategoryListView must hold.
    assert resp.status_code != 401, (
        "box:node-categories must remain publicly readable."
    )


def test_protected_workflow_list_rejects_anonymous(auth_client):
    client = auth_client()
    url = reverse("workflow:workflow-list-create")
    assert client.get(url).status_code == 401
    assert client.post(url, {"name": "X"}, format="json").status_code == 401


def test_protected_workflow_list_accepts_authenticated(auth_client, user_alice):
    client = auth_client(user_alice)
    resp = client.get(reverse("workflow:workflow-list-create"))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
