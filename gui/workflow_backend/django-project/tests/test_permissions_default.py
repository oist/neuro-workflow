"""Regression tests guarding DRF secure-by-default settings.

If a future change reverts DEFAULT_PERMISSION_CLASSES to AllowAny, drops
DEFAULT_AUTHENTICATION_CLASSES, or removes permission_classes from a
protected view, these tests fail.
"""
import json

import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db


def test_drf_defaults_are_secure_by_default(settings):
    framework = settings.REST_FRAMEWORK
    perms = framework.get("DEFAULT_PERMISSION_CLASSES", [])
    auths = framework.get("DEFAULT_AUTHENTICATION_CLASSES", [])
    assert "rest_framework.permissions.IsAuthenticated" in perms
    assert "rest_framework.permissions.AllowAny" not in perms
    assert "app.auth.authentication.CombinedJWTAuthentication" in auths


def test_health_check_allows_anonymous(auth_client):
    client = auth_client()
    resp = client.get(reverse("health_check"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_public_box_categories_allows_anonymous(auth_client, settings, tmp_path):
    """AllowAny on NodeCategoryListView keeps the public endpoint reachable.

    Point MEDIA_ROOT at a tmp dir and pre-create category folders + default
    .settings files so the view exercises its success path (returns 200 with
    a categories list) rather than just any non-401.
    """
    from app.box.models import NODE_CATEGORIES

    settings.MEDIA_ROOT = str(tmp_path)
    for cat, _label in NODE_CATEGORIES:
        cat_dir = tmp_path / cat
        cat_dir.mkdir()
        (cat_dir / ".settings").write_text(json.dumps({"color": "#6b46c1"}))

    client = auth_client()
    resp = client.get(reverse("box:node-categories"))

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("categories"), list)
    assert body.get("default") == "analysis"


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
