"""Category colors are stored in the caller's node tree."""
import json

import pytest
from django.urls import reverse

from app.tenants import TENANT_COMMUNITY, TENANT_PROJECT, set_user_tenant

pytestmark = pytest.mark.django_db


def _roots(monkeypatch, tmp_path):
    project_nodes = tmp_path / "project-nodes"
    community_nodes = tmp_path / "community-nodes"
    for root in (project_nodes, community_nodes):
        (root / "analysis").mkdir(parents=True)

    def nodes_root(tenant=None):
        if tenant == TENANT_COMMUNITY:
            return community_nodes
        return project_nodes

    monkeypatch.setattr("app.box.views.nodes_root", nodes_root)
    return project_nodes, community_nodes


def test_community_color_save_writes_the_community_tree(
    auth_client, user_alice, monkeypatch, tmp_path
):
    project_nodes, community_nodes = _roots(monkeypatch, tmp_path)
    set_user_tenant(user_alice, TENANT_COMMUNITY)
    client = auth_client(user_alice)

    resp = client.post(
        reverse("box:node-categories"),
        {"category": "analysis", "color": "#112233"},
        format="json",
    )

    assert resp.status_code == 201
    saved = json.loads((community_nodes / "analysis" / ".settings").read_text())
    assert saved["color"] == "#112233"
    assert not (project_nodes / "analysis" / ".settings").exists()


def test_project_color_save_writes_the_project_tree(
    auth_client, user_alice, monkeypatch, tmp_path
):
    project_nodes, community_nodes = _roots(monkeypatch, tmp_path)
    set_user_tenant(user_alice, TENANT_PROJECT)
    client = auth_client(user_alice)

    resp = client.post(
        reverse("box:node-categories"),
        {"category": "analysis", "color": "#aabbcc"},
        format="json",
    )

    assert resp.status_code == 201
    saved = json.loads((project_nodes / "analysis" / ".settings").read_text())
    assert saved["color"] == "#aabbcc"
    assert not (community_nodes / "analysis" / ".settings").exists()


def test_category_color_save_rejects_a_path_outside_the_tree(
    auth_client, user_alice, monkeypatch, tmp_path
):
    project_nodes, _community_nodes = _roots(monkeypatch, tmp_path)
    client = auth_client(user_alice)

    resp = client.post(
        reverse("box:node-categories"),
        {"category": "../outside", "color": "#112233"},
        format="json",
    )

    assert resp.status_code == 400
    assert list(project_nodes.rglob(".settings")) == []
