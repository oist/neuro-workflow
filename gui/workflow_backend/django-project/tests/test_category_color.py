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
    assert resp.json()["error"] == "invalid category"
    assert list(project_nodes.rglob(".settings")) == []


def test_category_color_save_rejects_a_bad_color(
    auth_client, user_alice, monkeypatch, tmp_path
):
    _roots(monkeypatch, tmp_path)
    client = auth_client(user_alice)

    resp = client.post(
        reverse("box:node-categories"),
        {"category": "analysis", "color": "purple"},
        format="json",
    )

    assert resp.status_code == 400
    assert resp.json()["error"] == "color must be #rrggbb"


def test_community_category_list_records_a_default_for_a_missing_folder(
    auth_client, user_alice, monkeypatch, tmp_path, settings
):
    project_nodes = tmp_path / "project-nodes"
    community_nodes = tmp_path / "community-nodes"
    (project_nodes / "analysis").mkdir(parents=True)
    (project_nodes / "stimulus").mkdir()
    (community_nodes / "analysis").mkdir(parents=True)
    (community_nodes / "analysis" / ".settings").write_text(
        json.dumps({"color": "#abcdef"})
    )
    settings.MEDIA_ROOT = str(project_nodes)

    def nodes_root(tenant=None):
        if tenant == TENANT_COMMUNITY:
            return community_nodes
        return project_nodes

    monkeypatch.setattr("app.box.views.nodes_root", nodes_root)
    set_user_tenant(user_alice, TENANT_COMMUNITY)
    client = auth_client(user_alice)

    resp = client.get(reverse("box:node-categories"))

    assert resp.status_code == 200
    by_value = {item["value"]: item for item in resp.json()["categories"]}
    assert by_value["analysis"]["settings"]["color"] == "#abcdef"
    assert by_value["stimulus"]["settings"]["color"] == "#6b46c1"
    saved = json.loads((community_nodes / "stimulus" / ".settings").read_text())
    assert saved["color"] == "#6b46c1"
