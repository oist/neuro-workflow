"""Tests for the brain-viewer chat tools (vendored functions + resolver + endpoint)."""

import json
import math

import pytest
from app.workflow.models import FlowProject
from app.workflow.viewer_tools import (
    ViewerData,
    explain_activity,
    get_activity,
    get_connections,
    highlight_region,
    list_signals,
    node_strength,
)
from app.workflow.viewer_tools import resolver as vt_resolver
from app.workflow.viewer_tools import search_regions
from django.urls import reverse

REGION_DESC = {
    "groups": {"DLP": "Dorsolateral prefrontal cortex"},
    "regions": {
        "A10": {
            "full_name": "Area 10",
            "group": "DLP",
            "lobe": "Frontal",
            "description": "frontal pole, executive functions",
            "keywords": ["planning", "executive"],
        },
        "FST": {
            "full_name": "Fundus of superior temporal area",
            "group": "TE",
            "lobe": "Temporal",
            "description": "motion-sensitive visual area",
            "keywords": ["motion", "face recognition", "visual"],
        },
    },
}


def _synthetic_connectivity(with_signal=True):
    regions = [
        {"name": "L_A10", "x": 0, "y": 0, "z": 0, "hemi": "L", "area": 10.0},
        {"name": "R_A10", "x": 1, "y": 0, "z": 0, "hemi": "R", "area": 11.0},
        {"name": "L_FST", "x": 2, "y": 0, "z": 0, "hemi": "L", "area": 9.0},
        {"name": "R_FST", "x": 3, "y": 0, "z": 0, "hemi": "R", "area": 8.0},
    ]
    conn = {
        "meta": {"species": "marmoset", "n_regions": 4},
        "regions": regions,
        "connections": [[0, 1, 0.9, 5.0], [0, 2, 0.5, 7.0], [2, 3, 0.7, 6.0]],
    }
    if with_signal:
        T = 40
        conn["temporal_average"] = {
            "time": [i * 2.0 for i in range(T)],
            "data": [[math.sin(i / 5) + k for k in range(4)] for i in range(T)],
        }
    return conn


def _vd(with_signal=True):
    return ViewerData(
        connectivity=_synthetic_connectivity(with_signal), region_desc=REGION_DESC
    )


class _StubProject:
    def __init__(self, pid):
        self.id = pid


# --- vendored function unit tests (no DB) -----------------------------------


def test_search_regions_returns_labels():
    hits = search_regions(_vd(), "the area for face recognition and motion")
    assert hits
    assert any(h["label"].endswith("FST") for h in hits)
    assert all("score" in h for h in hits)


def test_explain_activity_bundles_metrics():
    exp = explain_activity(_vd(), "L_FST", 0, 100)
    assert exp["region"]["label"] == "L_FST"
    assert exp["metrics"]
    assert "shape" in exp


def test_no_signal_paths():
    vd = _vd(with_signal=False)
    assert list_signals(vd)["has_signal"] is False
    with pytest.raises(ValueError):
        get_activity(vd, "L_A10", 0, 100)


def test_structure_works_without_signal():
    vd = _vd(with_signal=False)
    conns = get_connections(vd, "L_A10")
    assert conns and conns[0]["target_label"] in {"R_A10", "L_FST"}
    assert node_strength(vd, "L_A10")["degree"] == 2


def test_group5_returns_action_dict():
    assert highlight_region(_vd(), "L_A10") == {
        "action": "select_region",
        "index": 0,
        "label": "L_A10",
    }


# --- resolver tests ----------------------------------------------------------


def test_resolver_discovers_viewer_data(tmp_path, monkeypatch):
    viewer_dir = tmp_path / "results" / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "connectivity_data.json").write_text(
        json.dumps(_synthetic_connectivity())
    )
    monkeypatch.setattr(vt_resolver, "existing_project_dir", lambda project: tmp_path)

    vd = vt_resolver.load_project_viewer_data(_StubProject("proj-discover"))
    assert vd.has_signal
    assert vd.species == "marmoset"


def test_resolver_strips_cachebuster_query(tmp_path, monkeypatch):
    viewer_dir = tmp_path / "results" / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "connectivity_data.json").write_text(
        json.dumps(_synthetic_connectivity())
    )
    monkeypatch.setattr(vt_resolver, "existing_project_dir", lambda project: tmp_path)

    # The viewer's data_url carries a ?_ts=... cache-buster; it must be ignored.
    vd = vt_resolver.load_project_viewer_data(
        _StubProject("proj-ts"),
        data_path="results/viewer/connectivity_data.json?_ts=1784543680139",
    )
    assert vd.has_signal


def test_resolver_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(vt_resolver, "existing_project_dir", lambda project: tmp_path)
    with pytest.raises(ValueError):
        vt_resolver.load_project_viewer_data(
            _StubProject("proj-traverse"), data_path="../../etc/passwd"
        )


def test_resolver_missing_data_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vt_resolver, "existing_project_dir", lambda project: tmp_path)
    with pytest.raises(vt_resolver.ViewerDataNotFound):
        vt_resolver.load_project_viewer_data(_StubProject("proj-empty"))


# --- endpoint tests ----------------------------------------------------------


@pytest.mark.django_db
def test_viewer_chat_endpoint_ok(auth_client, user_alice, tmp_path, monkeypatch):
    project = FlowProject.objects.create(name="V", owner=user_alice)
    viewer_dir = tmp_path / "results" / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "connectivity_data.json").write_text(
        json.dumps(_synthetic_connectivity())
    )
    monkeypatch.setattr(vt_resolver, "existing_project_dir", lambda project: tmp_path)

    url = reverse("workflow:workflow-viewer-chat", args=[project.id])
    resp = auth_client(user_alice).post(
        url, {"tool": "list_signals", "args": {}}, format="json"
    )
    assert resp.status_code == 200
    assert resp.json()["has_signal"] is True


@pytest.mark.django_db
def test_viewer_chat_endpoint_action_tool(
    auth_client, user_alice, tmp_path, monkeypatch
):
    project = FlowProject.objects.create(name="V", owner=user_alice)
    viewer_dir = tmp_path / "results" / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "connectivity_data.json").write_text(
        json.dumps(_synthetic_connectivity())
    )
    monkeypatch.setattr(vt_resolver, "existing_project_dir", lambda project: tmp_path)

    url = reverse("workflow:workflow-viewer-chat", args=[project.id])
    resp = auth_client(user_alice).post(
        url, {"tool": "highlight_region", "args": {"region": "L_A10"}}, format="json"
    )
    assert resp.status_code == 200
    assert resp.json()["action"] == "select_region"


@pytest.mark.django_db
def test_viewer_chat_endpoint_no_signal_is_graceful(
    auth_client, user_alice, tmp_path, monkeypatch
):
    project = FlowProject.objects.create(name="V", owner=user_alice)
    viewer_dir = tmp_path / "results" / "viewer"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "connectivity_data.json").write_text(
        json.dumps(_synthetic_connectivity(with_signal=False))
    )
    monkeypatch.setattr(vt_resolver, "existing_project_dir", lambda project: tmp_path)

    url = reverse("workflow:workflow-viewer-chat", args=[project.id])
    resp = auth_client(user_alice).post(
        url,
        {
            "tool": "get_activity",
            "args": {"region": "L_A10", "t_start": 0, "t_end": 10},
        },
        format="json",
    )
    # No-signal is surfaced as a 200 status field, not a 500.
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_signal_or_bad_arg"


@pytest.mark.django_db
def test_viewer_chat_endpoint_forbidden_for_non_owner(
    auth_client, user_alice, user_bob
):
    project = FlowProject.objects.create(name="V", owner=user_alice)
    url = reverse("workflow:workflow-viewer-chat", args=[project.id])
    resp = auth_client(user_bob).post(
        url, {"tool": "list_signals", "args": {}}, format="json"
    )
    assert resp.status_code in (403, 404)


@pytest.mark.django_db
def test_viewer_chat_endpoint_rejects_non_string_tool(auth_client, user_alice):
    # A non-string 'tool' would be an unhashable TOOL_REGISTRY key -> clean 400,
    # not a 500. Validation runs before any data load, so no file is needed.
    project = FlowProject.objects.create(name="V", owner=user_alice)
    url = reverse("workflow:workflow-viewer-chat", args=[project.id])
    resp = auth_client(user_alice).post(
        url, {"tool": {"nested": 1}, "args": {}}, format="json"
    )
    assert resp.status_code == 400


@pytest.mark.django_db
def test_viewer_chat_endpoint_rejects_non_dict_args(auth_client, user_alice):
    # A non-dict 'args' would break args.items() in the registry -> clean 400.
    project = FlowProject.objects.create(name="V", owner=user_alice)
    url = reverse("workflow:workflow-viewer-chat", args=[project.id])
    resp = auth_client(user_alice).post(
        url, {"tool": "list_signals", "args": [1, 2]}, format="json"
    )
    assert resp.status_code == 400
