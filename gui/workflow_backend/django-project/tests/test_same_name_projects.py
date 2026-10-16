"""Projects that share a display name do not share a directory."""
import pytest
from app.tenants import TENANT_COMMUNITY, TENANT_PROJECT, set_user_tenant
from app.workflow.jupyter_views import visible_paths_payload
from app.workflow.models import FlowProject
from app.workflow.path_utils import code_file_path, existing_project_dir

pytestmark = pytest.mark.django_db


def _project(owner, name, tenant):
    return FlowProject.objects.create(
        name=name, owner=owner, visibility="public", tenant=tenant
    )


def test_shared_name_resolves_to_separate_uuid_directories(
    tmp_path, settings, user_alice, user_bob, monkeypatch
):
    settings.BASE_DIR = tmp_path
    monkeypatch.delenv("HOST_COMMUNITY_PATH", raising=False)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)
    first = _project(user_alice, "test", TENANT_PROJECT)
    second = _project(user_bob, "Test", TENANT_PROJECT)
    shared = tmp_path / "codes" / "projects" / "Test"
    shared.mkdir(parents=True)
    (shared / "workflow.py").write_text("shared")

    assert existing_project_dir(first) == tmp_path / "codes" / "projects" / str(first.id)
    assert existing_project_dir(second) == tmp_path / "codes" / "projects" / str(second.id)
    written = code_file_path(second, create=True)
    assert written.parent.name == str(second.id)
    assert (shared / "workflow.py").read_text() == "shared"


def test_unique_legacy_directory_is_still_used(tmp_path, settings, user_alice, monkeypatch):
    settings.BASE_DIR = tmp_path
    monkeypatch.delenv("HOST_COMMUNITY_PATH", raising=False)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)
    project = _project(user_alice, "Demo", TENANT_PROJECT)
    legacy = tmp_path / "codes" / "projects" / "Demo"
    legacy.mkdir(parents=True)

    assert existing_project_dir(project) == legacy


def test_jupyter_allow_list_omits_an_ambiguous_legacy_name(
    tmp_path, settings, user_alice, user_bob, monkeypatch
):
    settings.BASE_DIR = tmp_path
    monkeypatch.delenv("HOST_COMMUNITY_PATH", raising=False)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)
    (tmp_path / "codes-hackathon").mkdir()
    set_user_tenant(user_alice, TENANT_COMMUNITY)
    first = _project(user_alice, "test", TENANT_COMMUNITY)
    second = _project(user_bob, "test", TENANT_COMMUNITY)
    solo = _project(user_alice, "Solo", TENANT_COMMUNITY)

    payload = visible_paths_payload(user_alice)

    assert str(first.id) in payload["project_ids"]
    assert str(second.id) in payload["project_ids"]
    assert str(solo.id) in payload["project_ids"]
    assert "Test" not in payload["legacy_names"]
    assert "Solo" in payload["legacy_names"]
