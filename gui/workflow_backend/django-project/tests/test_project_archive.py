"""Download one project's Jupyter folder as a zip."""

import io
import zipfile

import pytest
from app.workflow.models import FlowProject
from app.workflow.path_utils import stable_project_dir
from django.urls import reverse

pytestmark = pytest.mark.django_db


def _isolate_codes(settings, monkeypatch, tmp_path):
    settings.BASE_DIR = tmp_path
    monkeypatch.delenv("HOST_COMMUNITY_PATH", raising=False)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)


def _project(owner, name, **kwargs):
    return FlowProject.objects.create(name=name, owner=owner, **kwargs)


def _archive_url(project):
    return reverse("workflow:workflow-project-archive", args=[project.id])


def _zip_payload(response):
    return b"".join(response.streaming_content)


def _zip_entries(response):
    payload = _zip_payload(response)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    return payload, entries


def test_archive_zips_this_project_and_skips_the_rest(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    _isolate_codes(settings, monkeypatch, tmp_path)
    project = _project(user_alice, "Demo Project")
    root = stable_project_dir(project, create=True)
    (root / "workflow.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "notes.txt").write_text("notes", encoding="utf-8")

    checkpoint = root / ".ipynb_checkpoints"
    checkpoint.mkdir()
    (checkpoint / "notes.txt").write_text("checkpoint", encoding="utf-8")
    pycache = root / "pkg" / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "mod.pyc").write_bytes(b"pycache-bytes")
    (root / "pkg" / "real.py").write_text("print(1)\n", encoding="utf-8")

    outside = tmp_path / "outside-secret.txt"
    outside.write_text("OUTSIDE-SECRET", encoding="utf-8")
    (root / "linked.txt").symlink_to(outside)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "hidden.txt").write_text("HIDDEN-DIR", encoding="utf-8")
    (root / "linked-dir").symlink_to(outside_dir, target_is_directory=True)

    other = _project(user_alice, "Other")
    other_root = stable_project_dir(other, create=True)
    (other_root / "secret-other.txt").write_text("other-only", encoding="utf-8")

    response = auth_client(user_alice).get(_archive_url(project))

    assert response.status_code == 200, response.content
    assert response["Content-Type"] == "application/zip"
    assert 'filename="Demo_Project.zip"' in response["Content-Disposition"]
    payload, entries = _zip_entries(response)
    assert set(entries) == {"workflow.py", "data/notes.txt", "pkg/real.py"}
    assert b"OUTSIDE-SECRET" not in payload
    assert b"HIDDEN-DIR" not in payload
    assert b"checkpoint" not in payload
    assert b"pycache-bytes" not in payload
    assert b"other-only" not in payload
    assert entries["data/notes.txt"] == b"notes"


def test_archive_name_falls_back_to_the_project_id(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    _isolate_codes(settings, monkeypatch, tmp_path)
    project = _project(user_alice, "脳")
    root = stable_project_dir(project, create=True)
    (root / "workflow.py").write_text("print('ok')\n", encoding="utf-8")

    response = auth_client(user_alice).get(_archive_url(project))

    assert response.status_code == 200, response.content
    assert f'filename="{project.id}.zip"' in response["Content-Disposition"]


def test_private_project_is_not_found_for_another_user(
    auth_client, user_alice, user_bob, tmp_path, settings, monkeypatch
):
    _isolate_codes(settings, monkeypatch, tmp_path)
    project = _project(user_alice, "Private")
    root = stable_project_dir(project, create=True)
    (root / "workflow.py").write_text("secret\n", encoding="utf-8")

    response = auth_client(user_bob).get(_archive_url(project))

    assert response.status_code == 404


def test_symlink_project_directory_is_not_walked(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    _isolate_codes(settings, monkeypatch, tmp_path)
    project = _project(user_alice, "Linked")
    sibling = _project(user_alice, "Sibling")
    sibling_root = stable_project_dir(sibling, create=True)
    (sibling_root / "secret.txt").write_text("SIBLING-SECRET", encoding="utf-8")
    project_root = stable_project_dir(project, create=False)
    project_root.parent.mkdir(parents=True, exist_ok=True)
    project_root.symlink_to(sibling_root, target_is_directory=True)

    response = auth_client(user_alice).get(_archive_url(project))

    assert response.status_code == 404
    assert response.json()["error"] == "Project folder not found"
    assert b"SIBLING-SECRET" not in response.content


def test_missing_project_folder_is_not_found(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    _isolate_codes(settings, monkeypatch, tmp_path)
    project = _project(user_alice, "Missing")

    response = auth_client(user_alice).get(_archive_url(project))

    assert response.status_code == 404
    assert response.json()["error"] == "Project folder not found"


def test_public_project_downloads_for_another_user_in_the_tenant(
    auth_client, user_alice, user_bob, tmp_path, settings, monkeypatch
):
    _isolate_codes(settings, monkeypatch, tmp_path)
    project = _project(user_alice, "Shared", visibility="public")
    root = stable_project_dir(project, create=True)
    (root / "shared.txt").write_text("shared", encoding="utf-8")

    response = auth_client(user_bob).get(_archive_url(project))

    assert response.status_code == 200, response.content
    _, entries = _zip_entries(response)
    assert set(entries) == {"shared.txt"}
    assert entries["shared.txt"] == b"shared"
