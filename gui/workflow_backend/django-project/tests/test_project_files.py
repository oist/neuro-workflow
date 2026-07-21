"""Tests for project data file upload / list / delete API."""
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from app.workflow.models import FlowProject
from app.workflow.path_utils import PROJECT_UPLOAD_MAX_BYTES, projects_root

pytestmark = pytest.mark.django_db


def _make_project(owner, *, visibility="private", name="DataProj"):
    return FlowProject.objects.create(name=name, owner=owner, visibility=visibility)


def _files_url(project):
    return reverse("workflow:workflow-project-files", args=[project.id])


def test_upload_list_delete_roundtrip(auth_client, user_alice, tmp_path, settings):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    url = _files_url(project)

    payload = b"col1,col2\n1,2\n"
    upload = SimpleUploadedFile("demo.csv", payload, content_type="text/csv")
    resp = client.post(url, {"file": upload}, format="multipart")
    assert resp.status_code == 201, resp.content
    body = resp.json()
    assert body["status"] == "success"
    assert body["uploaded"][0]["filename"] == "demo.csv"
    assert body["uploaded"][0]["size_bytes"] == len(payload)

    dest = projects_root() / str(project.id) / "demo.csv"
    assert dest.is_file()
    assert dest.read_bytes() == payload

    resp = client.get(url)
    assert resp.status_code == 200
    names = [f["filename"] for f in resp.json()["files"]]
    assert "demo.csv" in names
    assert resp.json()["max_bytes"] == PROJECT_UPLOAD_MAX_BYTES

    resp = client.delete(f"{url}?filename=demo.csv")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert not dest.exists()


def test_reject_path_traversal_and_protected(auth_client, user_alice, tmp_path, settings):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    url = _files_url(project)

    from app.workflow.path_utils import safe_project_upload_path
    import pytest as _pytest

    with _pytest.raises(ValueError, match="Invalid upload filename"):
        safe_project_upload_path(project, "../escape.csv")
    with _pytest.raises(ValueError, match="Invalid upload filename"):
        safe_project_upload_path(project, "subdir/data.csv")
    with _pytest.raises(ValueError, match="protected"):
        safe_project_upload_path(project, "workflow.py")

    protected = SimpleUploadedFile("workflow.py", b"print(1)\n", content_type="text/x-python")
    resp = client.post(url, {"file": protected}, format="multipart")
    assert resp.status_code == 400
    assert "protected" in resp.json()["errors"][0]["error"].lower()


def test_reject_disallowed_extension(auth_client, user_alice, tmp_path, settings):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    url = _files_url(project)

    exe = SimpleUploadedFile("malware.exe", b"MZ", content_type="application/octet-stream")
    resp = client.post(url, {"file": exe}, format="multipart")
    assert resp.status_code == 400
    assert "not allowed" in resp.json()["errors"][0]["error"].lower()


def test_overwrite_requires_flag(auth_client, user_alice, tmp_path, settings):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    url = _files_url(project)

    first = SimpleUploadedFile("data.csv", b"a\n", content_type="text/csv")
    assert client.post(url, {"file": first}, format="multipart").status_code == 201

    second = SimpleUploadedFile("data.csv", b"b\n", content_type="text/csv")
    resp = client.post(url, {"file": second}, format="multipart")
    assert resp.status_code == 400

    third = SimpleUploadedFile("data.csv", b"c\n", content_type="text/csv")
    resp = client.post(url, {"file": third, "overwrite": "true"}, format="multipart")
    assert resp.status_code == 201
    assert (projects_root() / str(project.id) / "data.csv").read_bytes() == b"c\n"


def test_stranger_cannot_upload_private(auth_client, user_alice, user_bob, tmp_path, settings):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice, visibility="private")
    client = auth_client(user_bob)
    url = _files_url(project)
    upload = SimpleUploadedFile("x.csv", b"1\n", content_type="text/csv")
    assert client.post(url, {"file": upload}, format="multipart").status_code == 404


def test_public_non_owner_can_upload(auth_client, user_alice, user_bob, tmp_path, settings):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice, visibility="public")
    client = auth_client(user_bob)
    url = _files_url(project)
    upload = SimpleUploadedFile("shared.csv", b"1\n", content_type="text/csv")
    resp = client.post(url, {"file": upload}, format="multipart")
    assert resp.status_code == 201, resp.content
