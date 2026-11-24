"""Creating a project also creates the folder Jupyter opens."""
import json

import pytest
from app.tenants import TENANT_COMMUNITY, TENANT_PROJECT, set_user_tenant
from app.workflow.models import FlowProject
from app.workflow.path_utils import community_codes_root
from django.urls import reverse

pytestmark = pytest.mark.django_db


@pytest.fixture
def community_user(db, django_user_model):
    user = django_user_model.objects.create_user(
        username="community-folder-user", email="community-folder@example.com"
    )
    set_user_tenant(user, TENANT_COMMUNITY)
    return user


def test_community_project_create_makes_its_own_jupyter_folder(
    auth_client, community_user, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    monkeypatch.delenv("HOST_COMMUNITY_PATH", raising=False)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)
    (tmp_path / "codes-hackathon").mkdir()
    client = auth_client(community_user)
    url = reverse("workflow:workflow-list-create")

    first = client.post(url, {"name": "First"}, format="json")
    assert first.status_code == 201, first.content
    first_id = first.json()["id"]
    first_dir = community_codes_root() / "projects" / first_id
    assert (first_dir / "workflow.py").is_file()
    assert (first_dir / "workflow.ipynb").is_file()
    first_py = (first_dir / "workflow.py").read_text(encoding="utf-8")

    second = client.post(url, {"name": "Second"}, format="json")
    assert second.status_code == 201, second.content
    second_id = second.json()["id"]
    second_dir = community_codes_root() / "projects" / second_id
    assert second_dir != first_dir
    assert (second_dir / "workflow.py").is_file()
    assert (second_dir / "workflow.ipynb").is_file()
    assert (first_dir / "workflow.py").read_text(encoding="utf-8") == first_py
    assert "Second" not in (first_dir / "workflow.py").read_text(encoding="utf-8")


def test_project_space_create_writes_the_project_tree(
    auth_client, django_user_model, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    monkeypatch.delenv("HOST_COMMUNITY_PATH", raising=False)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)
    (tmp_path / "codes-hackathon").mkdir()
    user = django_user_model.objects.create_user(
        username="project-folder-user", email="project-folder@example.com"
    )
    set_user_tenant(user, TENANT_PROJECT)
    client = auth_client(user)

    created = client.post(
        reverse("workflow:workflow-list-create"), {"name": "ProjectOne"}, format="json"
    )
    assert created.status_code == 201, created.content
    project_id = created.json()["id"]
    project_dir = tmp_path / "codes" / "projects" / project_id
    py_path = project_dir / "workflow.py"
    nb_path = project_dir / "workflow.ipynb"
    assert py_path.is_file()
    assert nb_path.is_file()

    py_text = py_path.read_text(encoding="utf-8")
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources = []
    for cell in notebook["cells"]:
        source = cell["source"]
        sources.append("".join(source) if isinstance(source, list) else source)
    assert "WorkflowBuilder" in py_text
    assert "WorkflowBuilder" in "\n".join(sources)
    assert not list((tmp_path / "codes-hackathon").rglob("workflow.py"))


def test_failed_notebook_write_removes_the_new_project(
    auth_client, django_user_model, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    monkeypatch.delenv("HOST_COMMUNITY_PATH", raising=False)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)
    (tmp_path / "codes-hackathon").mkdir()
    monkeypatch.setattr(
        "app.workflow.code_generation_service.CodeGenerationService._convert_py_to_ipynb",
        lambda self, project_id: False,
    )
    user = django_user_model.objects.create_user(
        username="failed-folder-user", email="failed-folder@example.com"
    )
    set_user_tenant(user, TENANT_COMMUNITY)
    client = auth_client(user)
    client.raise_request_exception = False

    created = client.post(
        reverse("workflow:workflow-list-create"), {"name": "Broken"}, format="json"
    )
    assert created.status_code >= 400
    assert not FlowProject.objects.filter(owner=user, is_active=True).exists()
