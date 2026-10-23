"""Creating a project also creates the folder Jupyter opens."""
import pytest
from app.tenants import TENANT_COMMUNITY, set_user_tenant
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
