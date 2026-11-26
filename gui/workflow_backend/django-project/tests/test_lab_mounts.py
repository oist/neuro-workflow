"""Lab bind mounts from jupyterhub_config.py (no JupyterHub or Docker needed)."""

import runpy
import sys
import types
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "neuroworkflow" / "jupyterhub_config.py"
PROJECT = "/srv/nw/django-project"
COMMUNITY = "/srv/nw/community"


class _Section:
    def __getattr__(self, name):
        value = _Section()
        object.__setattr__(self, name, value)
        return value


def _volumes(monkeypatch):
    monkeypatch.setenv("HOST_PROJECT_PATH", PROJECT)
    monkeypatch.setenv("HOST_COMMUNITY_PATH", COMMUNITY)
    monkeypatch.delenv("HOST_HACKATHON_PATH", raising=False)
    monkeypatch.delenv("JUPYTERHUB_AUTHENTICATOR", raising=False)
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "test-only-token")
    dockerspawner = types.ModuleType("dockerspawner")
    dockerspawner.DockerSpawner = type("DockerSpawner", (), {})
    handlers = types.ModuleType("custom_handlers")
    handlers.CORSHandler = handlers.AuthStatusHandler = object
    monkeypatch.setitem(sys.modules, "dockerspawner", dockerspawner)
    monkeypatch.setitem(sys.modules, "custom_handlers", handlers)
    namespace = runpy.run_path(str(CONFIG), init_globals={"get_config": _Section})
    return namespace["_volumes_for_username"]


def test_project_lab_reads_the_community_folder_read_only(monkeypatch):
    mounts = _volumes(monkeypatch)("internal")
    assert mounts[f"{COMMUNITY}/projects"] == {
        "bind": "/home/jovyan/codes-community/projects",
        "mode": "ro",
    }
    assert mounts[f"{COMMUNITY}/nodes"] == {
        "bind": "/home/jovyan/codes-community/nodes",
        "mode": "ro",
    }
    assert mounts[f"{PROJECT}/codes/projects"] == {
        "bind": "/home/jovyan/codes/projects",
        "mode": "rw",
    }


def test_community_lab_mounts_are_unchanged(monkeypatch):
    mounts = _volumes(monkeypatch)("external")
    assert mounts[f"{COMMUNITY}/projects"] == {
        "bind": "/home/jovyan/codes/projects",
        "mode": "rw",
    }
    assert not any(
        spec["bind"].startswith("/home/jovyan/codes-community") for spec in mounts.values()
    )
    assert f"{PROJECT}/codes/projects" not in mounts
