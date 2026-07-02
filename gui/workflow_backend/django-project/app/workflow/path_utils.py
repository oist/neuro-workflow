import re
from pathlib import Path

from django.conf import settings


WORKFLOW_CODE_FILENAME = "workflow.py"
WORKFLOW_NOTEBOOK_FILENAME = "workflow.ipynb"
ALLOWED_REPORT_SUFFIXES = {".md", ".markdown", ".txt"}


def projects_root() -> Path:
    root = Path(settings.BASE_DIR) / "codes" / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _ensure_under_root(path: Path) -> Path:
    root = projects_root().resolve()
    resolved = path.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        raise ValueError("Resolved path escapes the workflow projects directory")
    return path


def stable_project_dir(project, *, create: bool = False) -> Path:
    path = _ensure_under_root(projects_root() / str(project.id))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def batch_run_dir(project_id, run_id, *, create: bool = False) -> Path:
    """Per-run working dir for cluster (batch) executions, co-located with the
    project so Jupyter and cluster runs share ``codes/projects/<project_id>/``.

    Layout: ``codes/projects/<project_id>/batch/<run_id>/`` holding the staged
    inputs (workflow.py, run.sbatch, nodes/) and a ``results/`` subdir for the
    artifacts fetched back from the compute server.
    """
    path = _ensure_under_root(
        projects_root() / str(project_id) / "batch" / str(run_id)
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_project_dir(project) -> Path:
    legacy_name = (project.name or str(project.id)).replace(" ", "").capitalize()
    legacy_name = re.sub(r"[^A-Za-z0-9_.-]", "_", legacy_name) or str(project.id)
    return _ensure_under_root(projects_root() / legacy_name)


def existing_project_dir(project, *, create: bool = False) -> Path:
    stable = stable_project_dir(project, create=False)
    if stable.exists():
        return stable

    legacy = legacy_project_dir(project)
    if legacy.exists():
        return legacy

    if create:
        return stable_project_dir(project, create=True)
    return stable


def code_file_path(project, *, create: bool = False) -> Path:
    project_dir = stable_project_dir(project, create=create) if create else existing_project_dir(project)
    if project_dir.name == str(project.id):
        return _ensure_under_root(project_dir / WORKFLOW_CODE_FILENAME)
    return _ensure_under_root(project_dir / f"{project_dir.name}.py")


def notebook_file_path(project, *, create: bool = False) -> Path:
    project_dir = stable_project_dir(project, create=create) if create else existing_project_dir(project)
    if project_dir.name == str(project.id):
        return _ensure_under_root(project_dir / WORKFLOW_NOTEBOOK_FILENAME)
    return _ensure_under_root(project_dir / f"{project_dir.name}.ipynb")


def safe_report_path(project, filename: str, *, create_dir: bool = False) -> Path:
    name = (filename or "report.md").strip()
    candidate = Path(name)
    if (
        not name
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or name in {".", ".."}
        or ".." in candidate.parts
        or candidate.suffix.lower() not in ALLOWED_REPORT_SUFFIXES
    ):
        raise ValueError("Invalid report filename")

    project_dir = stable_project_dir(project, create=create_dir) if create_dir else existing_project_dir(project)
    return _ensure_under_root(project_dir / candidate.name)
