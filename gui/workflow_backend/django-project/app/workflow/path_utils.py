import re
from pathlib import Path

from django.conf import settings


WORKFLOW_CODE_FILENAME = "workflow.py"
WORKFLOW_NOTEBOOK_FILENAME = "workflow.ipynb"
ALLOWED_REPORT_SUFFIXES = {".md", ".markdown", ".txt"}

# Project data uploads (GUI / API → codes/projects/<id>/)
PROJECT_UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # keep in sync with nginx client_max_body_size
PROTECTED_PROJECT_FILENAMES = frozenset(
    {
        WORKFLOW_CODE_FILENAME,
        WORKFLOW_NOTEBOOK_FILENAME,
        "run.sbatch",
    }
)
ALLOWED_UPLOAD_COMPOUND_SUFFIXES = frozenset(
    {
        ".nii.gz",
        ".tar.gz",
        ".npz.gz",
        ".csv.gz",
        ".json.gz",
        ".tsv.gz",
        ".txt.gz",
    }
)
ALLOWED_UPLOAD_SUFFIXES = frozenset(
    {
        ".csv",
        ".tsv",
        ".txt",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".md",
        ".markdown",
        ".zip",
        ".tar",
        ".tgz",
        ".gz",
        ".bz2",
        ".xz",
        ".h5",
        ".hdf5",
        ".mat",
        ".npz",
        ".npy",
        ".pkl",
        ".pickle",
        ".parquet",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".pdf",
        ".xlsx",
        ".xls",
        ".ods",
        ".nii",
        ".nwb",
        ".edf",
        ".fif",
        ".py",
    }
)


def upload_file_suffix(filename: str) -> str:
    """Return the effective suffix used for allowlist checks (supports .nii.gz etc.)."""
    lower = (filename or "").lower()
    for compound in sorted(ALLOWED_UPLOAD_COMPOUND_SUFFIXES, key=len, reverse=True):
        if lower.endswith(compound):
            return compound
    return Path(filename or "").suffix.lower()


def is_allowed_upload_filename(filename: str) -> bool:
    suffix = upload_file_suffix(filename)
    return suffix in ALLOWED_UPLOAD_SUFFIXES or suffix in ALLOWED_UPLOAD_COMPOUND_SUFFIXES


def projects_root(tenant: str | None = None) -> Path:
    from app.tenants import TENANT_HACKATHON, normalize_tenant

    tenant = normalize_tenant(tenant)
    if tenant == TENANT_HACKATHON:
        root = Path(settings.BASE_DIR) / "codes-hackathon" / "projects"
    else:
        root = Path(settings.BASE_DIR) / "codes" / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def nodes_root(tenant: str | None = None) -> Path:
    from app.tenants import TENANT_HACKATHON, normalize_tenant

    tenant = normalize_tenant(tenant)
    if tenant == TENANT_HACKATHON:
        root = Path(settings.BASE_DIR) / "codes-hackathon" / "nodes"
    else:
        root = Path(getattr(settings, "MEDIA_ROOT", Path(settings.BASE_DIR) / "codes" / "nodes"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tenant_of(project=None, tenant=None) -> str | None:
    if tenant:
        return tenant
    if project is not None:
        return getattr(project, "tenant", None)
    return None


def _ensure_under_root(path: Path, *, project=None, tenant=None) -> Path:
    root = projects_root(_tenant_of(project, tenant)).resolve()
    resolved = path.resolve(strict=False)
    if root != resolved and root not in resolved.parents:
        raise ValueError("Resolved path escapes the workflow projects directory")
    return path


def stable_project_dir(project, *, create: bool = False) -> Path:
    path = _ensure_under_root(
        projects_root(_tenant_of(project)) / str(project.id), project=project
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def batch_run_dir(project_id, run_id, *, create: bool = False, project=None) -> Path:
    """Per-run working dir for cluster (batch) executions, co-located with the
    project so Jupyter and cluster runs share ``codes/projects/<project_id>/``.

    Layout: ``codes/projects/<project_id>/batch/<run_id>/`` holding the staged
    inputs (workflow.py, run.sbatch, nodes/) and a ``results/`` subdir for the
    artifacts fetched back from the compute server.
    """
    if project is None:
        from .models import FlowProject

        try:
            project = FlowProject.objects.get(id=project_id)
        except Exception:
            project = None
    tenant = _tenant_of(project)
    path = _ensure_under_root(
        projects_root(tenant) / str(project_id) / "batch" / str(run_id),
        project=project,
        tenant=tenant,
    )
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_project_dir(project) -> Path:
    legacy_name = (project.name or str(project.id)).replace(" ", "").capitalize()
    legacy_name = re.sub(r"[^A-Za-z0-9_.-]", "_", legacy_name) or str(project.id)
    return _ensure_under_root(
        projects_root(_tenant_of(project)) / legacy_name, project=project
    )


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
        return _ensure_under_root(project_dir / WORKFLOW_CODE_FILENAME, project=project)
    return _ensure_under_root(project_dir / f"{project_dir.name}.py", project=project)


def notebook_file_path(project, *, create: bool = False) -> Path:
    project_dir = stable_project_dir(project, create=create) if create else existing_project_dir(project)
    if project_dir.name == str(project.id):
        return _ensure_under_root(project_dir / WORKFLOW_NOTEBOOK_FILENAME, project=project)
    return _ensure_under_root(project_dir / f"{project_dir.name}.ipynb", project=project)


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
    return _ensure_under_root(project_dir / candidate.name, project=project)


def safe_project_upload_path(project, filename: str, *, create_dir: bool = True) -> Path:
    """Resolve a safe basename under the project's directory for data uploads.

    Rejects path traversal, absolute paths, disallowed extensions, and
    overwrites of generated workflow artifacts (``workflow.py`` / ``.ipynb``).
    """
    name = (filename or "").strip()
    candidate = Path(name)
    if (
        not name
        or "\x00" in name
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or name in {".", ".."}
        or ".." in candidate.parts
    ):
        raise ValueError("Invalid upload filename")

    if not is_allowed_upload_filename(candidate.name):
        raise ValueError(
            f"File type '{upload_file_suffix(candidate.name) or '(none)'}' is not allowed"
        )

    if candidate.name.lower() in {n.lower() for n in PROTECTED_PROJECT_FILENAMES}:
        raise ValueError(f"Cannot overwrite protected file '{candidate.name}'")

    project_dir = (
        stable_project_dir(project, create=True)
        if create_dir
        else existing_project_dir(project, create=False)
    )
    return _ensure_under_root(project_dir / candidate.name, project=project)
