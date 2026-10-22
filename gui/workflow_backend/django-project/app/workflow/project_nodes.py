"""Project copies of node code: ``<project dir>/project_nodes/<ClassName>.py``.

Editing a node's code from the canvas writes a copy into the project's own
directory, so the library node (``codes/nodes/<category>/<ClassName>.py``) and
every other project keep the original. Within the project, every node of that
class then uses the copy: generated workflows import it as
``project_nodes.<ClassName>``. The project directory is the one the Jupyter
kernel runs in, and cluster runs copy it whole, so the copy is found there too.
"""

import re
from pathlib import Path

from app.box.models import PythonFile
from app.box.services.python_analyzer import PythonNodeAnalyzer

from .path_utils import existing_project_dir

PROJECT_NODES_DIRNAME = "project_nodes"
PROJECT_NODE_PATH_KEY = "projectNodePath"

_CLASS_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def project_node_file(project, class_name, *, create=False) -> Path:
    """Path of the project copy of ``class_name`` (it may not exist)."""
    if not _CLASS_NAME_RE.match(class_name or ""):
        raise ValueError(f"Invalid node class name: {class_name!r}")
    directory = existing_project_dir(project, create=create) / PROJECT_NODES_DIRNAME
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{class_name}.py"


def project_node_jupyter_path(project, class_name) -> str:
    """The copy's path as JupyterLab's file browser shows it."""
    project_dir = existing_project_dir(project)
    return f"codes/projects/{project_dir.name}/{PROJECT_NODES_DIRNAME}/{class_name}.py"


def project_node_classes(project) -> set:
    """Class names that have a copy in this project."""
    directory = existing_project_dir(project) / PROJECT_NODES_DIRNAME
    if not directory.is_dir():
        return set()
    return {
        path.stem for path in directory.glob("*.py") if _CLASS_NAME_RE.match(path.stem)
    }


def node_schema_from_source(code, class_name):
    """Frontend schema of ``class_name`` parsed from ``code``, or None."""
    for info in PythonNodeAnalyzer().analyze_file_content(code):
        if info["class_name"] == class_name:
            return PythonFile()._convert_to_full_schema(info)
    return None


def apply_project_node(project, data):
    """Give a node dropped on the canvas its project copy's schema, if any."""
    class_name = (data or {}).get("label") or ""
    try:
        path = project_node_file(project, class_name)
    except ValueError:
        return
    if not path.is_file():
        return
    schema = node_schema_from_source(path.read_text(encoding="utf-8"), class_name)
    if schema is not None:
        data["schema"] = schema
        data[PROJECT_NODE_PATH_KEY] = project_node_jupyter_path(project, class_name)
