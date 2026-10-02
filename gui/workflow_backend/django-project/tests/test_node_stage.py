"""Node staging copies only the modules a batch workflow imports."""

from pathlib import Path

import pytest
from app.workflow.execution.node_stage import NodeStageError, stage_required_nodes


def _write(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "nodes"
    _write(root, "__init__.py", '"""nodes"""\n')
    _write(root, "analysis/__init__.py", '"""analysis"""\n')
    _write(root, "analysis/Needed.py", "class Needed:\n    pass\n")
    _write(root, "analysis/Other.py", "class Other:\n    pass\n")
    return root


def _py_files(dest: Path) -> set[str]:
    if not dest.exists():
        return set()
    return {
        path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file()
    }


def test_imported_class_is_copied_and_unrelated_class_is_not(tmp_path):
    src = _tree(tmp_path)
    dest = tmp_path / "out"
    stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    copied = _py_files(dest)
    assert "analysis/Needed.py" in copied
    assert "analysis/Other.py" not in copied


def test_package_inits_are_copied(tmp_path):
    src = _tree(tmp_path)
    dest = tmp_path / "out"
    stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    copied = _py_files(dest)
    assert "__init__.py" in copied
    assert "analysis/__init__.py" in copied


def test_same_directory_helper_is_copied(tmp_path):
    src = _tree(tmp_path)
    _write(src, "analysis/helper.py", "VALUE = 1\n")
    _write(src, "analysis/Needed.py", "import helper\nclass Needed:\n    pass\n")
    dest = tmp_path / "out"
    stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    copied = _py_files(dest)
    assert "analysis/helper.py" in copied
    assert "analysis/Other.py" not in copied


def test_named_sibling_directory_is_copied(tmp_path):
    src = _tree(tmp_path)
    _write(src, "analysis/viewer_static/mesh.json", "{}\n")
    _write(
        src,
        "analysis/Needed.py",
        'STATIC = "viewer_static"\nclass Needed:\n    pass\n',
    )
    dest = tmp_path / "out"
    stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    assert (dest / "analysis" / "viewer_static" / "mesh.json").is_file()
    assert "analysis/Other.py" not in _py_files(dest)


def test_missing_module_raises_and_leaves_dest_empty(tmp_path):
    src = _tree(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(NodeStageError, match="missing"):
        stage_required_nodes(src, dest, "from nodes.analysis.Missing import Missing\n")
    assert list(dest.iterdir()) == []


def test_parent_relative_import_that_escapes_raises(tmp_path):
    src = _tree(tmp_path)
    _write(src, "analysis/Needed.py", "from ...outside import leak\n")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(NodeStageError, match="escapes"):
        stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    assert list(dest.iterdir()) == []


def test_workflow_without_node_imports_copies_nothing(tmp_path):
    src = _tree(tmp_path)
    dest = tmp_path / "out"
    stage_required_nodes(src, dest, "import numpy as np\nprint(np.pi)\n")
    assert not dest.exists()


def test_literal_import_module_includes_that_module(tmp_path):
    src = _tree(tmp_path)
    _write(src, "analysis/Extra.py", "class Extra:\n    pass\n")
    dest = tmp_path / "out"
    code = 'import importlib\nimportlib.import_module("nodes.analysis.Extra")\n'
    stage_required_nodes(src, dest, code)
    copied = _py_files(dest)
    assert "analysis/Extra.py" in copied
    assert "analysis/Needed.py" not in copied
    assert "analysis/Other.py" not in copied


def test_incidental_path_strings_do_not_reject_the_node(tmp_path):
    src = _tree(tmp_path)
    _write(
        src,
        "analysis/Needed.py",
        "\n".join(
            [
                "MARK = '/'",
                "UP = '..'",
                "TMP = '/tmp/matplotlib'",
                "OUT = '../outputs/'",
                "STATIC = 'viewer_static'",
                "class Needed:",
                "    pass",
                "",
            ]
        ),
    )
    _write(src, "analysis/viewer_static/mesh.json", "{}\n")
    dest = tmp_path / "out"
    stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    assert (dest / "analysis" / "viewer_static" / "mesh.json").is_file()
    assert not (dest / "analysis" / "Other.py").exists()
    assert ".." not in _py_files(dest)
    assert not any(path.name == "matplotlib" for path in dest.rglob("*"))


def test_parents_dotdot_raises_and_leaves_dest_empty(tmp_path):
    src = _tree(tmp_path)
    _write(
        src,
        "analysis/Needed.py",
        "from pathlib import Path\n"
        "ESCAPE = Path(__file__).resolve().parents[0] / '..'\n",
    )
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(NodeStageError, match="outside"):
        stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    assert list(dest.iterdir()) == []


def test_nonliteral_import_module_raises_and_leaves_dest_empty(tmp_path):
    src = _tree(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    code = (
        "import importlib\n"
        "name = 'nodes.analysis.Needed'\n"
        "importlib.import_module(name)\n"
    )
    with pytest.raises(NodeStageError, match="not a string literal"):
        stage_required_nodes(src, dest, code)
    assert list(dest.iterdir()) == []


def test_existing_dest_file_outside_the_closure_is_kept(tmp_path):
    src = _tree(tmp_path)
    dest = tmp_path / "out"
    _write(dest, "local_helper.py", "KEEP = 1\n")
    stage_required_nodes(src, dest, "from nodes.analysis.Needed import Needed\n")
    assert (dest / "local_helper.py").read_text() == "KEEP = 1\n"
    assert (dest / "analysis" / "Needed.py").is_file()
    assert not (dest / "analysis" / "Other.py").exists()
