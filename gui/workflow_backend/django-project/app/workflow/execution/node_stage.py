"""Stage only the node modules a batch workflow imports.

The generated script runs with its directory on ``sys.path`` and imports
``nodes.<category>.<Class>``. Copying the whole tenant node tree would also
send every other user's node source to the shared cluster account. This module
parses the script (and each selected file) with ``ast`` and copies the closure.

Node modules are not imported. Importing them would run their top-level code
and pull in NEST, TVB, and NumPy on the app server.
"""

from __future__ import annotations

import ast
import shutil
from collections import deque
from pathlib import Path

_PYCACHE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


class NodeStageError(Exception):
    """The workflow's node imports cannot be staged safely."""


def stage_required_nodes(nodes_src: Path, dest: Path, code: str) -> None:
    """Copy the node files ``code`` needs from ``nodes_src`` into ``dest``.

    Copies nothing when ``code`` does not import the ``nodes`` package.
    Raises ``NodeStageError`` before creating ``dest`` when a required file
    cannot be resolved, so a failed call leaves ``dest`` untouched.
    """
    nodes_src = Path(nodes_src)
    dest = Path(dest)
    files, directories = _collect(nodes_src, code)
    if not files and not directories:
        return
    _copy_into(nodes_src, dest, files, directories)


def _collect(nodes_src: Path, code: str) -> tuple[set[Path], set[Path]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise NodeStageError(
            f"workflow.py: could not parse workflow code: {exc.msg}"
        ) from exc

    files: set[Path] = set()
    directories: set[Path] = set()
    scan_queue: deque[Path] = deque()
    seen: set[Path] = set()

    def add_py(rel: Path) -> None:
        rel = Path(rel)
        src = nodes_src / rel
        if not src.is_file():
            raise NodeStageError(f"Required node file is missing: {rel.as_posix()}")
        rel = _relative_inside(nodes_src, src)
        if rel in seen:
            files.add(rel)
            return
        seen.add(rel)
        files.add(rel)
        scan_queue.append(rel)

    def add_module(dotted: str) -> None:
        rel = _resolve_module(nodes_src, dotted)
        add_py(rel)
        for init in _ancestor_inits(rel):
            add_py(init)

    _walk_imports(
        tree,
        origin="workflow.py",
        file_rel=None,
        nodes_src=nodes_src,
        add_module=add_module,
    )

    while scan_queue:
        rel = scan_queue.popleft()
        file_path = nodes_src / rel
        try:
            source = file_path.read_text(encoding="utf-8")
            file_tree = ast.parse(source)
        except OSError as exc:
            raise NodeStageError(
                f"{rel.as_posix()}: could not read node file: {exc}"
            ) from exc
        except SyntaxError as exc:
            raise NodeStageError(
                f"{rel.as_posix()}: could not parse node file: {exc.msg}"
            ) from exc
        _walk_imports(
            file_tree,
            origin=rel.as_posix(),
            file_rel=rel,
            nodes_src=nodes_src,
            add_module=add_module,
        )
        _collect_sidecars(
            file_tree,
            origin=rel.as_posix(),
            file_rel=rel,
            nodes_src=nodes_src,
            add_py=add_py,
            files=files,
            directories=directories,
        )
        _collect_parent_paths(
            file_tree,
            origin=rel.as_posix(),
            file_rel=rel,
            nodes_src=nodes_src,
            add_py=add_py,
            files=files,
            directories=directories,
        )

    return files, directories


def _walk_imports(
    tree: ast.AST,
    *,
    origin: str,
    file_rel: Path | None,
    nodes_src: Path,
    add_module,
) -> None:
    file_dir = None if file_rel is None else (nodes_src / file_rel).parent
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            _handle_import_from(
                node,
                origin=origin,
                file_dir=file_dir,
                file_rel=file_rel,
                nodes_src=nodes_src,
                add_module=add_module,
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                _handle_import_name(
                    alias.name,
                    origin=origin,
                    file_dir=file_dir,
                    nodes_src=nodes_src,
                    add_module=add_module,
                )
        elif isinstance(node, ast.Call):
            _handle_dynamic_import(node, origin=origin, add_module=add_module)


def _handle_import_from(
    node: ast.ImportFrom,
    *,
    origin: str,
    file_dir: Path | None,
    file_rel: Path | None,
    nodes_src: Path,
    add_module,
) -> None:
    star = any(alias.name == "*" for alias in node.names)
    if node.level:
        if file_rel is None:
            raise NodeStageError(
                f"{origin}: relative import is outside the nodes package"
            )
        base = _relative_base(file_rel, node.level, origin)
        if star:
            raise NodeStageError(f"{origin}: wildcard import of nodes is not supported")
        if node.module:
            _reject_traversal(node.module, origin)
            add_module(f"{base}.{node.module}")
        else:
            for alias in node.names:
                _reject_traversal(alias.name, origin)
                add_module(f"{base}.{alias.name}")
        return

    module = node.module or ""
    if _is_nodes_module(module):
        if star:
            raise NodeStageError(f"{origin}: wildcard import of nodes is not supported")
        _reject_traversal(module, origin)
        add_module(module)
        return

    if (
        file_dir is not None
        and module.isidentifier()
        and (file_dir / f"{module}.py").is_file()
    ):
        if star:
            raise NodeStageError(f"{origin}: wildcard import of nodes is not supported")
        rel = _relative_inside(nodes_src, file_dir / f"{module}.py")
        add_module(_rel_to_dotted(rel))


def _handle_import_name(
    name: str,
    *,
    origin: str,
    file_dir: Path | None,
    nodes_src: Path,
    add_module,
) -> None:
    if _is_nodes_module(name):
        _reject_traversal(name, origin)
        add_module(name)
        return
    head = name.split(".", 1)[0]
    if (
        file_dir is not None
        and head.isidentifier()
        and "." not in name
        and (file_dir / f"{name}.py").is_file()
    ):
        rel = _relative_inside(nodes_src, file_dir / f"{name}.py")
        add_module(_rel_to_dotted(rel))


def _handle_dynamic_import(node: ast.Call, *, origin: str, add_module) -> None:
    func = node.func
    is_dynamic = (
        isinstance(func, ast.Name) and func.id in {"import_module", "__import__"}
    ) or (isinstance(func, ast.Attribute) and func.attr == "import_module")
    if not is_dynamic:
        return
    if (
        not node.args
        or not isinstance(node.args[0], ast.Constant)
        or not isinstance(node.args[0].value, str)
    ):
        raise NodeStageError(
            f"{origin}: dynamic import argument is not a string literal"
        )
    literal = node.args[0].value
    if _is_nodes_module(literal):
        _reject_traversal(literal, origin)
        add_module(literal)


def _collect_sidecars(
    tree: ast.AST,
    *,
    origin: str,
    file_rel: Path,
    nodes_src: Path,
    add_py,
    files: set[Path],
    directories: set[Path],
) -> None:
    file_dir = (nodes_src / file_rel).parent
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value
        if not _path_token(value):
            continue
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise NodeStageError(
                f"{origin}: refusing a path outside the nodes tree: {value}"
            )
        if len(path.parts) != 1:
            continue
        child = file_dir / path.parts[0]
        if not child.exists():
            continue
        _add_existing(
            child,
            nodes_src=nodes_src,
            add_py=add_py,
            files=files,
            directories=directories,
        )


def _collect_parent_paths(
    tree: ast.AST,
    *,
    origin: str,
    file_rel: Path,
    nodes_src: Path,
    add_py,
    files: set[Path],
    directories: set[Path],
) -> None:
    file_path = (nodes_src / file_rel).resolve()
    root = nodes_src.resolve()
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue
        index = _parents_index(node.left)
        if index is None:
            continue
        if not isinstance(node.right, ast.Constant) or not isinstance(
            node.right.value, str
        ):
            continue
        segment = node.right.value
        seg = Path(segment)
        if not segment or seg.is_absolute() or ".." in seg.parts:
            raise NodeStageError(
                f"{origin}: refusing a path outside the nodes tree: {segment}"
            )
        try:
            base = file_path.parents[index]
        except IndexError as exc:
            raise NodeStageError(
                f"{origin}: Path(__file__).parents[{index}] escapes the nodes tree"
            ) from exc
        target = base / seg
        try:
            resolved = target.resolve()
            resolved.relative_to(root)
        except ValueError as exc:
            raise NodeStageError(
                f"{origin}: refusing a path outside the nodes tree: {segment}"
            ) from exc
        if resolved == root or not target.exists():
            continue
        _add_existing(
            target,
            nodes_src=nodes_src,
            add_py=add_py,
            files=files,
            directories=directories,
        )


def _add_existing(
    path: Path,
    *,
    nodes_src: Path,
    add_py,
    files: set[Path],
    directories: set[Path],
) -> None:
    rel = _relative_inside(nodes_src, path)
    if rel == Path():
        return
    if path.is_dir():
        directories.add(rel)
        return
    if path.suffix == ".py":
        add_py(rel)
        for init in _ancestor_inits(rel):
            add_py(init)
        return
    if path.suffix == ".pyc" or "__pycache__" in rel.parts:
        return
    files.add(rel)


def _parents_index(node: ast.AST) -> int | None:
    if not isinstance(node, ast.Subscript):
        return None
    value = node.value
    if not isinstance(value, ast.Attribute) or value.attr != "parents":
        return None
    if not any(
        isinstance(child, ast.Name) and child.id == "__file__"
        for child in ast.walk(value)
    ):
        return None
    index = node.slice
    if isinstance(index, ast.Constant) and isinstance(index.value, int):
        if index.value < 0:
            raise NodeStageError("Path(__file__).parents index escapes the nodes tree")
        return index.value
    return None


def _resolve_module(nodes_src: Path, dotted: str) -> Path:
    _reject_traversal(dotted, dotted)
    parts = dotted.split(".")
    if (
        not parts
        or parts[0] != "nodes"
        or any(not part.isidentifier() for part in parts)
    ):
        raise NodeStageError(f"Invalid node import: {dotted}")
    if len(parts) == 1:
        rel = Path("__init__.py")
    else:
        parent = Path(*parts[1:-1]) if len(parts) > 2 else Path()
        name = parts[-1]
        package = parent / name / "__init__.py"
        module = parent / f"{name}.py"
        if (nodes_src / package).is_file():
            rel = package
        elif (nodes_src / module).is_file():
            rel = module
        else:
            raise NodeStageError(f"Required node module is missing: {dotted}")
    if not (nodes_src / rel).is_file():
        raise NodeStageError(f"Required node module is missing: {dotted}")
    return rel


def _ancestor_inits(rel: Path) -> list[Path]:
    inits = [Path("__init__.py")]
    acc = Path()
    for part in rel.parent.parts:
        acc = acc / part
        inits.append(acc / "__init__.py")
    return inits


def _relative_base(file_rel: Path, level: int, origin: str) -> str:
    package = ["nodes", *file_rel.parent.parts]
    trim = level - 1
    if trim < 0 or trim >= len(package):
        raise NodeStageError(f"{origin}: import escapes the nodes package")
    base = package[: len(package) - trim]
    if not base or base[0] != "nodes":
        raise NodeStageError(f"{origin}: import escapes the nodes package")
    return ".".join(base)


def _rel_to_dotted(rel: Path) -> str:
    if rel.name == "__init__.py":
        parts = list(rel.parent.parts)
    else:
        parts = [*rel.parent.parts, rel.stem]
    if not parts:
        return "nodes"
    return "nodes." + ".".join(parts)


def _relative_inside(nodes_src: Path, path: Path) -> Path:
    root = nodes_src.resolve()
    try:
        rel = path.resolve().relative_to(root)
    except ValueError as exc:
        raise NodeStageError(f"Refusing a path outside the nodes tree: {path}") from exc
    if ".." in rel.parts:
        raise NodeStageError(f"Refusing a path outside the nodes tree: {rel}")
    return rel


def _is_nodes_module(name: str) -> bool:
    return name == "nodes" or name.startswith("nodes.")


def _reject_traversal(value: str, origin: str) -> None:
    parts = Path(value).parts
    if value.startswith("/") or ".." in parts or ".." in value.split("."):
        raise NodeStageError(
            f"{origin}: refusing a path outside the nodes tree: {value}"
        )


def _path_token(value: str) -> bool:
    if not value or len(value) > 240 or any(ch.isspace() for ch in value):
        return False
    return True


def _copy_into(
    nodes_src: Path,
    dest: Path,
    files: set[Path],
    directories: set[Path],
) -> None:
    staging = dest.parent / f".{dest.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for rel in sorted(directories):
            if rel.suffix == ".pyc" or "__pycache__" in rel.parts:
                continue
            target = staging / rel
            if target.exists():
                continue
            shutil.copytree(
                nodes_src / rel,
                target,
                ignore=_PYCACHE_IGNORE,
                dirs_exist_ok=True,
            )
        for rel in sorted(files):
            if rel.suffix == ".pyc" or "__pycache__" in rel.parts:
                continue
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(nodes_src / rel, target)
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
