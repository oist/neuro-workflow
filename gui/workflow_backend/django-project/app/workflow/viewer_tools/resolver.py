"""Resolve a FlowProject to a loaded :class:`ViewerData`.

The chat tools operate on the SAME run data the browser viewer renders — the
``connectivity_data.json`` / ``human_data.json`` the ``TVBBrainViewerNode`` wrote
into the project's ``results/viewer/`` dir. This module finds that file on disk
(reusing the project-dir resolution the rest of the app uses), pairs it with the
species' region-description lookup shipped in ``region_data/``, and caches the
parsed result keyed by mtime so repeated tool calls in one chat turn don't
re-parse a multi-MB JSON.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..path_utils import existing_project_dir
from .data import ViewerData

# region_descriptions*.json live next to this package (chat-only; the viewer JS
# never reads them). Selected by the connectivity JSON's meta.species.
_REGION_DATA_DIR = Path(__file__).resolve().parent / "region_data"
_REGION_DESC = {
    "human": _REGION_DATA_DIR / "region_descriptions_human.json",
    "marmoset": _REGION_DATA_DIR / "region_descriptions.json",
}

# Filenames the node writes, in the order we prefer when discovering (matches the
# frontend: human -> human_data.json, else connectivity_data.json).
_VIEWER_DATA_NAMES = ("human_data.json", "connectivity_data.json")
_VIEWER_SUBDIR = os.path.join("results", "viewer")

# (project_id, resolved_path, mtime) -> ViewerData
_CACHE: dict[tuple, ViewerData] = {}


class ViewerDataNotFound(Exception):
    """No viewer data file could be resolved for the project."""


def _resolve_explicit(project_dir: Path, data_path: str) -> Path:
    """Resolve a caller-supplied relative data_path, rejecting traversal."""
    # Drop any query string / fragment — the viewer's data_url carries a
    # ?_ts=... cache-buster that is not part of the on-disk path.
    data_path = data_path.split("?", 1)[0].split("#", 1)[0]
    base = project_dir.resolve()
    target = (base / data_path).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"Invalid data_path (escapes project dir): {data_path!r}")
    return target


def _discover(project_dir: Path) -> Optional[Path]:
    """Find the newest viewer data file under the project dir."""
    # Fast path: the conventional results/viewer/ location.
    viewer_dir = project_dir / _VIEWER_SUBDIR
    candidates: list[Path] = []
    for name in _VIEWER_DATA_NAMES:
        p = viewer_dir / name
        if p.is_file():
            candidates.append(p)
    # Fallback: a broader search (workflows may configure a different output_dir).
    if not candidates:
        for name in _VIEWER_DATA_NAMES:
            candidates.extend(project_dir.rglob(name))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_region_desc(conn_meta_species: Optional[str]) -> dict:
    species = (conn_meta_species or "").lower()
    path = (
        _REGION_DESC.get("human")
        if species == "human"
        else _REGION_DESC.get("marmoset")
    )
    if not path or not path.is_file():
        return {}
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_project_viewer_data(project, data_path: Optional[str] = None) -> ViewerData:
    """Load the active run's ViewerData for ``project``.

    ``data_path`` (relative to the project dir) pins a specific run — pass the
    viewer's current data file so the chat reads exactly what is on screen. When
    omitted, the newest viewer data file under the project is discovered.
    """
    project_dir = existing_project_dir(project)

    if data_path:
        target = _resolve_explicit(project_dir, data_path)
        if not target.is_file():
            raise ViewerDataNotFound(f"Viewer data file not found: {data_path!r}")
    else:
        found = _discover(project_dir)
        if found is None:
            raise ViewerDataNotFound(
                "No viewer data (connectivity_data.json / human_data.json) found "
                "for this project — run the brain viewer node first."
            )
        target = found

    key = (str(project.id), str(target), target.stat().st_mtime)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # Parse the connectivity JSON once, then pair it with the species' region
    # descriptions (chosen from meta.species) — avoids a second full read.
    import json

    with open(target, encoding="utf-8") as f:
        conn = json.load(f)
    species = (conn.get("meta") or {}).get("species")
    rdesc = _load_region_desc(species)

    vd = ViewerData(connectivity=conn, region_desc=rdesc)
    _CACHE[key] = vd
    return vd
