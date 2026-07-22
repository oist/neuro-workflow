# Vendored verbatim from ../viewer_chatbot/functions/viewer_actions.py (numpy-only draft).
# Re-sync via diff; keep changes upstream. See viewer_tools/__init__.py for provenance.
"""Group 5 — viewer control actions (chat -> Three.js viewer bridge).

These do not touch data. Each returns a small structured ``action`` dict that the
chatbot emits and the FRONTEND applies to the running viewer (``initBrainViewer``).
The frontend engineer implements the handlers; the ``action`` names below map to
capabilities that already exist in brain_viewer.js:

    select_region     -> selectRegion(index)         (focus a sphere, show its panel)
    focus_region      -> selectRegion(index) + "Dim others" on + fit camera
    set_time_window   -> BOLD scrubber / play window  (t_start..t_end ms)
    show_trace        -> open the BOLD trace for the region
    clear_selection   -> selectRegion(-1)
"""
from __future__ import annotations

from .data import ViewerData, resolve_region_index


def highlight_region(vd: ViewerData, region) -> dict:
    idx = resolve_region_index(vd, region)
    return {"action": "select_region", "index": idx, "label": vd.regions[idx]["name"]}


def focus_region(vd: ViewerData, region) -> dict:
    idx = resolve_region_index(vd, region)
    return {"action": "focus_region", "index": idx, "label": vd.regions[idx]["name"],
            "note": "select + dim non-selected + fit camera to the region"}


def set_time_window(vd: ViewerData, t_start=None, t_end=None) -> dict:
    return {"action": "set_time_window", "t_start_ms": t_start, "t_end_ms": t_end}


def show_curve(vd: ViewerData, region) -> dict:
    idx = resolve_region_index(vd, region)
    return {"action": "show_trace", "index": idx, "label": vd.regions[idx]["name"]}


def clear_selection(vd: ViewerData) -> dict:
    return {"action": "clear_selection"}
