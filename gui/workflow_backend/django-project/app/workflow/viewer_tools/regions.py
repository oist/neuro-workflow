# Vendored verbatim from ../viewer_chatbot/functions/regions.py (numpy-only draft).
# Re-sync via diff; keep changes upstream. See viewer_tools/__init__.py for provenance.
"""Group 1 — region semantics / lookup.

Uses ``region_descriptions*.json``:
  - ``regions`` : abbreviation (e.g. 'A10') -> {full_name, group, lobe, description, keywords}
  - ``groups``  : functional-group code (e.g. 'DLP') -> one-line summary

Labels in the connectivity data carry a hemisphere prefix ('L_A10' / 'R_A10').
"""
from __future__ import annotations

import re

from .data import ViewerData, resolve_region_index


def _abbrev(label: str) -> str:
    return label[2:] if label[:2] in ("L_", "R_") else label


def _hemi(label: str) -> str:
    return {"L_": "left", "R_": "right"}.get(label[:2], "unknown")


def search_regions(vd: ViewerData, query: str, top_n: int = 8) -> list[dict]:
    """Natural-language query -> ranked matching regions.

    Scores each region abbreviation in ``region_descriptions`` against the query
    (keyword hits weighted heavily, then term overlap on name/description/group),
    and expands to the concrete labels present in the connectivity data (both
    hemispheres). The LLM can then pick the intended one.
    """
    q = query.lower()
    q_terms = set(re.findall(r"[a-z0-9]+", q))
    rdesc = vd.region_desc.get("regions", {})

    scored = []
    for abbr, info in rdesc.items():
        keyword_hits = sum(1 for kw in info.get("keywords", []) if kw.lower() in q)
        hay = " ".join([
            abbr, info.get("full_name", ""), info.get("description", ""),
            info.get("group", ""), info.get("lobe", ""),
            " ".join(info.get("keywords", [])),
        ]).lower()
        overlap = len(q_terms & set(re.findall(r"[a-z0-9]+", hay)))
        score = keyword_hits * 3 + overlap + (5 if abbr.lower() in q else 0)
        if score > 0:
            scored.append((score, abbr, info))

    scored.sort(key=lambda x: -x[0])

    out: list[dict] = []
    for score, abbr, info in scored:
        for prefix, hemi in (("L_", "left"), ("R_", "right")):
            idx = vd._name_to_index.get(prefix + abbr)
            if idx is None:
                continue
            out.append({
                "index": idx,
                "label": prefix + abbr,
                "abbrev": abbr,
                "full_name": info.get("full_name"),
                "group": info.get("group"),
                "lobe": info.get("lobe"),
                "hemisphere": hemi,
                "description": info.get("description"),
                "score": score,
            })
        if len(out) >= top_n:
            break
    return out[:top_n]


def get_region(vd: ViewerData, region) -> dict:
    """Full semantic + geometric info for one region (index or exact label)."""
    idx = resolve_region_index(vd, region)
    r = vd.regions[idx]
    abbr = _abbrev(r["name"])
    info = vd.region_desc.get("regions", {}).get(abbr, {})
    return {
        "index": idx,
        "label": r["name"],
        "abbrev": abbr,
        "hemisphere": _hemi(r["name"]),
        "position": {"x": r.get("x"), "y": r.get("y"), "z": r.get("z")},
        "surface_area_mm2": r.get("area"),
        "full_name": info.get("full_name"),
        "group": info.get("group"),
        "lobe": info.get("lobe"),
        "description": info.get("description"),
        "keywords": info.get("keywords", []),
    }


def list_groups(vd: ViewerData) -> dict:
    """Functional-group code -> summary (e.g. 'DLP' -> 'Dorsolateral prefrontal ...')."""
    return dict(vd.region_desc.get("groups", {}))
