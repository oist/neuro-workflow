# Vendored verbatim from ../viewer_chatbot/functions/structure.py (numpy-only draft).
# Re-sync via diff; keep changes upstream. See viewer_tools/__init__.py for provenance.
"""Group 2 — structural connectivity (no simulation signal needed).

Reads ``connectivity_data.json['connections']`` = ``[i, j, weight, tract_length_mm]``
(undirected; each pair listed once). Works even when the run had no monitor.
"""
from __future__ import annotations

from .data import ViewerData, resolve_region_index


def get_connections(vd: ViewerData, region, top_n: int = 10) -> list[dict]:
    """Strongest structural connections of a region, by weight."""
    idx = resolve_region_index(vd, region)
    hits = []
    for c in vd.connectivity.get("connections", []):
        i, j, w, length = c[0], c[1], c[2], c[3]
        if i == idx or j == idx:
            other = j if i == idx else i
            hits.append({
                "target_index": other,
                "target_label": vd.regions[other]["name"],
                "weight": w,
                "tract_length_mm": length,
            })
    hits.sort(key=lambda h: -h["weight"])
    return hits[:top_n]


def node_strength(vd: ViewerData, region) -> dict:
    """Total incident connection weight (structural hubness) + degree.

    definition: sum of weights of connections touching the region; degree is the
    count of such connections. Higher -> a more central structural hub.
    """
    idx = resolve_region_index(vd, region)
    conns = vd.connectivity.get("connections", [])
    strength = sum(c[2] for c in conns if c[0] == idx or c[1] == idx)
    degree = sum(1 for c in conns if c[0] == idx or c[1] == idx)
    return {
        "region": vd.regions[idx]["name"],
        "node_strength": strength,
        "degree": degree,
        "definition": "Sum of incident connection weights; degree is the number of connections.",
        "interpretation": "Higher strength/degree means the region is a more central structural hub.",
        "reference": "Graph metrics of structural connectivity (Rubinov & Sporns 2010).",
    }
