# Vendored verbatim from ../viewer_chatbot/functions/timeseries.py (numpy-only draft).
# Re-sync via diff; keep changes upstream. See viewer_tools/__init__.py for provenance.
"""Group 3 — simulation timeseries retrieval.

The signal lives under a MONITOR-NAMED key in the data JSON — ``temporal_average``
or ``bold`` — as ``{time[], data[t][region]}`` (oriented time x region). ViewerData
normalises whichever key is present into ``vd.times`` / ``vd.activity`` and sets
``vd.signal_type`` from the key name. Present only when the run used a monitor.
"""
from __future__ import annotations

import numpy as np

from .data import ViewerData, region_window, units_for


def list_signals(vd: ViewerData) -> dict:
    """What simulation signal (if any) is embedded in this dataset."""
    if not vd.has_signal:
        return {
            "has_signal": False,
            "signal_type": None,
            "note": "No timeseries embedded (workflow ran without a BOLD/TemporalAverage monitor).",
        }
    t = vd.times
    return {
        "has_signal": True,
        "signal_type": vd.signal_type,
        "n_timepoints": int(len(t)),
        "time_range_ms": [float(t[0]), float(t[-1])],
        "dt_ms": float(np.median(np.diff(t))) if len(t) > 1 else None,
        "n_regions": int(vd.activity.shape[1]),
        "units": units_for(vd.signal_type),
    }


def get_activity(vd: ViewerData, region, t_start=None, t_end=None, max_points: int | None = 400) -> dict:
    """The activity curve for one region over an optional [t_start, t_end] window (ms).

    ``max_points`` uniformly downsamples so the series fits an LLM context; set
    to None for the full-resolution curve.
    """
    idx, t, v = region_window(vd, region, t_start, t_end)
    if max_points and len(t) > max_points:
        stride = int(np.ceil(len(t) / max_points))
        t, v = t[::stride], v[::stride]
    return {
        "region": vd.regions[idx]["name"],
        "index": idx,
        "signal_type": vd.signal_type,
        "units": units_for(vd.signal_type),
        "time_ms": [round(float(x), 2) for x in t],
        "values": [round(float(x), 6) for x in v],
        "n_points": int(len(t)),
    }
