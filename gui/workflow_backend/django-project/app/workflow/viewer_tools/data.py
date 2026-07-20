# Vendored verbatim from ../viewer_chatbot/functions/data.py (numpy-only draft).
# Re-sync via diff; keep changes upstream. See viewer_tools/__init__.py for provenance.
"""Data loading + shared helpers for the viewer-chatbot functions.

Everything operates on the SAME `connectivity_data.json` / `human_data.json`
that the Three.js viewer renders, so the chatbot's numbers always match the
on-screen curves. Region semantics come from `region_descriptions*.json`.

Signal types
------------
- ``temporal_average`` : neural-mass state variable (fast; our first target)
- ``bold``             : haemodynamic signal (slow)
- ``spiking``          : spike trains (future; the spiking viewer does not exist yet)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

SIGNAL_TEMPORAL_AVERAGE = "temporal_average"
SIGNAL_BOLD = "bold"
SIGNAL_SPIKING = "spiking"
CONTINUOUS = (SIGNAL_TEMPORAL_AVERAGE, SIGNAL_BOLD)

# When the signal type is not recorded in the data, distinguish slow BOLD from
# fast TemporalAverage by the sampling interval. BOLD monitors usually sample
# every 500-2000 ms; TemporalAverage every ~1-10 ms.
_BOLD_DT_THRESHOLD_MS = 100.0


@dataclass
class ViewerData:
    """Holds one loaded viewer dataset + its region descriptions.

    Instantiate once per chat session (per run being viewed). All group 1-5
    functions take a ViewerData instance.
    """
    connectivity: dict                                # connectivity_data.json / human_data.json
    region_desc: dict = field(default_factory=dict)   # region_descriptions*.json
    signal_type: Optional[str] = None                 # override; else auto-detected

    _times: Optional[np.ndarray] = field(default=None, repr=False)
    _activity: Optional[np.ndarray] = field(default=None, repr=False)   # shape (T, N): activity[t, region]
    _name_to_index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # The timeseries is stored under a monitor-named key ('temporal_average'
        # or 'bold'). The key name IS the exact signal type — no inference needed.
        signal = None
        for key in (SIGNAL_TEMPORAL_AVERAGE, SIGNAL_BOLD):
            s = self.connectivity.get(key)
            if s:
                signal = s
                if self.signal_type is None:
                    self.signal_type = key
                break
        if signal:
            self._times = np.asarray(signal["time"], dtype=float)
            self._activity = np.asarray(signal["data"], dtype=float)     # (T, N)
        if self.signal_type is None:
            # Fallback for legacy data with no monitor-named key.
            self.signal_type = detect_signal_type(self._times, self.meta)
        self._name_to_index = {r["name"]: i for i, r in enumerate(self.regions)}

    # -- convenience accessors ------------------------------------------------
    @property
    def regions(self) -> list:
        return self.connectivity.get("regions", [])

    @property
    def meta(self) -> dict:
        return self.connectivity.get("meta", {})

    @property
    def species(self) -> Optional[str]:
        return self.meta.get("species")

    @property
    def has_signal(self) -> bool:
        return self._activity is not None

    @property
    def times(self) -> Optional[np.ndarray]:
        return self._times

    @property
    def activity(self) -> Optional[np.ndarray]:
        return self._activity     # (T, N) or None


def load_viewer_data(data_path: str,
                     region_desc_path: Optional[str] = None,
                     signal_type: Optional[str] = None) -> ViewerData:
    """Load a run's data JSON (+ optional region descriptions) into a ViewerData."""
    with open(data_path, encoding="utf-8") as f:
        conn = json.load(f)
    rdesc: dict = {}
    if region_desc_path:
        with open(region_desc_path, encoding="utf-8") as f:
            rdesc = json.load(f)
    return ViewerData(connectivity=conn, region_desc=rdesc, signal_type=signal_type)


def detect_signal_type(times: Optional[np.ndarray], meta: dict) -> str:
    """Best-effort signal type.

    Prefers an explicit ``meta['monitor_type']`` if present (TODO: have the
    TVBBrainViewerNode write it). Otherwise infers from the sampling interval.
    """
    mt = (meta or {}).get("monitor_type")
    if mt:
        mt = str(mt).lower()
        if "bold" in mt:
            return SIGNAL_BOLD
        if "temporal" in mt or "average" in mt:
            return SIGNAL_TEMPORAL_AVERAGE
    if times is None or len(times) < 2:
        return SIGNAL_TEMPORAL_AVERAGE
    dt = float(np.median(np.diff(times)))
    return SIGNAL_BOLD if dt >= _BOLD_DT_THRESHOLD_MS else SIGNAL_TEMPORAL_AVERAGE


def units_for(signal_type: Optional[str]) -> str:
    return {
        SIGNAL_TEMPORAL_AVERAGE: "model units (state variable, dimensionless)",
        SIGNAL_BOLD: "BOLD signal (arbitrary units)",
        SIGNAL_SPIKING: "spikes",
    }.get(signal_type, "signal units")


def resolve_region_index(vd: ViewerData, region) -> int:
    """Resolve a region given an int index or an EXACT label ('L_A10').

    For fuzzy / natural-language lookup, call ``search_regions`` first.
    """
    if isinstance(region, (int, np.integer)):
        idx = int(region)
    elif isinstance(region, str) and region in vd._name_to_index:
        idx = vd._name_to_index[region]
    else:
        raise ValueError(
            f"Cannot resolve region {region!r}. Pass an index, an exact label "
            f"like 'L_A10', or use search_regions() to find it first."
        )
    n = len(vd.regions)
    if not (0 <= idx < n):
        raise IndexError(f"Region index {idx} out of range (0..{n - 1}).")
    return idx


def window_mask(times: np.ndarray, t_start=None, t_end=None) -> np.ndarray:
    """Boolean mask for timepoints within [t_start, t_end] (ms, inclusive)."""
    mask = np.ones(len(times), dtype=bool)
    if t_start is not None:
        mask &= times >= t_start
    if t_end is not None:
        mask &= times <= t_end
    return mask


def region_window(vd: ViewerData, region, t_start=None, t_end=None):
    """Return (index, times[], values[]) for a region over an optional window."""
    if not vd.has_signal:
        raise ValueError("This dataset has no embedded timeseries "
                         "(the workflow ran without a BOLD/TemporalAverage monitor).")
    idx = resolve_region_index(vd, region)
    mask = window_mask(vd.times, t_start, t_end)
    return idx, vd.times[mask], vd.activity[mask, idx]
