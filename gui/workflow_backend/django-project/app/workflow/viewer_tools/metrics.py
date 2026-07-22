# Vendored verbatim from ../viewer_chatbot/functions/metrics.py (numpy-only draft).
# Re-sync via diff; keep changes upstream. See viewer_tools/__init__.py for provenance.
"""Group 4 — metrics + semantics (the core).

Every metric is TAGGED with the signal type(s) it applies to, so the caller
computes only what suits the loaded signal:

    temporal_average  neural-mass state variable (fast) — FIRST TARGET
    bold              haemodynamic signal (slow)
    spiking           spike trains (FUTURE; the spiking viewer does not exist yet)

Continuous metrics take (t, v) numpy arrays from ``region_window``.
Spiking metrics take spike trains — DRAFTED as stubs until the spiking viewer exists.

Each result carries: value, units, one-line ``definition`` (the semantics),
a short contextual ``interpretation``, and a ``reference``.

Reference shorthand: "Gerstner Ch.N" = Gerstner, Kistler, Naud & Paninski,
*Neuronal Dynamics* (2014), https://neuronaldynamics.epfl.ch/
"""
from __future__ import annotations

import numpy as np

from .data import (ViewerData, region_window, units_for, window_mask,
                   resolve_region_index, SIGNAL_TEMPORAL_AVERAGE, SIGNAL_BOLD,
                   SIGNAL_SPIKING, CONTINUOUS)
from .regions import get_region


# ---------------------------------------------------------------------------
# Individual continuous-signal metric computers.  fn(t, v) -> scalar | dict
# ---------------------------------------------------------------------------
# np.trapz was renamed np.trapezoid in numpy 2.0 — support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _mean(t, v):        return float(np.mean(v))
def _std(t, v):         return float(np.std(v))
def _variance(t, v):    return float(np.var(v))
def _cv(t, v):
    m = float(np.mean(v))
    return float(np.std(v) / m) if m != 0 else float("nan")
def _amplitude(t, v):   return float(np.max(v) - np.min(v))
def _rms(t, v):         return float(np.sqrt(np.mean(np.square(v))))
def _integral(t, v):    return float(_trapz(v, t))
def _baseline(t, v):
    k = max(1, len(v) // 10)
    return float(np.mean(v[-k:]))
def _trend_slope(t, v):
    return float(np.polyfit(t, v, 1)[0]) if len(t) >= 2 else float("nan")
def _peak(t, v):
    i = int(np.argmax(v));  return {"value": float(v[i]), "time_ms": float(t[i])}
def _trough(t, v):
    i = int(np.argmin(v));  return {"value": float(v[i]), "time_ms": float(t[i])}
def _autocorr_time(t, v):
    x = np.asarray(v, float) - np.mean(v)
    if np.allclose(x, 0):
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / ac[0]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    below = np.where(ac < 1.0 / np.e)[0]
    return float(below[0] * dt) if len(below) else float(len(x) * dt)
def _dominant_frequency(t, v):
    if len(t) < 4:
        return {"freq_hz": float("nan"), "power": float("nan")}
    dt_s = (float(np.median(np.diff(t))) / 1000.0)  # ms -> s
    # linear-detrend so a slow drift doesn't masquerade as the dominant rhythm
    x = np.asarray(v, float)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    freqs = np.fft.rfftfreq(len(x), d=dt_s)
    power = np.abs(np.fft.rfft(x)) ** 2
    if len(freqs) < 2:
        return {"freq_hz": float("nan"), "power": float("nan")}
    i = int(np.argmax(power[1:]) + 1)  # skip DC
    return {"freq_hz": float(freqs[i]), "power": float(power[i])}


# ---------------------------------------------------------------------------
# Metric catalog. `applies_to` drives which metrics run for a given signal.
# `units`: None -> use the signal's units; a string -> fixed units.
# ---------------------------------------------------------------------------
METRIC_CATALOG = [
    {"key": "mean_activity", "name": "Mean activity", "applies_to": CONTINUOUS, "fn": _mean,
     "units": None,
     "definition": "Time-averaged value of the signal over the window, <A> = (1/T)∫A(t) dt.",
     "reference": "Gerstner Ch.7 (population activity A(t))"},

    {"key": "std", "name": "Standard deviation", "applies_to": CONTINUOUS, "fn": _std,
     "units": None,
     "definition": "Typical size of fluctuations around the mean.",
     "reference": "Gerstner Ch.7"},

    {"key": "variance", "name": "Variance", "applies_to": CONTINUOUS, "fn": _variance,
     "units": None,
     "definition": "Squared standard deviation — spread of the signal.",
     "reference": "Gerstner Ch.7"},

    {"key": "coefficient_of_variation", "name": "Coefficient of variation", "applies_to": CONTINUOUS, "fn": _cv,
     "units": "dimensionless",
     "definition": "σ/μ — variability relative to the mean level. NOTE: this is the "
                   "CV of the continuous signal, not the ISI CV used for spike trains.",
     "reference": "Gerstner Ch.7 (CV concept)"},

    {"key": "amplitude_range", "name": "Amplitude (range)", "applies_to": CONTINUOUS, "fn": _amplitude,
     "units": None,
     "definition": "max − min — the full span of the signal's excursion in the window.",
     "reference": "—"},

    {"key": "rms", "name": "RMS amplitude", "applies_to": CONTINUOUS, "fn": _rms,
     "units": None,
     "definition": "Root-mean-square amplitude, sqrt(mean(A²)) — overall signal magnitude.",
     "reference": "—"},

    {"key": "integral", "name": "Integrated activity", "applies_to": CONTINUOUS, "fn": _integral,
     "units": "signal·ms",
     "definition": "∫A dt over the window — total accumulated activity ('dose').",
     "reference": "—"},

    {"key": "baseline", "name": "Baseline / steady state", "applies_to": CONTINUOUS, "fn": _baseline,
     "units": None,
     "definition": "Mean over the final 10% of the window — an estimate of the settled level.",
     "reference": "—"},

    {"key": "trend_slope", "name": "Trend (slope)", "applies_to": CONTINUOUS, "fn": _trend_slope,
     "units": "signal/ms",
     "definition": "Slope of a linear fit — net rise (>0) or fall (<0) across the window.",
     "reference": "—"},

    {"key": "peak", "name": "Peak", "applies_to": CONTINUOUS, "fn": _peak,
     "units": None,
     "definition": "Maximum value and the time at which it occurs.",
     "reference": "—"},

    {"key": "trough", "name": "Trough", "applies_to": CONTINUOUS, "fn": _trough,
     "units": None,
     "definition": "Minimum value and the time at which it occurs.",
     "reference": "—"},

    {"key": "autocorrelation_time", "name": "Autocorrelation time", "applies_to": CONTINUOUS, "fn": _autocorr_time,
     "units": "ms",
     "definition": "Characteristic decay time τ of the autocorrelation (first 1/e crossing) "
                   "— the timescale over which the signal 'remembers' itself.",
     "reference": "Gerstner Ch.7"},

    {"key": "dominant_frequency", "name": "Dominant frequency", "applies_to": (SIGNAL_TEMPORAL_AVERAGE,), "fn": _dominant_frequency,
     "units": "Hz",
     "definition": "Frequency of the largest peak in the power spectral density — the main "
                   "rhythm of the signal. Meaningful for fast TemporalAverage; for slow BOLD "
                   "the spectrum only covers sub-Hz fluctuations.",
     "reference": "Gerstner Ch.7 / Ch.13 (oscillations & synchrony)"},
]

# Between-region metrics (documented; computed by functional_connectivity()).
PAIR_METRIC_CATALOG = [
    {"key": "functional_connectivity", "name": "Functional connectivity", "applies_to": CONTINUOUS,
     "units": "dimensionless",
     "definition": "Pearson correlation between two regions' timeseries — statistical coupling "
                   "of their activity.",
     "reference": "Functional connectivity (Friston 2011)"},
    {"key": "cross_correlation_lag", "name": "Cross-correlation lag", "applies_to": CONTINUOUS,
     "units": "ms",
     "definition": "Time lag at which the cross-correlation of two regions peaks — a "
                   "directional-timing hint.",
     "reference": "Gerstner Ch.7"},
]

# Structural metrics (no signal). Computed in structure.py; catalog'd here for docs.
STRUCTURAL_METRIC_CATALOG = [
    {"key": "node_strength", "name": "Node strength", "applies_to": ("structural",),
     "units": "sum of weights",
     "definition": "Sum of incident connection weights — structural hubness.",
     "reference": "Rubinov & Sporns 2010"},
    {"key": "degree", "name": "Degree", "applies_to": ("structural",),
     "units": "count",
     "definition": "Number of connections incident to the region.",
     "reference": "Rubinov & Sporns 2010"},
]

# Spiking metrics (FUTURE — need spike trains, not TVB continuous output). Drafted
# as stubs so the catalog is complete for the doc and the guy has the signatures.
SPIKING_METRIC_CATALOG = [
    {"key": "mean_firing_rate", "name": "Mean firing rate", "applies_to": (SIGNAL_SPIKING,),
     "units": "Hz",
     "definition": "ν = spike count / duration — average rate over the window.",
     "reference": "Gerstner Ch.7"},
    {"key": "isi_cv", "name": "ISI coefficient of variation", "applies_to": (SIGNAL_SPIKING,),
     "units": "dimensionless",
     "definition": "CV of inter-spike intervals — spike-timing irregularity "
                   "(CV≈0 regular, CV≈1 Poisson-like, CV>1 bursty).",
     "reference": "Gerstner Ch.7"},
    {"key": "fano_factor", "name": "Fano factor", "applies_to": (SIGNAL_SPIKING,),
     "units": "dimensionless",
     "definition": "Variance/mean of spike counts across windows or trials — spike-count "
                   "variability.",
     "reference": "Gerstner Ch.7"},
    {"key": "population_rate", "name": "Population rate (PSTH)", "applies_to": (SIGNAL_SPIKING,),
     "units": "Hz",
     "definition": "A(t) = spikes across the population per bin per neuron — the population "
                   "activity time course.",
     "reference": "Gerstner Ch.7 (population activity A(t))"},
]


def _interpret(key, value) -> str:
    """Short contextual note for a few metrics (empty when not meaningful)."""
    if key == "trend_slope":
        if not np.isfinite(value):
            return ""
        return "rising over the window" if value > 0 else ("falling over the window" if value < 0 else "flat")
    if key == "coefficient_of_variation":
        if not np.isfinite(value):
            return ""
        return ("very regular (low relative variability)" if value < 0.1
                else "moderately variable" if value < 0.5 else "highly variable")
    if key == "dominant_frequency":
        f = value.get("freq_hz") if isinstance(value, dict) else value
        return f"main rhythm ≈ {f:.3g} Hz" if f and np.isfinite(f) else ""
    return ""


def compute_metrics(vd: ViewerData, region, t_start=None, t_end=None, metrics=None) -> list[dict]:
    """Compute the metrics suitable for this signal over [t_start, t_end].

    ``metrics`` optionally restricts to a subset of keys. Returns one dict per
    metric with value/units/definition/interpretation/reference.
    """
    idx, t, v = region_window(vd, region, t_start, t_end)
    st = vd.signal_type
    default_units = units_for(st)

    results = []
    for m in METRIC_CATALOG:
        if st not in m["applies_to"]:
            continue
        if metrics is not None and m["key"] not in metrics:
            continue
        value = m["fn"](t, v)
        results.append({
            "metric": m["key"],
            "name": m["name"],
            "value": value,
            "units": m["units"] or default_units,
            "signal": st,
            "definition": m["definition"],
            "interpretation": _interpret(m["key"], value),
            "reference": m["reference"],
        })
    return results


def functional_connectivity(vd: ViewerData, region_a, region_b, t_start=None, t_end=None) -> dict:
    """Pearson correlation (+ peak-cross-correlation lag) between two regions."""
    ia = resolve_region_index(vd, region_a)
    ib = resolve_region_index(vd, region_b)
    if not vd.has_signal:
        raise ValueError("No timeseries in this dataset.")
    mask = window_mask(vd.times, t_start, t_end)
    a = vd.activity[mask, ia]
    b = vd.activity[mask, ib]
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
    a0, b0 = a - np.mean(a), b - np.mean(b)
    xc = np.correlate(a0, b0, mode="full")
    lag_samples = int(np.argmax(xc) - (len(a) - 1))
    dt = float(np.median(np.diff(vd.times[mask]))) if mask.sum() > 1 else 1.0
    return {
        "region_a": vd.regions[ia]["name"],
        "region_b": vd.regions[ib]["name"],
        "pearson_r": r,
        "lag_ms": lag_samples * dt,
        "signal": vd.signal_type,
        "definition": "Pearson correlation of the two regions' timeseries (functional "
                      "connectivity); lag is the delay of peak cross-correlation.",
        "interpretation": _fc_interp(r),
        "reference": "Functional connectivity (Friston 2011); cross-correlation — Gerstner Ch.7.",
    }


def _fc_interp(r) -> str:
    if not np.isfinite(r):
        return ""
    a = abs(r)
    strength = "strong" if a >= 0.6 else "moderate" if a >= 0.3 else "weak"
    sign = "co-activation" if r >= 0 else "anti-correlation"
    return f"{strength} {sign} (r={r:.2f})"


def explain_activity(vd: ViewerData, region, t_start=None, t_end=None) -> dict:
    """HEADLINE tool — 'explain the curve for this area'.

    Bundles: region semantics + the metrics suitable for the signal + a compact
    shape description, ready for the LLM to narrate. This is the function the
    chatbot calls for "what was the activity of area X from t1 to t2".
    """
    info = get_region(vd, region)
    mets = compute_metrics(vd, region, t_start, t_end)
    by_key = {m["metric"]: m for m in mets}

    shape_bits = []
    if "trend_slope" in by_key and by_key["trend_slope"]["interpretation"]:
        shape_bits.append(by_key["trend_slope"]["interpretation"])
    if "coefficient_of_variation" in by_key and by_key["coefficient_of_variation"]["interpretation"]:
        shape_bits.append(by_key["coefficient_of_variation"]["interpretation"])
    if "dominant_frequency" in by_key and by_key["dominant_frequency"]["interpretation"]:
        shape_bits.append(by_key["dominant_frequency"]["interpretation"])

    return {
        "region": {"label": info["label"], "full_name": info.get("full_name"),
                   "description": info.get("description")},
        "window_ms": [t_start, t_end],
        "signal_type": vd.signal_type,
        "units": units_for(vd.signal_type),
        "shape": "; ".join(shape_bits) or "no notable trend",
        "metrics": mets,
    }


# ---------------------------------------------------------------------------
# Spiking metrics — FUTURE. Signatures fixed; bodies stubbed until the spiking
# viewer exists. Inputs are spike trains, NOT the continuous TVB signal.
# ---------------------------------------------------------------------------
def mean_firing_rate(spike_times_ms, t_start, t_end) -> dict:
    """ν = spikes / duration (Hz). NEEDS SPIKING DATA (list of spike times, ms)."""
    raise NotImplementedError("Spiking viewer not available yet — signature placeholder.")


def isi_cv(spike_times_ms) -> dict:
    """CV of inter-spike intervals. NEEDS SPIKING DATA."""
    raise NotImplementedError("Spiking viewer not available yet — signature placeholder.")


def fano_factor(spike_counts) -> dict:
    """Variance/mean of spike counts across windows/trials. NEEDS SPIKING DATA."""
    raise NotImplementedError("Spiking viewer not available yet — signature placeholder.")
