import json
import os
from typing import Dict, Any

import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


# Reference data-viz palette: categorical slots in fixed order, status colours, text inks.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_GOOD, _CRITICAL = "#0ca30c", "#d03b3b"
_INK, _INK_2, _GRID, _SURFACE, _BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f0efec"


class BG_Maq_RestingAnalysis(Node):
    """Validate and plot a BG_Maq resting-state run.

    Compares the mean firing rate of every nucleus with its physiological resting
    range (bgParams['normalrate']), and draws three figures: rate vs target range,
    a spike raster and the smoothed population rates A(t).
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_resting_analysis",
        stage="analysis",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — normalrate targets from top_BG_nest3/bgParams.py",
        description=(
            "Check resting firing rates against the physiological ranges, report a pass/fail table and "
            "a score (fraction of nuclei in range), and plot rates vs targets, a raster and A(t)."
        ),
        parameters={
            "target_rates": ParameterDefinition(
                default_value={
                    "MSN_d1": [0.05, 1.0], "MSN_d2": [0.05, 1.0],
                    "FSI": [7.8, 14.0], "STN": [15.2, 22.8],
                    "GPe": [55.7, 74.5], "GPi": [59.1, 79.5],
                },
                description=(
                    "Resting firing-rate range [min, max] in Hz per nucleus (normalrate). Only nuclei "
                    "listed here are validated and plotted (striatum -> output order)."
                ),
                unit="Hz",
            ),
            "raster_window_ms": ParameterDefinition(
                default_value=500.0,
                description="Length of the raster window, starting at the end of the warm-up.",
                constraints={"min": 10.0}, unit="ms",
            ),
            "raster_max_neurons": ParameterDefinition(
                default_value=50,
                description="Neurons shown per nucleus in the raster.",
                constraints={"min": 1, "max": 1000},
            ),
            "smoothing_ms": ParameterDefinition(
                default_value=5.0,
                description="Width of the boxcar used to smooth A(t) in the plot (0 = raw).",
                constraints={"min": 0.0}, unit="ms",
            ),
            "save_figures": ParameterDefinition(
                default_value=True,
                description="Save the figures as PNG and the validation as validation.json in output_dir.",
            ),
            "show_figures": ParameterDefinition(
                default_value=True,
                description="Display the figures inline (Jupyter).",
            ),
        },
        inputs={
            "mean_fr": PortDefinition(type=PortType.DICT, description="From BG_Maq_RestingState."),
            "at_fr": PortDefinition(type=PortType.DICT, description="From BG_Maq_RestingState."),
            "spikes": PortDefinition(type=PortType.OBJECT, description="From BG_Maq_RestingState."),
            "run_info": PortDefinition(type=PortType.DICT, description="From BG_Maq_RestingState."),
        },
        outputs={
            "validation": PortDefinition(
                type=PortType.DICT,
                description="{nucleus: {'rate_hz', 'min_hz', 'max_hz', 'in_range', 'distance_hz'}}.",
            ),
            "score": PortDefinition(
                type=PortType.FLOAT,
                description="Fraction of validated nuclei whose rate lies inside its target range (0–1).",
            ),
            "figure_paths": PortDefinition(
                type=PortType.LIST, description="Paths of the saved PNG figures.",
            ),
        },
        methods={
            "analyze": MethodDefinition(
                description="Validate firing rates against targets and draw the summary figures.",
                inputs=["mean_fr", "at_fr", "spikes", "run_info"],
                outputs=["validation", "score", "figure_paths"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("analyze", self.analyze, method_key="analyze")

    def analyze(self, mean_fr, at_fr, spikes, run_info) -> Dict[str, Any]:
        p = self._parameters
        targets = {k: v for k, v in p["target_rates"].items() if k in mean_fr}
        # fixed anatomical order: the web app may store the dict with re-sorted keys
        order = ["MSN_d1", "MSN_d2", "MSN", "FSI", "STN", "GPe", "GPi", "GPi_fake", "CSN", "PTN", "CMPf"]
        nuclei = sorted(targets, key=lambda k: order.index(k) if k in order else len(order))

        validation = {}
        for nuc in nuclei:
            lo, hi = targets[nuc]
            r = mean_fr[nuc]
            validation[nuc] = {
                "rate_hz": r, "min_hz": lo, "max_hz": hi,
                "in_range": bool(lo <= r <= hi),
                "distance_hz": float(0.0 if lo <= r <= hi else (lo - r if r < lo else r - hi)),
            }
        score = sum(v["in_range"] for v in validation.values()) / max(len(validation), 1)

        print(f"[{self.name}] resting-state validation ({score * 100:.0f}% in range)")
        print(f"  {'nucleus':8s} {'rate (Hz)':>10s}   {'target (Hz)':>14s}")
        for nuc, v in validation.items():
            mark = "✓ in range" if v["in_range"] else "✗ out of range"
            print(f"  {nuc:8s} {v['rate_hz']:10.2f}   {v['min_hz']:6.2f} – {v['max_hz']:<6.2f} {mark}")
        others = [k for k in mean_fr if k not in validation]
        if others:
            print("  not validated: " + ", ".join(f"{k}={mean_fr[k]:.2f}" for k in others))

        figure_paths = []
        out_dir = run_info.get("data_path", ".")
        if p["save_figures"] or p["show_figures"]:
            figure_paths = self._plot(nuclei, validation, at_fr, spikes, run_info, out_dir)
        if p["save_figures"]:
            with open(os.path.join(out_dir, "validation.json"), "w") as f:
                json.dump({"score": score, "nuclei": validation}, f, indent=1)

        return {"validation": validation, "score": float(score), "figure_paths": figure_paths}

    # ------------------------------------------------------------- figures
    def _plot(self, nuclei, validation, at_fr, spikes, run_info, out_dir):
        import matplotlib.pyplot as plt

        p = self._parameters
        colors = {nuc: _SERIES[i % len(_SERIES)] for i, nuc in enumerate(nuclei)}
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": _GRID, "axes.labelcolor": _INK_2,
                             "xtick.color": _INK_2, "ytick.color": _INK_2, "text.color": _INK,
                             "figure.facecolor": _SURFACE, "axes.facecolor": _SURFACE})
        paths = []

        # 1) mean rate vs physiological range (log scale: MSN ~0.1 Hz, GPi ~70 Hz)
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(nuclei) + 1.4))
        for i, nuc in enumerate(nuclei[::-1]):
            v = validation[nuc]
            ax.barh(i, v["max_hz"] - v["min_hz"], left=v["min_hz"], height=0.5, color=_BAND, edgecolor="none")
            status = _GOOD if v["in_range"] else _CRITICAL
            rate = max(v["rate_hz"], 1e-3)
            ax.plot(rate, i, "o", ms=8, color=status, mec=_SURFACE, mew=2, zorder=3)
            label = (f"{v['rate_hz']:.3g}" if v["rate_hz"] < 1 else f"{v['rate_hz']:.2f}") + " Hz  " + ("✓" if v["in_range"] else "✗")
            ax.annotate(label, (rate, i), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9, color=_INK)
        ax.set_yticks(range(len(nuclei)))
        ax.set_yticklabels(nuclei[::-1], color=_INK)
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 300)
        ax.set_xlabel("Mean firing rate (Hz, log scale) — grey band = physiological resting range")
        ax.grid(axis="x", color=_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        n_ok = sum(validation[n]["in_range"] for n in nuclei)
        ax.set_title(f"Resting-state firing rates: {n_ok}/{len(nuclei)} nuclei in range",
                     loc="left", fontsize=11, color=_INK)
        fig.tight_layout()
        paths.append(self._finish(fig, out_dir, "fig_rates_vs_targets.png"))

        # 2) raster
        t0 = run_info["analysis_window_ms"][0]
        t1 = min(t0 + float(p["raster_window_ms"]), run_info["analysis_window_ms"][1])
        n_max = int(p["raster_max_neurons"])
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(nuclei) + 2.2))
        yticks = []
        for i, nuc in enumerate(nuclei):
            sp = spikes[nuc]
            idx = sp["senders"] - sp["first_id"]
            m = (idx < n_max) & (sp["times"] >= t0) & (sp["times"] < t1)
            offset = (len(nuclei) - 1 - i) * (n_max + 5)
            ax.scatter(sp["times"][m], idx[m] + offset, s=2, color=colors[nuc], linewidths=0)
            yticks.append(offset + n_max / 2)
        ax.set_yticks(yticks)
        ax.set_yticklabels(nuclei, color=_INK)
        ax.set_xlim(t0, t1)
        ax.set_xlabel("Time (ms)")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"Spike raster — first {n_max} neurons per nucleus", loc="left", fontsize=11, color=_INK, pad=14)
        fig.tight_layout()
        paths.append(self._finish(fig, out_dir, "fig_raster.png"))

        # 3) population rate A(t), one panel (own y-axis) per nucleus
        t = np.asarray(at_fr["time_ms"])
        step = float(t[1] - t[0]) if len(t) > 1 else 1.0
        k = max(int(round(float(p["smoothing_ms"]) / step)), 1)
        fig, axes = plt.subplots(len(nuclei), 1, figsize=(9, 1.25 * len(nuclei) + 0.8), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, nuc in zip(axes, nuclei):
            a = np.asarray(at_fr["rates"][nuc])
            if k > 1:
                a = np.convolve(a, np.ones(k) / k, mode="same")
            ax.plot(t, a, color=colors[nuc], lw=1.2)
            ax.axhspan(validation[nuc]["min_hz"], validation[nuc]["max_hz"], color=_BAND, zorder=0)
            ax.set_ylabel(nuc, rotation=0, ha="right", va="center", color=_INK)
            ax.grid(axis="y", color=_GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[-1].set_xlabel("Time (ms)")
        axes[0].set_title(f"Population rate A(t) in Hz ({p['smoothing_ms']} ms boxcar); grey band = target range",
                          loc="left", fontsize=11, color=_INK)
        fig.tight_layout()
        paths.append(self._finish(fig, out_dir, "fig_population_rates.png"))
        return [x for x in paths if x]

    def _finish(self, fig, out_dir, filename):
        import matplotlib.pyplot as plt

        path = None
        if self._parameters["save_figures"]:
            path = os.path.join(out_dir, filename)
            fig.savefig(path, dpi=130)
        if self._parameters["show_figures"]:
            plt.show()
        plt.close(fig)
        return path
