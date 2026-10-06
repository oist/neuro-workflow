import os
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


class NW_TVB_TimeSeriesPlot(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_timeseries_plot",
        stage="analysis",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Plots one recorded variable of one monitor from NW_TVB_Simulator as stacked region "
            "traces, optionally with its power spectrum, and saves it to a .npz file readable "
            "by the TVB brain viewers."
        ),
        parameters={
            "monitor": ParameterDefinition(
                default_value="",
                description=(
                    "Label of the monitor to plot (NW_TVB_Monitor 'label'). Examples: 'tavg', "
                    "'bold'. Empty = the first monitor connected to the simulator."
                ),
            ),
            "variable": ParameterDefinition(
                default_value="",
                description=(
                    "Recorded variable to plot, by name or index. Examples: 'V' "
                    "(Generic2dOscillator), 'r' (MontbrioPazoRoxin), 'S_e' "
                    "(ReducedWongWangExcInh), 'x2 - x1' (Epileptor), 'y1-y2' (JansenRit "
                    "EEG-like signal; any 'a-b' of two recorded variables works). Empty = the "
                    "first recorded variable."
                ),
            ),
            "regions": ParameterDefinition(
                default_value=10,
                description=(
                    "Which regions to draw: a number N (the first N regions), or a list of "
                    "indices/labels. Examples: 10; [0, 5, 12]; ['rV1', 'rA1']. Saving always "
                    "includes all regions."
                ),
            ),
            "normalize": ParameterDefinition(
                default_value=True,
                description=(
                    "True scales each trace to its own range so all regions are visible in the "
                    "stack; False keeps real amplitudes (useful to compare regions)."
                ),
            ),
            "show_spectrum": ParameterDefinition(
                default_value=False,
                description=(
                    "True adds a panel with the region-averaged power spectrum (log scale), "
                    "handy to check the dominant rhythm (e.g. ~10 Hz for JansenRit). Use with "
                    "fast monitors (period <= 5 ms)."
                ),
            ),
            "title": ParameterDefinition(
                default_value="TVB simulated activity",
                description="Figure title. Example: 'Human MPR - BOLD'.",
            ),
            "save_to_file": ParameterDefinition(
                default_value=False,
                description=(
                    "True saves 'time' (ms) and 'data' (time x regions) of the selected "
                    "variable, plus 'region_labels' and 'variable', to output_path. The file "
                    "can feed the bold_file / temporal_average_file inputs of the brain viewers."
                ),
            ),
            "output_path": ParameterDefinition(
                default_value="./results/timeseries.npz",
                description=(
                    "Where to save the .npz (relative to the project folder). Examples: "
                    "'./results/bold.npz', './results/tavg.npz'."
                ),
            ),
        },
        inputs={
            "tvb_results": PortDefinition(
                type=PortType.DICT,
                description="Labelled monitor results from NW_TVB_Simulator.",
            ),
        },
        outputs={
            "saved_file_path": PortDefinition(
                type=PortType.STR,
                description="Absolute path of the saved .npz (None if save_to_file is False).",
            ),
            "visualization_completed": PortDefinition(
                type=PortType.BOOL,
                description="True once the figure was drawn.",
            ),
        },
        methods={
            "plot_timeseries": MethodDefinition(
                description=(
                    "Select monitor and variable, optionally save all regions to .npz, and "
                    "plot the chosen regions (and spectrum)."
                ),
                inputs=["tvb_results"],
                outputs=["saved_file_path", "visualization_completed"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("plot_timeseries", self.plot_timeseries, method_key="plot_timeseries")

    def plot_timeseries(self, tvb_results) -> Dict[str, Any]:
        res = select_monitor(tvb_results, self._parameters["monitor"])
        name, x = select_variable(res, self._parameters["variable"])
        time = res["time"]
        labels = list(res["region_labels"])

        saved = None
        if self._parameters["save_to_file"]:
            saved = os.path.abspath(self._parameters["output_path"])
            os.makedirs(os.path.dirname(saved), exist_ok=True)
            np.savez(saved, time=time, data=x, region_labels=np.array(labels), variable=name)
            print(f"[{self.name}] saved {x.shape} '{name}' to {saved}")

        sel = self._parameters["regions"]
        if isinstance(sel, (int, float)):
            idx = list(range(min(int(sel), x.shape[1])))
        else:
            idx = [labels.index(r) if isinstance(r, str) else int(r) for r in sel]
        traces = x[:, idx]
        if self._parameters["normalize"]:
            rng = traces.max(0) - traces.min(0)
            rng[rng == 0] = 1.0
            traces = (traces - traces.min(0)) / rng

        n_panels = 2 if self._parameters["show_spectrum"] else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(13 if n_panels == 1 else 16, 7),
                                 squeeze=False, gridspec_kw={"width_ratios": [3, 1][:n_panels]})
        ax = axes[0, 0]
        offset = 1.0 if self._parameters["normalize"] else np.nanmax(np.ptp(traces, axis=0)) or 1.0
        for k in range(traces.shape[1]):
            ax.plot(time, traces[:, k] + k * offset, color="k", lw=0.7, alpha=0.7)
        ax.set_yticks(np.arange(len(idx)) * offset)
        ax.set_yticklabels([labels[i] for i in idx], fontsize=8)
        ax.set_xlabel("Time [ms]")
        ax.set_title(f"{self._parameters['title']} - {res['monitor_type']} '{name}'")

        if n_panels == 2:
            dt_s = (time[1] - time[0]) / 1000.0
            f = np.fft.rfftfreq(len(x), dt_s)
            p = (np.abs(np.fft.rfft(x - x.mean(0), axis=0)) ** 2).mean(1)
            axes[0, 1].semilogy(f[1:], p[1:], color="tab:blue")
            axes[0, 1].set_xlim(0, min(100, f[-1]))
            axes[0, 1].set_xlabel("Frequency [Hz]")
            axes[0, 1].set_title(f"Power spectrum (peak {f[1:][p[1:].argmax()]:.1f} Hz)")
        fig.tight_layout()
        plt.show()
        return {"saved_file_path": saved, "visualization_completed": True}


def select_monitor(tvb_results: Dict[str, Any], label: str) -> Dict[str, Any]:
    if not tvb_results:
        raise ValueError("tvb_results is empty.")
    if not label:
        return next(iter(tvb_results.values()))
    if label not in tvb_results:
        raise ValueError(f"Monitor '{label}' not in results; available: {list(tvb_results)}")
    return tvb_results[label]


def select_variable(res: Dict[str, Any], variable):
    """Return (name, array time x regions) for a variable name, index or 'a-b' difference."""
    names = list(res["variables"])
    data = res["data"][..., 0]  # first mode
    if variable in ("", None):
        return names[0], data[:, 0, :]
    if isinstance(variable, str):
        if variable in names:
            return variable, data[:, names.index(variable), :]
        parts = [p.strip() for p in variable.split("-")]
        if len(parts) == 2 and all(p in names for p in parts):
            return variable, data[:, names.index(parts[0]), :] - data[:, names.index(parts[1]), :]
        raise ValueError(f"Variable '{variable}' not recorded; available: {names}")
    return names[int(variable)], data[:, int(variable), :]
