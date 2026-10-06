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


class NW_TVB_FunctionalConnectivity(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_functional_connectivity",
        stage="analysis",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Computes functional connectivity (FC, Pearson correlation between regions), its "
            "dynamics (FCD, from edge co-activations or sliding windows) and fit metrics "
            "(FC-SC and FC-empirical correlation) from a simulated monitor, typically BOLD; "
            "the metrics can be optimisation objectives (e.g. tune coupling G)."
        ),
        parameters={
            "monitor": ParameterDefinition(
                default_value="bold",
                description=(
                    "Label of the monitor to analyse (NW_TVB_Monitor 'label'). BOLD is the "
                    "usual choice to compare with fMRI; a TemporalAverage monitor gives "
                    "'neural' FC. Examples: 'bold', 'tavg'. Empty = first monitor."
                ),
            ),
            "variable": ParameterDefinition(
                default_value="",
                description=(
                    "Recorded variable to correlate, by name or index. Examples: 'V' "
                    "(MontbrioPazoRoxin BOLD, as in Rabuffo 2021), 'S_e' "
                    "(ReducedWongWangExcInh), 'x' (SupHopf). Empty = first recorded variable."
                ),
            ),
            "fcd_method": ParameterDefinition(
                default_value="edge",
                description=(
                    "How FC dynamics (FCD, a time x time similarity matrix) are computed. "
                    "'edge' - correlate edge co-activation patterns (product of z-scored "
                    "signals of each region pair) between time points; no window needed "
                    "(Faskowitz 2020, Rabuffo 2021). 'sliding_window' - correlate FC matrices "
                    "of overlapping windows (window_length / window_step). 'none' - skip FCD."
                ),
                constraints={"allowed_values": ["edge", "sliding_window", "none"]},
            ),
            "window_length": ParameterDefinition(
                default_value=60000.0,
                description=(
                    "Window length in ms for fcd_method='sliding_window'. Example: 60000 "
                    "(60 s, i.e. 30 BOLD samples at TR=2 s)."
                ),
                constraints={"min": 1.0, "max": 1e8},
                unit="ms",
            ),
            "window_step": ParameterDefinition(
                default_value=2000.0,
                description=(
                    "Shift between consecutive windows in ms for 'sliding_window'. Example: "
                    "2000 (one TR)."
                ),
                constraints={"min": 1.0, "max": 1e8},
                unit="ms",
            ),
            "empirical_fc_file": ParameterDefinition(
                default_value="",
                description=(
                    "Optional empirical FC matrix (regions x regions, same region order as the "
                    "connectome) to compare with: .npy, .npz (first array), .txt or .csv. "
                    "Example: './data/empirical_fc_76.npy'. Empty = no comparison."
                ),
            ),
            "target_fc_emp_corr": ParameterDefinition(
                default_value=0.5,
                description=(
                    "Target for the correlation between simulated and empirical FC "
                    "(fc_metrics.fc_emp_corr), usable as an optimisation objective: set "
                    "is_objective with measures '<this node>.fc_metrics.fc_emp_corr'. Typical "
                    "whole-brain models reach 0.3-0.6."
                ),
                is_objective=False,
                objective_range=[0.4, 1.0],
                measures="",
            ),
            "show_plot": ParameterDefinition(
                default_value=True,
                description="True draws SC, simulated FC, FCD and (if given) empirical FC.",
            ),
            "save_to_file": ParameterDefinition(
                default_value=False,
                description="True saves fc, fcd and the metrics to output_path (.npz).",
            ),
            "output_path": ParameterDefinition(
                default_value="./results/fc.npz",
                description="Where to save the .npz. Example: './results/fc_bold.npz'.",
            ),
        },
        inputs={
            "tvb_results": PortDefinition(
                type=PortType.DICT,
                description="Labelled monitor results from NW_TVB_Simulator.",
            ),
            "tvb_connectivity": PortDefinition(
                type=PortType.OBJECT,
                description="Optional connectome, to compute the FC-SC correlation and plot SC.",
                optional=True,
            ),
        },
        outputs={
            "fc_matrix": PortDefinition(
                type=PortType.OBJECT,
                description="Simulated FC, numpy array (regions x regions).",
            ),
            "fcd_matrix": PortDefinition(
                type=PortType.OBJECT,
                description="FCD, numpy array (time points or windows, squared); None if skipped.",
            ),
            "fc_metrics": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Scalar metrics (optimisation targets): mean_fc, fc_sc_corr, fc_emp_corr, "
                    "fcd_mean, fcd_var (spread of FCD values, a metastability proxy), n_samples."
                ),
            ),
        },
        methods={
            "compute_fc": MethodDefinition(
                description=(
                    "Select monitor and variable, compute FC, FCD and the fit metrics, "
                    "optionally save and plot them."
                ),
                inputs=["tvb_results", "tvb_connectivity"],
                outputs=["fc_matrix", "fcd_matrix", "fc_metrics"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("compute_fc", self.compute_fc, method_key="compute_fc")

    def compute_fc(self, tvb_results, tvb_connectivity=None) -> Dict[str, Any]:
        res = self._select_monitor(tvb_results, self._parameters["monitor"])
        name, x = self._select_variable(res, self._parameters["variable"])
        n_t, n_r = x.shape
        if n_t < 3:
            raise ValueError(f"Only {n_t} samples in '{name}': simulate longer or reduce the transient.")
        iu = np.triu_indices(n_r, k=1)

        fc = np.corrcoef(x.T)
        metrics = {"mean_fc": float(np.nanmean(fc[iu])), "n_samples": float(n_t)}

        sc = None
        if tvb_connectivity is not None:
            sc = np.asarray(tvb_connectivity.weights)
            metrics["fc_sc_corr"] = self._corr(fc[iu], sc[iu])

        emp = self._load_empirical(n_r)
        if emp is not None:
            metrics["fc_emp_corr"] = self._corr(fc[iu], emp[iu])

        fcd = self._fcd(x, res["time"], iu)
        if fcd is not None:
            fu = fcd[np.triu_indices(len(fcd), k=1)]
            metrics["fcd_mean"] = float(np.nanmean(fu))
            metrics["fcd_var"] = float(np.nanvar(fu))

        print(f"[{self.name}] FC of '{name}' ({n_t} samples x {n_r} regions): "
              + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "n_samples"))

        if self._parameters["save_to_file"]:
            path = os.path.abspath(self._parameters["output_path"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, fc=fc, fcd=fcd if fcd is not None else np.array([]),
                     region_labels=np.array(res["region_labels"]), **metrics)
            print(f"[{self.name}] saved to {path}")

        if self._parameters["show_plot"]:
            self._plot(sc, fc, fcd, emp, name, metrics)

        return {"fc_matrix": fc, "fcd_matrix": fcd, "fc_metrics": metrics}

    def _fcd(self, x, time, iu):
        method = self._parameters["fcd_method"]
        if method == "none":
            return None
        if method == "edge":
            z = (x - x.mean(0)) / np.where(x.std(0) == 0, 1.0, x.std(0))
            edges = z[:, iu[0]] * z[:, iu[1]]          # time x edges co-activation
            return np.corrcoef(edges)
        dt = time[1] - time[0]
        win = max(3, int(round(self._parameters["window_length"] / dt)))
        step = max(1, int(round(self._parameters["window_step"] / dt)))
        if win >= len(x):
            raise ValueError(
                f"window_length ({win} samples) must be shorter than the signal ({len(x)} samples)."
            )
        fcs = [np.corrcoef(x[s:s + win].T)[iu] for s in range(0, len(x) - win + 1, step)]
        return np.corrcoef(np.array(fcs))

    def _load_empirical(self, n_r):
        path = self._parameters["empirical_fc_file"]
        if not path:
            return None
        path = os.path.abspath(path)
        if path.endswith(".npy"):
            emp = np.load(path)
        elif path.endswith(".npz"):
            npz = np.load(path)
            emp = npz[npz.files[0]]
        else:
            emp = np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
        if emp.shape != (n_r, n_r):
            raise ValueError(f"Empirical FC has shape {emp.shape}, expected ({n_r}, {n_r}).")
        return emp

    @staticmethod
    def _corr(a, b) -> float:
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])

    def _plot(self, sc, fc, fcd, emp, name, metrics):
        panels = [(m, t, c) for m, t, c in [
            (sc, "Structural connectivity", "viridis"),
            (fc, f"Simulated FC ('{name}')", "RdBu_r"),
            (emp, "Empirical FC", "RdBu_r"),
            (fcd, f"FCD ({self._parameters['fcd_method']})", "plasma"),
        ] if m is not None]
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
        for ax, (mat, title, cmap) in zip(axes[0], panels):
            kw = {"vmin": -1, "vmax": 1} if cmap == "RdBu_r" else {}
            im = ax.imshow(mat, cmap=cmap, interpolation="nearest", **kw)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "n_samples"))
        fig.tight_layout()
        plt.show()

    @staticmethod
    def _select_monitor(tvb_results, label):
        if not tvb_results:
            raise ValueError("tvb_results is empty.")
        if not label:
            return next(iter(tvb_results.values()))
        if label not in tvb_results:
            raise ValueError(f"Monitor '{label}' not in results; available: {list(tvb_results)}")
        return tvb_results[label]

    @staticmethod
    def _select_variable(res, variable):
        names = list(res["variables"])
        data = res["data"][..., 0]
        if variable in ("", None):
            return names[0], data[:, 0, :]
        if isinstance(variable, str):
            if variable not in names:
                raise ValueError(f"Variable '{variable}' not recorded; available: {names}")
            return variable, data[:, names.index(variable), :]
        return names[int(variable)], data[:, int(variable), :]
