import os
import sys
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

from tvb.simulator.lab import connectivity


class NW_TVB_Connectivity(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_connectivity",
        stage="connectivity",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Loads a structural connectome (weights, tract lengths, region labels and centres) "
            "from a TVB zip file and prepares it for whole-brain simulation: removes "
            "self-connections, normalises weights and sets the conduction speed (signal delays)."
        ),
        parameters={
            "connectivity_file": ParameterDefinition(
                default_value="./data/connectivity_76.zip",
                description=(
                    "TVB connectivity zip (weights.txt, tract_lengths.txt, centres.txt, "
                    "region_labels.txt ...). Accepts an absolute path; a path relative to the "
                    "project folder (where the workflow runs), e.g. './data/connectivity_76.zip' "
                    "to keep the data with the project; a path relative to the codes folder; or "
                    "a bare file name from the tvb_data package. Examples: "
                    "'neuroworkflow/data/tvb_data/connectivity_human.zip' (human), "
                    "'neuroworkflow/data/tvb_data/connectivity_marmoset.zip' (marmoset), "
                    "'connectivity_76.zip' / 'connectivity_96.zip' / 'connectivity_192.zip' "
                    "(TVB demo human connectomes with 76/96/192 regions)."
                ),
            ),
            "normalize_weights": ParameterDefinition(
                default_value="max",
                description=(
                    "How to rescale the connection weights so coupling strengths are comparable "
                    "across connectomes. 'max' - divide by the largest weight (max=1; the model "
                    "presets assume this, as in the TVB tutorials). 'row_sum' - each region's "
                    "incoming weights sum to 1. 'none' - keep the file's raw weights (V2 "
                    "behaviour). Example: 'max'."
                ),
                constraints={"allowed_values": ["max", "row_sum", "none"]},
            ),
            "remove_self_connections": ParameterDefinition(
                default_value=True,
                description=(
                    "True sets the diagonal of the weight matrix to 0, so no region excites "
                    "itself through the long-range network (standard). False keeps the file's "
                    "diagonal."
                ),
            ),
            "conduction_speed": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Axonal conduction speed in mm/ms (= m/s); the delay between two regions is "
                    "tract_length / speed. Typical values are 3-10 mm/ms; delays matter for "
                    "fast rhythms (alpha/gamma) and are usually negligible for BOLD. 0 = "
                    "infinite speed, no delays (fastest; V2 behaviour). Example: 3.0."
                ),
                constraints={"min": 0.0, "max": 1000.0},
                unit="mm/ms",
            ),
            "show_plot": ParameterDefinition(
                default_value=True,
                description=(
                    "True draws the weight and tract-length matrices with region labels; "
                    "set it to False in parameter sweeps or optimisation runs."
                ),
            ),
        },
        inputs={
            "connectivity_file_path": PortDefinition(
                type=PortType.STR,
                description=(
                    "Optional path to a connectivity zip produced by another node; overrides "
                    "the connectivity_file parameter."
                ),
                optional=True,
            ),
        },
        outputs={
            "tvb_connectivity": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "Configured TVB Connectivity object, ready for NW_TVB_Model, "
                    "NW_TVB_Stimulus, NW_TVB_Simulator and the brain viewers."
                ),
            ),
            "connectivity_info": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Summary: n_regions, region_labels, normalisation, conduction_speed, "
                    "mean_in_strength (average sum of incoming weights) and max_delay_ms."
                ),
            ),
            "visualization_completed": PortDefinition(
                type=PortType.BOOL,
                description="True once the connectivity figure was drawn (or skipped).",
            ),
        },
        methods={
            "load_connectivity": MethodDefinition(
                description=(
                    "Read the zip, remove self-connections, normalise the weights, set the "
                    "conduction speed and configure the TVB Connectivity."
                ),
                inputs=["connectivity_file_path"],
                outputs=["tvb_connectivity", "connectivity_info"],
            ),
            "plot_connectivity": MethodDefinition(
                description="Plot the weight and tract-length matrices with region labels.",
                inputs=["tvb_connectivity"],
                outputs=["visualization_completed"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step(
            "load_connectivity", self.load_connectivity, method_key="load_connectivity"
        )
        self.add_process_step(
            "plot_connectivity", self.plot_connectivity, method_key="plot_connectivity"
        )

    def load_connectivity(self, connectivity_file_path: str = None) -> Dict[str, Any]:
        file_path = self._resolve_path(connectivity_file_path or self._parameters["connectivity_file"])
        print(f"[{self.name}] Loading connectivity: {file_path}")

        con = connectivity.Connectivity.from_file(file_path)
        weights = np.array(con.weights, dtype=float)
        if self._parameters["remove_self_connections"]:
            np.fill_diagonal(weights, 0.0)

        norm = self._parameters["normalize_weights"]
        if norm == "max":
            weights = weights / weights.max()
        elif norm == "row_sum":
            rows = weights.sum(axis=1, keepdims=True)
            rows[rows == 0] = 1.0
            weights = weights / rows
        con.weights = weights

        speed = float(self._parameters["conduction_speed"])
        con.speed = np.array([speed if speed > 0 else sys.float_info.max])
        con.configure()

        max_delay = float(con.tract_lengths.max() / speed) if speed > 0 else 0.0
        info = {
            "n_regions": int(con.number_of_regions),
            "region_labels": [str(l) for l in con.region_labels],
            "normalisation": norm,
            "conduction_speed": speed if speed > 0 else float("inf"),
            "mean_in_strength": float(weights.sum(axis=1).mean()),
            "max_delay_ms": max_delay,
            "file": file_path,
        }
        print(f"[{self.name}] {info['n_regions']} regions, weights normalised '{norm}', "
              f"mean in-strength {info['mean_in_strength']:.2f}, max delay {max_delay:.1f} ms")
        return {"tvb_connectivity": con, "connectivity_info": info}

    def plot_connectivity(self, tvb_connectivity) -> Dict[str, Any]:
        if not self._parameters["show_plot"]:
            return {"visualization_completed": True}

        labels = tvb_connectivity.region_labels
        n = len(labels)
        fontsize = 7 if n <= 100 else 4

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle("TVB Structural Connectivity", fontsize=18)
        for ax, mat, title in [
            (axes[0], tvb_connectivity.weights, "Weights"),
            (axes[1], tvb_connectivity.tract_lengths, "Tract lengths [mm]"),
        ]:
            im = ax.imshow(mat, interpolation="nearest", aspect="equal", cmap="viridis")
            ax.set_title(title)
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, fontsize=fontsize, rotation=90)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=fontsize)
            fig.colorbar(im, ax=ax, shrink=0.5)
        fig.tight_layout()
        plt.show()
        return {"visualization_completed": True}

    @staticmethod
    def _resolve_path(file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path
        codes_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        candidates = [os.path.abspath(file_path), os.path.join(codes_root, file_path)]
        try:
            import tvb_data
            candidates.append(os.path.join(os.path.dirname(tvb_data.__file__), "connectivity", file_path))
        except ImportError:
            pass
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Connectivity file '{file_path}' not found; tried {candidates}")
