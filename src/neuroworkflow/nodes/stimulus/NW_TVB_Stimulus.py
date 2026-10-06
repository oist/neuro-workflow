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

from tvb.simulator.lab import patterns
from tvb.datatypes import equations


class NW_TVB_Stimulus(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_stimulus",
        stage="stimulus",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Builds an external stimulus for chosen brain regions: a spatial pattern (which "
            "regions, how strongly) times a temporal waveform (Gaussian pulse, sinusoid, pulse "
            "train, ...). It is added to the model's stimulus_variables during simulation."
        ),
        parameters={
            "regions": ParameterDefinition(
                default_value=[0],
                description=(
                    "Regions to stimulate, by index (0-based row of the connectivity) or by "
                    "label. Examples: [0, 7, 13]; ['rV1', 'lV1'] for both primary visual "
                    "cortices in connectivity_76; ['L_A10'] in the marmoset connectome."
                ),
            ),
            "weights": ParameterDefinition(
                default_value=1.0,
                description=(
                    "Stimulus strength per region: one number for all regions, or a list "
                    "matching 'regions'. Examples: 1.0; [0.25, 0.125, 0.0625] for a "
                    "decreasing gradient. The final input is weight * temporal waveform."
                ),
            ),
            "temporal_equation": ParameterDefinition(
                default_value="PulseTrain",
                description=(
                    "Shape of the stimulus over time. 'PulseTrain' - rectangular pulses "
                    "(T period, tau width, amp, onset; all ms). 'Gaussian' - single bump "
                    "(midpoint, sigma, amp). 'DoubleGaussian' - two overlapping bumps. "
                    "'Sinusoid' / 'Cosine' - periodic drive (frequency in kHz: 0.01 = 10 Hz; "
                    "amp). 'Alpha' - alpha-function synaptic-like response (onset, alpha, "
                    "beta). 'Linear' - ramp a*t + b. Example: 'Sinusoid'."
                ),
                constraints={
                    "allowed_values": [
                        "PulseTrain",
                        "Gaussian",
                        "DoubleGaussian",
                        "Sinusoid",
                        "Cosine",
                        "Alpha",
                        "Linear",
                    ]
                },
            ),
            "equation_params": ParameterDefinition(
                default_value={"onset": 1000.0, "T": 500.0, "tau": 50.0, "amp": 1.0},
                description=(
                    "Parameters of the temporal equation (times in ms); unspecified ones keep "
                    "TVB defaults. Examples: PulseTrain {'onset': 1000, 'T': 500, 'tau': 50, "
                    "'amp': 1.0} (50 ms pulse every 500 ms from 1 s); Gaussian {'midpoint': "
                    "4000, 'sigma': 200, 'amp': 1.0}; Sinusoid {'frequency': 0.01, 'amp': 0.5} "
                    "(10 Hz); Alpha {'onset': 500, 'alpha': 13, 'beta': 42}; Linear {'a': "
                    "0.001, 'b': 0}."
                ),
            ),
            "show_plot": ParameterDefinition(
                default_value=True,
                description="True plots the stimulated regions and the waveform over plot_length.",
            ),
            "plot_length": ParameterDefinition(
                default_value=3000.0,
                description=(
                    "Time span in ms shown in the waveform plot (plot only, does not affect "
                    "the simulation). Usually the simulation length. Example: 10000."
                ),
                constraints={"min": 1.0, "max": 1e8},
                unit="ms",
            ),
        },
        inputs={
            "tvb_connectivity": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "Connectome from NW_TVB_Connectivity; gives the number of regions and the "
                    "labels used to resolve region names."
                ),
            ),
        },
        outputs={
            "tvb_stimulus": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "TVB StimuliRegion object for NW_TVB_Simulator (and for the brain viewers, "
                    "which mark the stimulated regions)."
                ),
            ),
            "visualization_completed": PortDefinition(
                type=PortType.BOOL,
                description="True once the stimulus figure was drawn (or skipped).",
            ),
        },
        methods={
            "build_stimulus": MethodDefinition(
                description=(
                    "Resolve the regions, build the spatial weight vector and the temporal "
                    "equation, and create the TVB StimuliRegion."
                ),
                inputs=["tvb_connectivity"],
                outputs=["tvb_stimulus"],
            ),
            "plot_stimulus": MethodDefinition(
                description="Plot spatial weights per region and the temporal waveform.",
                inputs=["tvb_stimulus", "tvb_connectivity"],
                outputs=["visualization_completed"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build_stimulus", self.build_stimulus, method_key="build_stimulus")
        self.add_process_step("plot_stimulus", self.plot_stimulus, method_key="plot_stimulus")

    def build_stimulus(self, tvb_connectivity) -> Dict[str, Any]:
        labels = [str(l) for l in tvb_connectivity.region_labels]
        indices = []
        for r in self._parameters["regions"]:
            if isinstance(r, str):
                if r not in labels:
                    raise ValueError(f"Region label '{r}' not in connectivity; e.g. {labels[:8]}")
                indices.append(labels.index(r))
            else:
                if not 0 <= int(r) < len(labels):
                    raise ValueError(f"Region index {r} out of range (0..{len(labels) - 1})")
                indices.append(int(r))

        w = np.atleast_1d(np.asarray(self._parameters["weights"], dtype=float))
        if w.size == 1:
            w = np.full(len(indices), w[0])
        if w.size != len(indices):
            raise ValueError(f"weights has {w.size} values but {len(indices)} regions were given.")

        weighting = np.zeros(len(labels))
        weighting[indices] = w

        eq_type = self._parameters["temporal_equation"]
        eqn = getattr(equations, eq_type)()
        for k, v in (self._parameters["equation_params"] or {}).items():
            if k not in eqn.parameters:
                raise ValueError(f"{eq_type} has no parameter '{k}'; valid: {list(eqn.parameters)}")
            eqn.parameters[k] = float(v)

        stim = patterns.StimuliRegion(temporal=eqn, connectivity=tvb_connectivity, weight=weighting)
        print(f"[{self.name}] {eq_type} {dict(eqn.parameters)} on "
              f"{[labels[i] for i in indices]}")
        return {"tvb_stimulus": stim}

    def plot_stimulus(self, tvb_stimulus, tvb_connectivity) -> Dict[str, Any]:
        if not self._parameters["show_plot"]:
            return {"visualization_completed": True}
        t = np.arange(0.0, float(self._parameters["plot_length"]), 1.0)
        waveform = tvb_stimulus.temporal.evaluate(t)
        weights = np.asarray(tvb_stimulus.weight)
        idx = np.nonzero(weights)[0]
        labels = np.asarray(tvb_connectivity.region_labels)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 2]})
        axes[0].bar(range(len(idx)), weights[idx], color="tab:blue")
        axes[0].set_xticks(range(len(idx)))
        axes[0].set_xticklabels(labels[idx], rotation=90, fontsize=8)
        axes[0].set_title("Stimulated regions (weight)")
        axes[1].plot(t, waveform, color="k")
        axes[1].set_xlabel("Time [ms]")
        axes[1].set_title(f"Temporal profile: {type(tvb_stimulus.temporal).__name__}")
        fig.tight_layout()
        plt.show()
        return {"visualization_completed": True}
