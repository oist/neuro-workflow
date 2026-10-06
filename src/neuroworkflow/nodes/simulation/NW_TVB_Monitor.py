from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType

from tvb.simulator import monitors
from tvb.datatypes import equations


class NW_TVB_Monitor(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_tvb_monitor",
        stage="simulation",
        tool="TVB",
        model_source="https://github.com/the-virtual-brain/tvb-root",
        description=(
            "Defines one recording of the simulation (raw activity, time-averaged activity, "
            "whole-brain average, BOLD fMRI or afferent coupling). Connect any number of "
            "monitors to NW_TVB_Simulator; each result is stored under this monitor's label."
        ),
        parameters={
            "monitor_type": ParameterDefinition(
                default_value="TemporalAverage",
                description=(
                    "What is recorded. 'Raw' - every integration step (huge output; short runs "
                    "only). 'SubSample' - every 'period' ms, without averaging. "
                    "'TemporalAverage' - average over each 'period' window (e.g. 1 ms -> 1 kHz "
                    "LFP-like signal; standard). 'GlobalAverage' - mean over all regions (one "
                    "trace). 'Bold' - simulated fMRI BOLD via a haemodynamic response "
                    "(period = TR, e.g. 2000 ms; needs >= 20-60 s of simulation). "
                    "'AfferentCouplingTemporalAverage' - the network input each region "
                    "receives, time-averaged. Example: 'Bold'."
                ),
                constraints={
                    "allowed_values": [
                        "Raw",
                        "SubSample",
                        "TemporalAverage",
                        "GlobalAverage",
                        "Bold",
                        "AfferentCouplingTemporalAverage",
                    ]
                },
            ),
            "period": ParameterDefinition(
                default_value=1.0,
                description=(
                    "Sampling period in ms (1000/period = sampling rate in Hz). Examples: 1.0 "
                    "for neural activity at 1 kHz; 10.0 for 100 Hz; 720 or 2000 for BOLD "
                    "(fMRI TR of 0.72 s or 2 s). Ignored by 'Raw' (records every dt)."
                ),
                constraints={"min": 0.001, "max": 100000.0},
                unit="ms",
            ),
            "variables": ParameterDefinition(
                default_value=[],
                description=(
                    "Which of the model's recorded variables (NW_TVB_Model "
                    "variables_of_interest) this monitor keeps, by name or index. Examples: "
                    "['V'] to keep only the membrane potential of MontbrioPazoRoxin; ['S_e'] "
                    "for ReducedWongWangExcInh BOLD; [0]. Empty [] = all of them."
                ),
            ),
            "hrf_kernel": ParameterDefinition(
                default_value="FirstOrderVolterra",
                description=(
                    "Haemodynamic response function for 'Bold' only. 'FirstOrderVolterra' - "
                    "Friston 2000 balloon-model approximation (TVB default). 'Gamma' - single "
                    "gamma function. 'DoubleExponential' - sum of two damped oscillations. "
                    "'MixtureOfGammas' - SPM-like canonical HRF with undershoot."
                ),
                constraints={
                    "allowed_values": [
                        "FirstOrderVolterra",
                        "Gamma",
                        "DoubleExponential",
                        "MixtureOfGammas",
                    ]
                },
            ),
            "label": ParameterDefinition(
                default_value="",
                description=(
                    "Name of this recording in the simulator results; downstream nodes select "
                    "it by this name. Must be unique per simulator. Examples: 'tavg', 'bold', "
                    "'eeg_like'. Empty = monitor_type in lower case (e.g. 'temporalaverage')."
                ),
            ),
        },
        inputs={},
        outputs={
            "tvb_monitor": PortDefinition(
                type=PortType.DICT,
                description=(
                    "{'label', 'monitor' (TVB monitor object), 'monitor_type', 'variables'}; "
                    "connect to the tvb_monitors fan-in port of NW_TVB_Simulator."
                ),
            ),
        },
        methods={
            "build_monitor": MethodDefinition(
                description="Instantiate the selected TVB monitor with its period and HRF kernel.",
                inputs=[],
                outputs=["tvb_monitor"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build_monitor", self.build_monitor, method_key="build_monitor")

    def build_monitor(self) -> Dict[str, Any]:
        monitor_type = self._parameters["monitor_type"]
        period = float(self._parameters["period"])

        cls = getattr(monitors, monitor_type)
        if monitor_type == "Raw":
            mon = cls()
        elif monitor_type == "Bold":
            hrf = getattr(equations, self._parameters["hrf_kernel"])()
            mon = cls(period=period, hrf_kernel=hrf)
        else:
            mon = cls(period=period)

        label = self._parameters["label"] or monitor_type.lower()
        print(f"[{self.name}] {monitor_type} monitor '{label}'"
              + ("" if monitor_type == "Raw" else f", period={period} ms"))
        return {
            "tvb_monitor": {
                "label": label,
                "monitor": mon,
                "monitor_type": monitor_type,
                # Resolved to indices by NW_TVB_Simulator, which knows the model.
                "variables": list(self._parameters["variables"] or []),
            }
        }
