from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition,
)
from neuroworkflow.core.port import PortType


class NW_IClamp(Node):
    """
    Somatic current clamp stimulus for BMTK simulations.

    Packages amplitude, delay, and duration into the dict that
    create_environment() expects as its current_clamp argument.
    Connect the output to NW_Population's iclamp port.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_iclamp",
        stage="stimulus",
        tool="BMTK",
        model_source="https://alleninstitute.github.io/bmtk/",
        description=(
            "Defines a somatic current clamp stimulus. "
            "Output is a dict passed directly to BMTK's create_environment(current_clamp=...). "
            "Connect to NW_Population.iclamp."
        ),
        parameters={
            "amp_na": ParameterDefinition(
                default_value=376.0,
                description=(
                    "Injected current amplitude. BMTK passes this to the simulator "
                    "unscaled, so the unit follows the backend: picoamperes (pA) in "
                    "PointNet/NEST, nanoamperes (nA) in BioNet/NEURON. The default "
                    "376 pA is roughly the rheobase of a default NEST iaf_psc_alpha "
                    "- the smallest steady current that makes it fire."
                ),
                constraints={"min": -10000.0, "max": 10000.0},
            ),
            "delay_ms": ParameterDefinition(
                default_value=500.0,
                description="Onset delay in milliseconds before current injection begins.",
                constraints={"min": 0.0},
            ),
            "duration_ms": ParameterDefinition(
                default_value=2000.0,
                description="Duration of the current pulse in milliseconds.",
                constraints={"min": 0.0},
            ),
            "waveform_csv": ParameterDefinition(
                default_value="",
                description=(
                    "Path to a CSV of an arbitrary current waveform, with a column of "
                    "times in ms and a column of amplitudes. When set, this replaces "
                    "the step defined by amp/delay/duration and BMTK reads the waveform "
                    "instead. Leave empty for the step."
                ),
            ),
            "waveform_time_column": ParameterDefinition(
                default_value="timestamps",
                description="Name of the time column in waveform_csv.",
            ),
            "waveform_amp_column": ParameterDefinition(
                default_value="amps",
                description="Name of the amplitude column in waveform_csv.",
            ),
            "waveform_separator": ParameterDefinition(
                default_value=" ",
                description="Column separator in waveform_csv. Use ',' for comma-separated.",
            ),
            "node_set": ParameterDefinition(
                default_value="all",
                description=(
                    "Which cells receive the current: 'all', or a population name to "
                    "stimulate only that population."
                ),
            ),
        },
        inputs={},
        outputs={
            "iclamp": PortDefinition(
                type=PortType.DICT,
                description=(
                    "A step clamp as keys amp, delay, duration, node_set; or, when "
                    "waveform_csv is set, a ready BMTK config 'inputs' entry for the "
                    "waveform. Connect to NW_Population.iclamp either way."
                ),
            ),
        },
        methods={
            "build": MethodDefinition(
                description=(
                    "Package the stimulus into the dict NW_Population passes on: a "
                    "step by default, or a waveform entry when waveform_csv is set."
                ),
                inputs=[],
                outputs=["iclamp"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build", self.build, method_key="build")

    def build(self) -> Dict[str, Any]:
        import os

        p = self._parameters
        waveform = str(p["waveform_csv"]).strip()

        if not waveform:
            # The shape create_environment() takes, unchanged from every existing
            # workflow. node_set defaults to "all" there, so sending it is a no-op.
            return {
                "iclamp": {
                    "amp":      float(p["amp_na"]),
                    "delay":    float(p["delay_ms"]),
                    "duration": float(p["duration_ms"]),
                    "node_set": str(p["node_set"]),
                }
            }

        if not os.path.isfile(waveform):
            raise ValueError(f"waveform_csv not found: {waveform}")

        # A waveform reaches BMTK only through a config "inputs" entry.
        # create_environment() writes this case as module "FileIClamp", which
        # neither PointNet nor BioNet dispatches, so the stimulus would be
        # silently dropped. module "IClamp" with input_type "csv" is the
        # combination both simulators read.
        return {
            "iclamp": {
                "input_type":        "csv",
                "module":            "IClamp",
                "node_set":          str(p["node_set"]),
                "file":              os.path.abspath(waveform),
                "timestamps_column": str(p["waveform_time_column"]),
                "amplitudes_column": str(p["waveform_amp_column"]),
                "separator":         str(p["waveform_separator"]),
            }
        }
