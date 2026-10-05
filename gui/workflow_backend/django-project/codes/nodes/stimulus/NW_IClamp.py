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
                    "Path to a text file holding an arbitrary current waveform. When "
                    "set it replaces the step defined by amp/delay/duration; leave it "
                    "empty for the step. Two columns with a header row: a time in "
                    "milliseconds and the current at that time, in the same unit as "
                    "amp_na (pA for NEST, nA for NEURON). Default column names are "
                    "'timestamps' and 'amps', the default separator is a single SPACE "
                    "(use waveform_separator=',' for a comma file), and at least two "
                    "rows are required. Example:\n"
                    "    timestamps amps\n"
                    "    100.0 0.00\n"
                    "    110.0 2.15\n"
                    "    120.0 4.30\n"
                    "Each amplitude is HELD until the next timestamp - the waveform is "
                    "a staircase, not an interpolated curve, so the row spacing is the "
                    "resolution. The final amplitude is held to the end of the run: add "
                    "a last row with amplitude 0 to switch the current off. Rows at or "
                    "before the simulation timestep (dt_ms) are skipped by BMTK."
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
                description=(
                    "Column separator in waveform_csv. BMTK's default is a single "
                    "space; set ',' for a comma-separated file or '\\t' for tabs."
                ),
            ),
            "node_set": ParameterDefinition(
                default_value="",
                description=(
                    "Leave empty. The clamp drives the population it is connected to, "
                    "which the graph already states - every neuron of that population "
                    "and no other. Connect the clamp to a second population to drive "
                    "that one too."
                    "\n\nSet it only to reach part of a population rather than all of "
                    "it: a BMTK node set, such as a filter {'ei_type': 'exc'} or a list "
                    "of node ids. Whatever is set here is passed to BMTK untouched."
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

        # A clamp cannot know its own target: it is the graph edge into a
        # population that decides. The key is omitted when the user left it empty,
        # and NW_SimConfig fills it with the population that received the clamp.
        node_set = str(p["node_set"]).strip()
        target = {"node_set": node_set} if node_set else {}

        if not waveform:
            return {
                "iclamp": {
                    "amp":      float(p["amp_na"]),
                    "delay":    float(p["delay_ms"]),
                    "duration": float(p["duration_ms"]),
                    **target,
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
                **target,
                "file":              os.path.abspath(waveform),
                "timestamps_column": str(p["waveform_time_column"]),
                "amplitudes_column": str(p["waveform_amp_column"]),
                "separator":         str(p["waveform_separator"]),
            }
        }
