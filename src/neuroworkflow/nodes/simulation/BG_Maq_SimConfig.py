from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


class BG_Maq_SimConfig(Node):
    """Simulation and NEST-kernel settings for the macaque topological BG model.

    Node version of top_BG_nest3/simParams.py (resting-state relevant entries).
    Every value is a node parameter; nothing is read from the source scripts.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bg_maq_sim_config",
        stage="setup",
        tool="NEST",
        model_source="Girard et al. topological macaque BG model — top_BG_nest3/simParams.py",
        description=(
            "Simulation timing, NEST kernel, spatial scaling and recording settings for the "
            "macaque Basal Ganglia model (BG_Maq). Defaults reproduce simParams.py."
        ),
        parameters={
            "sim_duration_ms": ParameterDefinition(
                default_value=3000.0,
                description=(
                    "simDuration. End of the analysis window in ms. The network is simulated for "
                    "sim_duration_ms + initial_ignore_ms."
                ),
                constraints={"min": 100.0},
                unit="ms",
            ),
            "warmup_ms": ParameterDefinition(
                default_value=1000.0,
                description=(
                    "start_time_sp. Spike recorders start at this time; firing rates are computed "
                    "over [warmup_ms, sim_duration_ms]. Must be smaller than sim_duration_ms."
                ),
                constraints={"min": 0.0},
                unit="ms",
            ),
            "initial_ignore_ms": ParameterDefinition(
                default_value=0.0,
                description="initial_ignore. Extra simulated time appended to sim_duration_ms.",
                constraints={"min": 0.0},
                unit="ms",
            ),
            "dt_ms": ParameterDefinition(
                default_value=0.1,
                description="NEST resolution (simulation time step).",
                constraints={"min": 0.01, "max": 1.0},
                unit="ms",
            ),
            "n_threads": ParameterDefinition(
                default_value=10,
                description="nbcpu. Number of NEST threads (local_num_threads).",
                constraints={"min": 1, "max": 128},
            ),
            "rng_seed": ParameterDefinition(
                default_value=42,
                description=(
                    "msd. NEST rng_seed and seed of the position generators. The source draws it at "
                    "random (random.randint(0, 1000)); set -1 to reproduce that behaviour."
                ),
                constraints={"min": -1},
            ),
            "scalefactor_x": ParameterDefinition(
                default_value=1.0,
                description=(
                    "scalefactor[0]. Surface scaling of every layer along x: neuron counts are "
                    "multiplied by scalefactor_x * scalefactor_y and the layer extent grows with it "
                    "(constant density). 1.0 = published model."
                ),
                constraints={"min": 0.1, "max": 10.0},
            ),
            "scalefactor_y": ParameterDefinition(
                default_value=1.0,
                description="scalefactor[1]. Surface scaling of every layer along y (see scalefactor_x).",
                constraints={"min": 0.1, "max": 10.0},
            ),
            "density_scale": ParameterDefinition(
                default_value=1.0,
                description=(
                    "NOT in the original model — added for quick exploration. Multiplies every "
                    "population size without changing the layer geometry or the in-degrees "
                    "(e.g. 0.1 builds a 10x smaller network for fast tests). Keep 1.0 for the published model."
                ),
                constraints={"min": 0.01, "max": 1.0},
            ),
            "channels": ParameterDefinition(
                default_value=True,
                description=(
                    "Compute the hexagonal channel centres (saved to centers.txt). Has no effect on "
                    "resting-state dynamics; kept for action-selection compatibility."
                ),
            ),
            "channels_nb": ParameterDefinition(
                default_value=6,
                description="Number of action channels placed on a hexagon.",
                constraints={"min": 1, "max": 6},
            ),
            "hex_radius": ParameterDefinition(
                default_value=0.24,
                description="Radius of the hexagon on which channel centres are placed (layer units).",
                constraints={"min": 0.0},
            ),
            "channels_radius": ParameterDefinition(
                default_value=0.12,
                description="Radius of one channel column (layer units). Used by action-selection protocols only.",
                constraints={"min": 0.0},
            ),
            "record_to": ParameterDefinition(
                default_value="memory",
                description=(
                    "Spike recorder backend. 'memory' keeps spikes in RAM (fast); 'ascii' writes one "
                    "file per nucleus into output_dir like the original script."
                ),
                constraints={"allowed_values": ["memory", "ascii"]},
            ),
            "output_dir": ParameterDefinition(
                default_value="results/BG_Maq",
                description=(
                    "Folder for all outputs (positions, spike files, mean_fr.json, At.json, figures). "
                    "Relative paths are resolved from the working directory of the run."
                ),
            ),
        },
        inputs={},
        outputs={
            "sim_config": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Simulation settings dict using the original simParams keys (simDuration, "
                    "start_time_sp, initial_ignore, dt, nbcpu, msd, scalefactor, channels, ...) "
                    "plus density_scale, record_to and data_path."
                ),
            ),
        },
        methods={
            "build_config": MethodDefinition(
                description="Validate the timing parameters and assemble the sim_config dict.",
                inputs=[],
                outputs=["sim_config"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build_config", self.build_config, method_key="build_config")

    def build_config(self) -> Dict[str, Any]:
        p = self._parameters
        if p["warmup_ms"] >= p["sim_duration_ms"]:
            raise ValueError(
                f"warmup_ms ({p['warmup_ms']}) must be smaller than sim_duration_ms ({p['sim_duration_ms']})"
            )
        seed = int(p["rng_seed"])
        if seed < 0:
            import random
            seed = random.randint(0, 1000)

        sim_config = {
            "simDuration": float(p["sim_duration_ms"]),
            "start_time_sp": float(p["warmup_ms"]),
            "initial_ignore": float(p["initial_ignore_ms"]),
            "dt": float(p["dt_ms"]),
            "nbcpu": int(p["n_threads"]),
            "msd": seed,
            "scalefactor": [float(p["scalefactor_x"]), float(p["scalefactor_y"])],
            "density_scale": float(p["density_scale"]),
            "channels": bool(p["channels"]),
            "channels_nb": int(p["channels_nb"]),
            "hex_radius": float(p["hex_radius"]),
            "channels_radius": float(p["channels_radius"]),
            "record_to": p["record_to"],
            "data_path": p["output_dir"],
            "overwrite_files": True,
        }
        print(f"[{self.name}] {sim_config['simDuration']} ms, warmup {sim_config['start_time_sp']} ms, "
              f"dt {sim_config['dt']} ms, {sim_config['nbcpu']} threads, seed {seed}")
        return {"sim_config": sim_config}
