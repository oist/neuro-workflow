import os
from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition,
)
from neuroworkflow.core.port import PortType


class NW_SpikeTrains(Node):
    """
    A population of virtual cells firing independent, randomly generated spike trains.

    This is BMTK's standard way of driving a network, and the closest equivalent to
    NEST's poisson_generator: a population of cells that produce spikes but simulate
    no membrane dynamics, connected to real neurons through ordinary synapses. Each
    virtual cell fires its own independent train, so N cells give N different
    realisations of the same statistics - not N copies of one train.

    Three pieces, all BMTK's own:
      NetworkBuilder(..., model_type='virtual')  → the cells
      PoissonSpikeGenerator / GammaSpikeGenerator → the spike times
      a config "inputs" entry                    → tells the simulator to use them

    The output is shaped exactly like NW_Population's, so NW_Connectivity wires this
    to a target population with normal weights and delays.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_spike_trains",
        stage="stimulus",
        tool="BMTK",
        model_source="https://alleninstitute.github.io/bmtk/tutorials/Ch_advanced_spikes_input.html",
        description=(
            "Creates a population of virtual cells whose spike trains are drawn from a "
            "Poisson or gamma renewal process, writes them as a SONATA spikes file, and "
            "declares them as a simulation input. Connect to NW_Connectivity to drive a "
            "real population through synapses, the way NEST's poisson_generator does."
        ),
        parameters={
            "pop_name": ParameterDefinition(
                default_value="inputs",
                description=(
                    "Name of this virtual population. Used as the SONATA population "
                    "name, the spikes file name, and the name NW_Connectivity refers "
                    "to as a connection source. Must differ from every real population."
                ),
            ),
            "n_trains": ParameterDefinition(
                default_value=10,
                description=(
                    "Number of virtual cells, each firing its own independent train. "
                    "Set this to the number of distinct inputs you want, not to the "
                    "size of the target population - one virtual cell can connect to "
                    "many targets."
                ),
                constraints={"min": 1},
            ),
            "distribution": ParameterDefinition(
                default_value="poisson",
                description=(
                    "Spike time statistics. 'poisson' is memoryless: intervals are "
                    "exponential and the coefficient of variation is 1. 'gamma' is a "
                    "renewal process whose regularity is set by gamma_shape."
                ),
                constraints={"allowed_values": ["poisson", "gamma"]},
            ),
            "firing_rate_hz": ParameterDefinition(
                default_value=10.0,
                description=(
                    "Firing rate of each virtual cell, in Hz. A single number holds "
                    "that rate for the whole window. A list makes the rate vary over "
                    "time - the values are spread evenly between start_ms and stop_ms, "
                    "so [2.0, 2.0, 50.0, 50.0] is low then high. Poisson only; the "
                    "gamma process uses the first value if given a list. Must not be "
                    "negative. No min/max constraint is declared because the framework "
                    "compares constraints with '<', which a list cannot answer."
                ),
                optimizable=True,
                optimization_range=[0.0, 100.0],
            ),
            "gamma_shape": ParameterDefinition(
                default_value=1.0,
                description=(
                    "Shape parameter of the gamma process, used when distribution is "
                    "'gamma'. 1.0 reproduces Poisson. Above 1 the firing becomes more "
                    "regular (clock-like); below 1 it becomes burstier than Poisson."
                ),
                constraints={"min": 0.01},
            ),
            "start_ms": ParameterDefinition(
                default_value=0.0,
                description="Time the spike trains begin, in milliseconds.",
                constraints={"min": 0.0},
            ),
            "stop_ms": ParameterDefinition(
                default_value=1000.0,
                description=(
                    "Time the spike trains end, in milliseconds. Set this to at least "
                    "the simulation's tstop_ms, otherwise the input stops early and the "
                    "network falls silent for the remainder of the run."
                ),
                constraints={"min": 0.0},
            ),
            "abs_refractory_ms": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Absolute refractory period in milliseconds: no two spikes in one "
                    "train are closer than this. 0 leaves the process untouched. "
                    "Poisson only."
                ),
                constraints={"min": 0.0},
            ),
            "tau_refractory_ms": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Relative refractory time constant in milliseconds: the rate "
                    "recovers exponentially with this constant after each spike. "
                    "0 disables it. Poisson only."
                ),
                constraints={"min": 0.0},
            ),
            "seed": ParameterDefinition(
                default_value=0,
                description=(
                    "Random seed, so a run reproduces. Change it to draw a different "
                    "realisation of the same statistics."
                ),
            ),
        },
        inputs={},
        outputs={
            "population": PortDefinition(
                type=PortType.OBJECT,
                description=(
                    "Dict with: builder (live NetworkBuilder holding the virtual cells), "
                    "pop_name (str), network_dir (str), and _sim_inputs carrying the "
                    "SONATA spikes file as a simulation input. Consumed by "
                    "NW_Connectivity, which connects it to a real population, or by "
                    "NW_SimConfig directly."
                ),
            ),
        },
        methods={
            "build": MethodDefinition(
                description=(
                    "Create the virtual cells, generate their spike trains, write the "
                    "SONATA spikes file, and declare it as a simulation input."
                ),
                inputs=[],
                outputs=["population"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("build", self.build, method_key="build")

    def _generate(self, node_ids, start_s: float, stop_s: float):
        """The spike trains, from BMTK's own generators."""
        import numpy as np
        from bmtk.utils.reports.spike_trains import (
            GammaSpikeGenerator, PoissonSpikeGenerator,
        )

        p = self._parameters
        pop_name = str(p["pop_name"])
        rate = p["firing_rate_hz"]
        seed = int(p["seed"])

        if str(p["distribution"]).lower() == "gamma":
            generator = GammaSpikeGenerator(population=pop_name, seed=seed)
            # The gamma process is defined for a stationary rate only.
            scalar_rate = float(rate[0] if isinstance(rate, (list, tuple)) else rate)
            generator.add(
                node_ids=node_ids,
                firing_rate=scalar_rate,
                a=float(p["gamma_shape"]),
                times=(start_s, stop_s),
            )
            return generator

        generator = PoissonSpikeGenerator(population=pop_name, seed=seed)
        if isinstance(rate, (list, tuple)):
            # BMTK wants a time for every rate value, not a start/stop pair, so the
            # values are spread evenly across the window the user asked for.
            rates = [float(r) for r in rate]
            times = np.linspace(start_s, stop_s, len(rates))
        else:
            rates = float(rate)
            times = (start_s, stop_s)

        generator.add(
            node_ids=node_ids,
            firing_rate=rates,
            times=times,
            abs_ref=float(p["abs_refractory_ms"]) / 1000.0,
            tau_ref=float(p["tau_refractory_ms"]) / 1000.0,
        )
        return generator

    def build(self) -> Dict[str, Any]:
        from bmtk.builder.networks import NetworkBuilder

        p = self._parameters
        base_dir    = self._context.get("results_path", "results")
        network_dir = os.path.join(base_dir, "network")
        inputs_dir  = os.path.join(base_dir, "inputs")
        os.makedirs(network_dir, exist_ok=True)
        os.makedirs(inputs_dir, exist_ok=True)

        pop_name = str(p["pop_name"])
        n_trains = int(p["n_trains"])
        start_ms = float(p["start_ms"])
        stop_ms  = float(p["stop_ms"])
        if stop_ms <= start_ms:
            raise ValueError(
                f"stop_ms ({stop_ms}) must be after start_ms ({start_ms}): "
                f"the spike trains would cover no time at all."
            )

        node_ids = list(range(n_trains))
        generator = self._generate(node_ids, start_ms / 1000.0, stop_ms / 1000.0)

        spikes = generator.to_dataframe()
        if len(spikes) and float(spikes["timestamps"].min()) <= 0.0:
            # NEST refuses a virtual cell whose spike time is zero or negative, and
            # the failure surfaces inside the simulator rather than here.
            raise ValueError(
                f"{pop_name}: a spike was generated at t<=0, which NEST rejects. "
                f"Set start_ms above 0."
            )

        spikes_file = os.path.join(inputs_dir, f"{pop_name}_spikes.h5")
        generator.to_sonata(spikes_file)
        print(f"[NW_SpikeTrains] {len(spikes)} spikes for {n_trains} virtual cells "
              f"-> {spikes_file}")

        # model_type='virtual' is the only thing marking these cells as spike sources;
        # they carry no model_template and no dynamics_params because nothing about
        # them is simulated.
        net = NetworkBuilder(pop_name)
        net.add_nodes(N=n_trains, model_type="virtual", ei_type="exc")

        return {
            "population": {
                "builder":     net,
                "pop_name":    pop_name,
                "network_dir": network_dir,
                # These cells carry no membrane potential. NW_SimConfig reads this
                # to keep them out of membrane reports written over "all" cells,
                # which would otherwise fail inside the simulator.
                "_virtual":    True,
                "_signature":  dict(p),
                "_sim_inputs": {
                    f"{pop_name}_spikes": {
                        "input_type": "spikes",
                        "module":     "sonata",
                        "input_file": spikes_file,
                        # BMTK registers every population name as a node set of its
                        # own (simulator_network.py:101), so the bare name "drive"
                        # and this dict are the same selection. The dict is written
                        # because it says so without depending on that.
                        "node_set":   {"population": pop_name},
                    }
                },
            }
        }
