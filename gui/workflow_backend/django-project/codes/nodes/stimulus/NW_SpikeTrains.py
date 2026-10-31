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

    With nwb_file set, the trains are recorded spikes instead: BMTK's ecephys_probe
    module reads the NWB units table and plays one recorded neuron into each virtual
    cell. The node only creates the cells and writes that module's config entry.
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
            "real population through synapses, the way NEST's poisson_generator does. "
            "With nwb_file set, the trains are instead recorded spike times read from "
            "the units table of an NWB file, one recorded neuron per virtual cell."
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
                    "many targets. With nwb_file set it is used by the two 'sample' "
                    "mappings only; 'units_map' makes one cell per recorded neuron."
                ),
                constraints={"min": 1},
            ),
            "distribution": ParameterDefinition(
                default_value="poisson",
                description=(
                    "Spike time statistics. 'poisson' is memoryless: intervals are "
                    "exponential and the coefficient of variation is 1. 'gamma' is a "
                    "renewal process whose regularity is set by gamma_shape. Not used "
                    "when nwb_file is set."
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
                    "compares constraints with '<', which a list cannot answer. Not "
                    "used when nwb_file is set."
                ),
                optimizable=True,
                optimization_range=[0.0, 100.0],
            ),
            "gamma_shape": ParameterDefinition(
                default_value=1.0,
                description=(
                    "Shape parameter of the gamma process, used when distribution is "
                    "'gamma'. 1.0 reproduces Poisson. Above 1 the firing becomes more "
                    "regular (clock-like); below 1 it becomes burstier than Poisson. "
                    "Not used when nwb_file is set."
                ),
                constraints={"min": 0.01},
            ),
            "start_ms": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Time the spike trains begin, in milliseconds of model time. With "
                    "nwb_file set, the model time at which the piece cut from the "
                    "recording starts playing (where in the recording it is cut from is "
                    "nwb_start_ms)."
                ),
                constraints={"min": 0.0},
            ),
            "stop_ms": ParameterDefinition(
                default_value=1000.0,
                description=(
                    "Time the spike trains end, in milliseconds. Set this to at least "
                    "the simulation's tstop_ms, otherwise the input stops early and the "
                    "network falls silent for the remainder of the run. With nwb_file "
                    "set, stop_ms - start_ms is also the length of the piece cut from "
                    "the recording."
                ),
                constraints={"min": 0.0},
            ),
            "abs_refractory_ms": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Absolute refractory period in milliseconds: no two spikes in one "
                    "train are closer than this. 0 leaves the process untouched. "
                    "Poisson only; not used when nwb_file is set."
                ),
                constraints={"min": 0.0},
            ),
            "tau_refractory_ms": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Relative refractory time constant in milliseconds: the rate "
                    "recovers exponentially with this constant after each spike. "
                    "0 disables it. Poisson only; not used when nwb_file is set."
                ),
                constraints={"min": 0.0},
            ),
            "seed": ParameterDefinition(
                default_value=0,
                description=(
                    "Random seed, so a run reproduces. Change it to draw a different "
                    "realisation of the same statistics. Not used when nwb_file is "
                    "set: BMTK draws the 'sample' mappings without a seed."
                ),
            ),
            "nwb_file": ParameterDefinition(
                default_value="",
                description=(
                    "Path to an NWB file of recorded spikes. Leave it empty to generate "
                    "random trains as above. When set, the spike times of the recorded "
                    "neurons in the file's units table (spike_times, in seconds) are "
                    "played into the virtual cells by BMTK's ecephys_probe module, and "
                    "the random-train parameters are not used. BMTK reads the units "
                    "table joined to the electrodes table through a 'peak_channel_id' "
                    "column, as in the Allen Brain Observatory Neuropixels sessions; a "
                    "file without that column or without an electrodes table fails "
                    "inside BMTK. Needs the pynwb package."
                ),
            ),
            "nwb_mapping": ParameterDefinition(
                default_value="units_map",
                description=(
                    "How the recorded neurons in nwb_file are given to the virtual "
                    "cells. Used only when nwb_file is set. The candidates are the "
                    "neurons nwb_units selects (all of them when it is empty): choose "
                    "the brain area with nwb_units, then how to use it here.\n"
                    "'units_map' (default): one virtual cell per selected neuron, in "
                    "file order (cell 0 gets the first neuron, and so on). n_trains is "
                    "ignored; the number of cells is the number of selected neurons. "
                    "Every run gives the same result.\n"
                    "'sample': n_trains cells, each given a different selected neuron "
                    "picked at random. n_trains must not exceed the number of selected "
                    "neurons. Which cell gets which neuron changes from run to run "
                    "(BMTK draws it without a seed).\n"
                    "'sample_with_replacement': n_trains cells, each given a random "
                    "selected neuron, reused as needed - for more inputs than "
                    "recordings. Cells sharing a neuron fire at identical times. "
                    "Changes from run to run."
                ),
                constraints={
                    "allowed_values": ["units_map", "sample", "sample_with_replacement"]
                },
            ),
            "nwb_start_ms": ParameterDefinition(
                default_value=0.0,
                description=(
                    "Where in the recording the piece is cut from, in milliseconds of "
                    "recording time (the NWB file's own clock; 0 is the session start). "
                    "Used only when nwb_file is set. The piece runs from nwb_start_ms "
                    "for stop_ms - start_ms and plays in the model from start_ms, so a "
                    "few seconds from anywhere in a recording hours long fit a short "
                    "simulation. Example: nwb_start_ms=1800000, start_ms=0, "
                    "stop_ms=3000 plays minute 30 of the recording, 3 s of it, from "
                    "model time 0. Recorded spikes are times in seconds with sub-"
                    "millisecond precision (Neuropixels samples at 30 kHz); NEST "
                    "delivers each at the nearest simulation step (dt_ms). Recordings "
                    "rarely start at 0 - the Allen session 715093703 has no spike "
                    "before 11.9 s - so check that the piece holds spikes: the node "
                    "prints how many it found."
                ),
                constraints={"min": 0.0},
            ),
            "nwb_units": ParameterDefinition(
                default_value={},
                description=(
                    "Which recorded neurons of nwb_file to use, as a filter on their "
                    "properties; empty uses every neuron in the file. Used only when "
                    "nwb_file is set. Keys are columns of the NWB units table or of the "
                    "electrodes table it is joined to. The usual one is 'location', the "
                    "brain area of the neuron's recording electrode (Allen acronyms such "
                    "as 'VISp', 'VISl', 'CA1', 'LGd'). Values:\n"
                    "  a string - equal to it: {'location': 'VISl'}\n"
                    "  a list - any of them: {'location': ['VISp', 'VISl']}\n"
                    "  a comparison - {'firing_rate': {'operation': '>', 'value': 1.0}}\n"
                    "Several keys must all hold. The filter is applied by BMTK's own "
                    "filter_table, before nwb_mapping gives the selected neurons to the "
                    "cells; a key that is not a column stops the run with BMTK's "
                    "message naming it."
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
                    "SONATA spikes file - or, with nwb_file set, the NWB file - as a "
                    "simulation input. Consumed by "
                    "NW_Connectivity, which connects it to a real population, or by "
                    "NW_SimConfig directly."
                ),
            ),
        },
        methods={
            "build": MethodDefinition(
                description=(
                    "Create the virtual cells, generate their spike trains, write the "
                    "SONATA spikes file, and declare it as a simulation input. With "
                    "nwb_file set, declare the NWB file as the input instead."
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

        if str(p["nwb_file"]).strip():
            n_trains, sim_input = self._nwb_input(
                pop_name, n_trains, start_ms, stop_ms, inputs_dir
            )
        else:
            sim_input = self._generated_input(pop_name, n_trains, start_ms, stop_ms, inputs_dir)

        # model_type='virtual' is the only thing marking these cells as spike sources;
        # they carry no model_template and no dynamics_params because nothing about
        # them is simulated.
        net = NetworkBuilder(pop_name)
        net.add_nodes(N=n_trains, model_type="virtual", ei_type="exc")

        # BMTK registers every population name as a node set of its own
        # (simulator_network.py:101), so the bare name "drive" and this dict are the
        # same selection. The dict is written because it says so without depending on
        # that.
        sim_input["node_set"] = {"population": pop_name}

        return {
            "population": {
                "builder":     net,
                "pop_name":    pop_name,
                "network_dir": network_dir,
                # These cells carry no membrane potential. NW_SimConfig reads this
                # to keep them out of membrane reports written over "all" cells,
                # which would otherwise fail inside the simulator.
                "_virtual":    True,
                # Only what ends up inside the SONATA network files. The trains live
                # in the spikes file, rewritten on every build, so changing the rate,
                # the distribution or the seed must not invalidate the network - the
                # same reason NW_Population leaves nest_params out of its signature.
                # Including them forces a rebuild per trial, which an optimization
                # study does hundreds of times into one directory.
                "_signature":  {"pop_name": pop_name, "n_trains": n_trains},
                "_sim_inputs": {f"{pop_name}_spikes": sim_input},
            }
        }

    def _generated_input(self, pop_name: str, n_trains: int, start_ms: float,
                         stop_ms: float, inputs_dir: str) -> Dict[str, Any]:
        """Draw the random trains, write them as a SONATA spikes file, and return the
        config entry that reads it."""
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

        return {"input_type": "spikes", "module": "sonata", "input_file": spikes_file}

    def _nwb_input(self, pop_name: str, n_trains: int, start_ms: float,
                   stop_ms: float, inputs_dir: str):
        """The number of virtual cells and the ecephys_probe config entry for nwb_file.

        BMTK does the work when the simulation starts: it reads the units table, keeps
        the neurons nwb_units selects, gives each virtual cell one of them, cuts the
        piece of recording and moves it to model time. The neurons are also read here,
        with BMTK's own reader and filter, so that the cell count and the spike-time
        check see exactly what the simulation will.
        """
        import numpy as np
        import pandas as pd
        from bmtk.simulator.core.modules.ecephys_module import MappingStrategy

        p = self._parameters
        nwb_file = os.path.abspath(str(p["nwb_file"]).strip())
        mapping = str(p["nwb_mapping"])
        filters = dict(p["nwb_units"] or {})
        if not os.path.isfile(nwb_file):
            raise ValueError(f"nwb_file not found: {nwb_file}")
        if mapping not in ("units_map", "sample", "sample_with_replacement"):
            raise ValueError(
                f"nwb_mapping must be 'units_map', 'sample' or "
                f"'sample_with_replacement', not {mapping!r}."
            )

        units = MappingStrategy(input_file=nwb_file, units=filters).units_table
        if units is None or len(units) == 0:
            raise ValueError(
                f"{nwb_file}: no recorded neurons"
                + (f" match nwb_units {filters}." if filters else " in the units table.")
            )
        n_units = len(units)

        # The piece [nwb_start_ms, nwb_start_ms + (stop_ms - start_ms)] of the
        # recording plays from start_ms in the model: BMTK subtracts the window start
        # and adds interval_offset. NEST refuses a spike at t<=0, which only a
        # recorded spike at the very start of the piece with start_ms 0 can produce.
        nwb_start_ms = float(p["nwb_start_ms"])
        nwb_stop_ms = nwb_start_ms + (stop_ms - start_ms)
        rec_start_s, rec_stop_s = nwb_start_ms / 1000.0, nwb_stop_ms / 1000.0
        in_piece = [
            t[(t >= rec_start_s) & (t <= rec_stop_s)]
            for t in (np.asarray(st, dtype=float) for st in units["spike_times"])
        ]
        n_spikes = sum(len(t) for t in in_piece)
        if any(len(t) and (t.min() - rec_start_s) * 1000.0 + start_ms <= 0.0 for t in in_piece):
            raise ValueError(
                f"{pop_name}: a recorded spike would play at model time t<=0, which "
                f"NEST rejects. Set start_ms above 0."
            )

        entry = {
            "input_type":      "spikes",
            "module":          "ecephys_probe",
            "input_file":      nwb_file,
            "mapping":         mapping,
            "interval":        [nwb_start_ms, nwb_stop_ms],
            "interval_offset": start_ms,
        }
        if mapping == "units_map":
            # BMTK takes a fixed mapping only as a file: one row per virtual cell,
            # with the id of the recorded neuron it replays. BMTK ignores the units
            # filter for this mapping, so the list holds only the neurons it selected.
            n_trains = n_units
            map_file = os.path.join(inputs_dir, f"{pop_name}_units_map.csv")
            pd.DataFrame({"node_ids": range(n_units), "unit_ids": units.index.values}).to_csv(
                map_file, sep=" ", index=False
            )
            entry["units"] = map_file
        else:
            if filters:
                entry["units"] = filters
            if mapping == "sample" and n_trains > n_units:
                raise ValueError(
                    f"{pop_name}: n_trains ({n_trains}) is more than the {n_units} "
                    f"selected neurons in {nwb_file}; 'sample' gives each cell a "
                    f"different one. Use 'sample_with_replacement' or lower n_trains."
                )

        print(f"[NW_SpikeTrains] {n_units} selected neurons, {n_spikes} spikes in "
              f"recording [{nwb_start_ms}, {nwb_stop_ms}] ms, played at model "
              f"[{start_ms}, {stop_ms}] ms -> {n_trains} virtual cells ({mapping})")
        return n_trains, entry
