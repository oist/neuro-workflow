from typing import Dict, Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition,
)
from neuroworkflow.core.port import PortType


class NW_Analysis(Node):
    """
    Generic BMTK analysis and visualization node — spike raster and membrane traces.

    Tutorial reference: Ch2 Single Cell —
        from bmtk.analyzer.spike_trains import plot_raster, to_dataframe
        from bmtk.analyzer.compartment import plot_traces
        _ = plot_raster(config_file='config.iclamp.json', with_histogram=False)
        _ = plot_traces(config_file='config.iclamp.json', report_name='v_report')

    Scaling path: same node works for single cell, multi-population, any simulator —
    the config JSON already knows which cells were recorded.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_analysis",
        stage="analysis",
        tool="BMTK",
        model_source="https://alleninstitute.github.io/bmtk/",
        description=(
            "Reads BMTK simulation output and produces spike raster plots and "
            "membrane potential traces using bmtk.analyzer. Works with any BMTK "
            "simulator (PointNet or BioNet) and any number of populations."
        ),
        parameters={
            "plot_raster": ParameterDefinition(
                default_value=True,
                description="Generate a spike raster plot from the output spikes file.",
            ),
            "with_histogram": ParameterDefinition(
                default_value=False,
                description="Add a firing-rate histogram panel below the raster plot.",
            ),
            "plot_traces": ParameterDefinition(
                default_value=True,
                description=(
                    "Generate membrane potential traces. "
                    "Requires a membrane_report entry in NW_SimConfig.reports."
                ),
            ),
            "report_name": ParameterDefinition(
                default_value="v_report",
                description=(
                    "Name of the compartment report to plot (as set in NW_SimConfig.reports). "
                    "Default 'v_report' matches the tutorial."
                ),
            ),
            "populations": ParameterDefinition(
                default_value=[],
                description=(
                    "Population names to analyze. Empty list = auto-detect all populations "
                    "from the spikes file and plot each one. "
                    "Example: ['popA', 'popB'] to restrict to specific populations."
                ),
            ),
            "trace_node_ids": ParameterDefinition(
                default_value=[],
                description=(
                    "Node IDs to plot as individual traces. "
                    "Empty list = all recorded neurons. "
                    "Example: [0, 1, 2] plots three specific neurons."
                ),
            ),
            "save_figures": ParameterDefinition(
                default_value=True,
                description="Save figure PNG files to results_path instead of only displaying.",
            ),
        },
        inputs={
            "results": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict from NW_SimConfig with: config_file (str), "
                    "output_dir (str), simulator (str)."
                ),
            ),
        },
        outputs={
            "figures": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict with keys 'raster' and 'traces', each containing "
                    "the path to the saved PNG file (or None if disabled)."
                ),
            ),
            "firing_rate_hz": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Mean firing rate per population in Hz, keyed by population name: "
                    "total spikes / (population size x simulation duration). A population "
                    "of size 1 reports that single neuron's rate. Populations that "
                    "produced no spikes appear with a rate of 0.0 rather than being "
                    "omitted. Empty dict if the spikes file cannot be read."
                ),
            ),

            "isi_stats": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Inter-spike interval distribution per population, keyed by population "
                    "name: {'mean_ms', 'std_ms', 'cv', 'n_intervals'}. Intervals are taken "
                    "per neuron and then pooled - never across neurons, which would be "
                    "meaningless. 'cv' is std/mean, the classic irregularity measure: near 0 "
                    "is clock-like firing, near 1 is Poisson-like. A population that never "
                    "spiked appears with zeros rather than being omitted."
                ),
            ),
        },
        methods={
            "analyze": MethodDefinition(
                description=(
                    "Load simulation results, measure per-population firing rates, and "
                    "produce spike raster and/or membrane potential trace plots."
                ),
                inputs=["results"],
                outputs=["figures", "firing_rate_hz", "isi_stats"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("analyze", self.analyze, method_key="analyze")

    def _detect_populations(self, output_dir: str, report_name: str) -> list:
        """Detect population names from spikes.h5, falling back to the membrane report."""
        import os
        import h5py

        spikes_path = os.path.join(output_dir, "spikes.h5")
        if os.path.exists(spikes_path):
            with h5py.File(spikes_path, "r") as f:
                pops = list(f.get("spikes", {}).keys())
            if pops:
                return pops

        # No spikes (or silent network) — read populations from the membrane report
        report_path = os.path.join(output_dir, f"{report_name}.h5")
        if os.path.exists(report_path):
            with h5py.File(report_path, "r") as f:
                return list(f.get("report", {}).keys())

        return []

    def _population_sizes(self) -> Dict[str, int]:
        """Neurons per population, from the SONATA node files the network was built from.

        Used as the denominator for firing rate, and to decide whether a spike train
        can be split per neuron when the spikes file carries no node_ids.
        """
        import glob
        import os

        import h5py

        network_dir = os.path.join(self._context.get("results_path", "results"), "network")
        sizes: Dict[str, int] = {}
        for nodes_file in sorted(glob.glob(os.path.join(network_dir, "*_nodes.h5"))):
            with h5py.File(nodes_file, "r") as f:
                for pop, group in f.get("nodes", {}).items():
                    if "node_id" in group:
                        sizes[pop] = len(group["node_id"])
        return sizes

    def _measure_isi_stats(self, output_dir: str) -> Dict[str, Dict[str, float]]:
        """Summarise each population's inter-spike interval distribution.

        Intervals are computed per neuron and then pooled, so a population of one and a
        population of many are described the same way. Pooling the raw timestamps
        instead would produce intervals *between different neurons*, which mean nothing.
        """
        try:
            import os

            import h5py
            import numpy as np

            spikes_path = os.path.join(output_dir, "spikes.h5")
            if not os.path.exists(spikes_path):
                return {}

            sizes = self._population_sizes()
            empty = {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0, "n_intervals": 0}
            stats: Dict[str, Dict[str, float]] = {pop: dict(empty) for pop in sizes}

            with h5py.File(spikes_path, "r") as f:
                for pop, group in f.get("spikes", {}).items():
                    if "timestamps" not in group:
                        continue
                    times = np.asarray(group["timestamps"], dtype=float)
                    ids = (np.asarray(group["node_ids"])
                           if "node_ids" in group else None)

                    per_neuron = []
                    if ids is not None and len(ids) == len(times):
                        for neuron in np.unique(ids):
                            train = np.sort(times[ids == neuron])
                            if train.size > 1:
                                per_neuron.append(np.diff(train))
                    elif sizes.get(pop, 0) == 1:
                        # One neuron: every timestamp is its own, so no grouping needed.
                        train = np.sort(times)
                        if train.size > 1:
                            per_neuron.append(np.diff(train))
                    else:
                        print(f"[NW_Analysis] isi stats skipped for {pop!r}: "
                              f"spikes.h5 has no node_ids to separate neurons")
                        continue

                    pooled = np.concatenate(per_neuron) if per_neuron else np.array([])
                    if pooled.size == 0:
                        continue
                    mean = float(pooled.mean())
                    std = float(pooled.std())
                    stats[pop] = {
                        "mean_ms": mean,
                        "std_ms": std,
                        "cv": std / mean if mean > 0 else 0.0,
                        "n_intervals": int(pooled.size),
                    }

            return stats

        except Exception as e:
            print(f"[NW_Analysis] isi statistics skipped: {e}")
            return {}

    def _measure_firing_rates(self, config_file: str, output_dir: str) -> Dict[str, float]:
        """Mean firing rate per population, in Hz, read from the SONATA output.

        The denominator is the population size from the network files, not the
        number of neurons that happened to spike: counting only spiking neurons
        would make a mostly-silent network report a healthy rate.

        The network directory comes from the context's ``results_path`` — the same
        value NW_SimConfig used to build it — and the spikes file from the
        ``output_dir`` that node reports. The config is read only for the run
        duration, since its paths are still unexpanded $VAR manifest entries.
        """
        try:
            import glob
            import json
            import os

            import h5py

            with open(config_file) as fh:
                run = json.load(fh).get("run", {})
            duration_s = (float(run["tstop"]) - float(run.get("tstart", 0.0))) / 1000.0
            if duration_s <= 0:
                print(f"[NW_Analysis] firing rate skipped: run duration is {duration_s}s")
                return {}

            # Same run root NW_SimConfig used to build the network, and the value
            # a re-run updates when it points the workflow at a new results dir.
            network_dir = os.path.join(
                self._context.get("results_path", "results"), "network")
            sizes = self._population_sizes()

            spikes_path = os.path.join(output_dir, "spikes.h5")
            if not sizes:
                print(f"[NW_Analysis] firing rate skipped: no node files in {network_dir}")
                return {}
            if not os.path.exists(spikes_path):
                print(f"[NW_Analysis] firing rate skipped: no spikes file at {spikes_path}")
                return {}

            counts: Dict[str, int] = {}
            with h5py.File(spikes_path, "r") as f:
                for pop, group in f.get("spikes", {}).items():
                    if "timestamps" in group:
                        counts[pop] = len(group["timestamps"])

            # A silent population stays in the dict at 0.0: a missing key would turn
            # a meaningful result into an unresolvable measurement.
            return {
                pop: counts.get(pop, 0) / (n * duration_s)
                for pop, n in sizes.items() if n > 0
            }

        except Exception as e:
            print(f"[NW_Analysis] firing rate measurement skipped: {e}")
            return {}

    def analyze(self, results: Dict) -> Dict[str, Any]:
        import os
        import matplotlib.pyplot as plt

        config_file = results["config_file"]
        output_dir  = results["output_dir"]
        p = self._parameters

        # Measured before plotting, so it does not depend on the plotting flags.
        firing_rate_hz = self._measure_firing_rates(config_file, output_dir)
        isi_stats = self._measure_isi_stats(output_dir)

        populations = list(p["populations"]) or self._detect_populations(output_dir, str(p["report_name"]))
        node_ids    = list(p["trace_node_ids"]) or None
        figures: Dict[str, Any] = {"raster": None, "traces": None}

        if bool(p["plot_raster"]):
            try:
                from bmtk.analyzer.spike_trains import plot_raster
                raster_paths = []
                for pop in populations:
                    plot_raster(
                        config_file    = config_file,
                        population     = pop,
                        with_histogram = bool(p["with_histogram"]),
                        show           = False,
                    )
                    plt.title(pop)
                    if bool(p["save_figures"]):
                        path = os.path.join(output_dir, f"raster_{pop}.png")
                        plt.savefig(path, bbox_inches="tight")
                        raster_paths.append(path)
                    plt.show()
                figures["raster"] = raster_paths or None
            except Exception as e:
                print(f"[NW_Analysis] plot_raster skipped: {e}")

        if bool(p["plot_traces"]):
            try:
                from bmtk.analyzer.compartment import plot_traces
                trace_paths = []
                for pop in populations:
                    plot_traces(
                        config_file = config_file,
                        report_name = str(p["report_name"]),
                        population  = pop,
                        node_ids    = node_ids,
                        show        = False,
                    )
                    plt.title(pop)
                    if bool(p["save_figures"]):
                        path = os.path.join(output_dir, f"traces_{pop}.png")
                        plt.savefig(path, bbox_inches="tight")
                        trace_paths.append(path)
                    plt.show()
                figures["traces"] = trace_paths or None
            except Exception as e:
                print(f"[NW_Analysis] plot_traces skipped: {e}")

        return {"figures": figures, "firing_rate_hz": firing_rate_hz,
                "isi_stats": isi_stats}
