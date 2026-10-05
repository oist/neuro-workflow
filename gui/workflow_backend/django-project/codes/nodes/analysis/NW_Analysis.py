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
            "plot_rate": ParameterDefinition(
                default_value=True,
                description=(
                    "Generate a population firing rate over time plot, one figure per "
                    "population, separate from the raster."
                ),
            ),
            "rate_bin_ms": ParameterDefinition(
                default_value=10.0,
                description=(
                    "Bin width in milliseconds for the firing rate over time. A single "
                    "recorded neuron holds at most one spike per small bin, so bins of "
                    "50-100 ms read better there; 10 ms suits a population."
                ),
                constraints={"min": 0.1, "max": 10000.0},
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
                    "Dict with keys 'raster', 'traces' and 'rate', each containing "
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
            "rate_over_time": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Population firing rate as a function of time, keyed by population "
                    "name: {'t_ms': bin centre times, 'rate_hz': rate in each bin, "
                    "'bin_ms': bin width}. Each rate is spikes in the bin divided by "
                    "(population size x bin width in seconds), so it carries the same "
                    "units as firing_rate_hz and averaging it over the run reproduces "
                    "that value. A population that never spiked reports zeros rather "
                    "than being omitted."
                ),
            ),
        },
        methods={
            "analyze": MethodDefinition(
                description=(
                    "Load simulation results, measure per-population firing rates over "
                    "the whole run and over time, and produce spike raster, membrane "
                    "potential trace and firing rate plots."
                ),
                inputs=["results"],
                outputs=["figures", "firing_rate_hz", "isi_stats", "rate_over_time"],
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

    def _spike_trains(self, output_dir: str, config_file: str = "") -> Dict[str, Dict[str, Any]]:
        """Spike trains per population, from the simulation output and from the inputs.

        A population of virtual cells never appears in the output spikes file: BMTK
        connects its spike recorder to the real cells only, and virtual cells live in
        a separate id pool. Their spikes are real all the same - they are what drives
        the network - and the file holding them is already named in the config's
        ``inputs`` section. Reading both means a driving population is measured like
        any other instead of being reported as silent.

        Without a config, or with a config that declares no spike inputs, this returns
        exactly what the output file alone contains.
        """
        import json
        import os

        import h5py
        import numpy as np

        paths = []
        out_path = os.path.join(output_dir, "spikes.h5")
        if os.path.exists(out_path):
            paths.append(out_path)

        config_inputs = {}
        if config_file:
            try:
                with open(config_file) as fh:
                    config_inputs = json.load(fh).get("inputs", {}) or {}
            except Exception as e:
                print(f"[NW_Analysis] input spikes not read from {config_file}: {e}")

        for entry in config_inputs.values():
            # Only SONATA spike files are readable here. A current clamp, or spikes
            # held in a csv, carries no HDF5 groups and is skipped rather than guessed.
            if not isinstance(entry, dict) or entry.get("input_type") != "spikes":
                continue
            if entry.get("module") not in ("sonata", "h5", "hdf5"):
                continue
            path = entry.get("input_file")
            if isinstance(path, str) and os.path.exists(path) and path not in paths:
                paths.append(path)

        trains: Dict[str, Dict[str, Any]] = {}
        for path in paths:
            with h5py.File(path, "r") as f:
                for pop, group in f.get("spikes", {}).items():
                    if "timestamps" not in group:
                        continue
                    trains[pop] = {
                        "times": np.asarray(group["timestamps"], dtype=float),
                        "ids": (np.asarray(group["node_ids"])
                                if "node_ids" in group else None),
                    }
        return trains

    def _measure_isi_stats(self, output_dir: str,
                           config_file: str = "") -> Dict[str, Dict[str, float]]:
        """Summarise each population's inter-spike interval distribution.

        Intervals are computed per neuron and then pooled, so a population of one and a
        population of many are described the same way. Pooling the raw timestamps
        instead would produce intervals *between different neurons*, which mean nothing.
        """
        try:
            import os

            import h5py
            import numpy as np

            trains = self._spike_trains(output_dir, config_file)
            if not os.path.exists(os.path.join(output_dir, "spikes.h5")) and not trains:
                return {}

            sizes = self._population_sizes()
            empty = {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0, "n_intervals": 0}
            stats: Dict[str, Dict[str, float]] = {pop: dict(empty) for pop in sizes}

            for pop, train_data in trains.items():
                times = train_data["times"]
                ids = train_data["ids"]

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
                          f"the spikes file has no node_ids to separate neurons")
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

            if not sizes:
                print(f"[NW_Analysis] firing rate skipped: no node files in {network_dir}")
                return {}
            trains = self._spike_trains(output_dir, config_file)
            # A file that exists but holds no population means a network that stayed
            # silent, and every population is measured at 0 Hz below. Only a missing
            # file is unmeasurable.
            if not os.path.exists(os.path.join(output_dir, "spikes.h5")) and not trains:
                print(f"[NW_Analysis] firing rate skipped: no spikes file in {output_dir} "
                      f"and none declared as an input in {config_file}")
                return {}

            counts = {pop: int(train["times"].size) for pop, train in trains.items()}

            # A silent population stays in the dict at 0.0: a missing key would turn
            # a meaningful result into an unresolvable measurement.
            return {
                pop: counts.get(pop, 0) / (n * duration_s)
                for pop, n in sizes.items() if n > 0
            }

        except Exception as e:
            print(f"[NW_Analysis] firing rate measurement skipped: {e}")
            return {}

    def _measure_rate_over_time(
        self, config_file: str, output_dir: str, bin_ms: float
    ) -> Dict[str, Dict[str, Any]]:
        """Population firing rate per time bin, in Hz.

        Same quantity as ``firing_rate_hz`` but resolved in time: spikes falling in
        each bin divided by (population size x bin width in seconds). The denominator
        is the whole population, not the neurons that spiked in that bin, so the mean
        of this curve over the run equals the single ``firing_rate_hz`` value.
        """
        try:
            import json
            import os

            import h5py
            import numpy as np

            with open(config_file) as fh:
                run = json.load(fh).get("run", {})
            if "tstop" not in run:
                print(f"[NW_Analysis] rate over time skipped: {config_file} has no "
                      f"run.tstop to bound the bins")
                return {}
            tstart = float(run.get("tstart", 0.0))
            tstop = float(run["tstop"])
            if tstop <= tstart:
                print(f"[NW_Analysis] rate over time skipped: run window is "
                      f"{tstart}-{tstop} ms")
                return {}
            if bin_ms <= 0:
                print(f"[NW_Analysis] rate over time skipped: bin width is {bin_ms} ms")
                return {}

            sizes = self._population_sizes()
            if not sizes:
                print("[NW_Analysis] rate over time skipped: no node files to size "
                      "the populations")
                return {}
            trains = self._spike_trains(output_dir, config_file)
            # As in _measure_firing_rates: an empty file is a silent network, which
            # is measured as zeros, not as nothing.
            if not os.path.exists(os.path.join(output_dir, "spikes.h5")) and not trains:
                print(f"[NW_Analysis] rate over time skipped: no spikes file in "
                      f"{output_dir} and none declared as an input in {config_file}")
                return {}

            # A final short bin would report a rate from an incomplete window, so the
            # edges stop at the last whole bin inside the run.
            n_bins = int((tstop - tstart) // bin_ms)
            if n_bins < 1:
                print(f"[NW_Analysis] rate over time skipped: bin width {bin_ms} ms "
                      f"does not fit in a {tstop - tstart} ms run")
                return {}
            edges = tstart + np.arange(n_bins + 1) * bin_ms
            centres = (edges[:-1] + edges[1:]) / 2.0
            bin_s = bin_ms / 1000.0

            series: Dict[str, Dict[str, Any]] = {}
            for pop, n in sizes.items():
                if n <= 0:
                    continue
                train = trains.get(pop)
                times = train["times"] if train is not None else None
                if times is None or times.size == 0:
                    rates = np.zeros(n_bins)
                else:
                    counts, _ = np.histogram(times, bins=edges)
                    rates = counts / (n * bin_s)
                series[pop] = {
                    "t_ms": [float(t) for t in centres],
                    "rate_hz": [float(r) for r in rates],
                    "bin_ms": float(bin_ms),
                }
            return series

        except Exception as e:
            print(f"[NW_Analysis] rate over time skipped: {e}")
            return {}

    def _print_summary(
        self,
        firing_rate_hz: Dict[str, float],
        isi_stats: Dict[str, Dict[str, float]],
        rate_over_time: Dict[str, Dict[str, Any]],
    ) -> None:
        """Print what was measured. The ports carry these values, but a run that
        only calls ``execute()`` displays figures and nothing else."""
        pops = sorted(set(firing_rate_hz) | set(isi_stats) | set(rate_over_time))
        if not pops:
            print("[NW_Analysis] no values measured")
            return

        print("[NW_Analysis] measured values:")
        for pop in pops:
            parts = []
            if pop in firing_rate_hz:
                parts.append(f"rate {firing_rate_hz[pop]:.2f} Hz")
            isi = isi_stats.get(pop) or {}
            if isi.get("n_intervals"):
                parts.append(
                    f"ISI mean {isi['mean_ms']:.1f} ms, CV {isi['cv']:.2f} "
                    f"(n={isi['n_intervals']})"
                )
            else:
                parts.append("ISI unavailable (no neuron spiked twice)")
            rates = (rate_over_time.get(pop) or {}).get("rate_hz") or []
            if rates:
                bin_ms = rate_over_time[pop]["bin_ms"]
                parts.append(
                    f"peak {max(rates):.2f} Hz in {len(rates)} bins of {bin_ms:g} ms"
                )
            print(f"  {pop}: " + " | ".join(parts))

    def analyze(self, results: Dict) -> Dict[str, Any]:
        import os
        import matplotlib.pyplot as plt

        config_file = results["config_file"]
        output_dir  = results["output_dir"]
        p = self._parameters

        # Measured before plotting, so it does not depend on the plotting flags.
        firing_rate_hz = self._measure_firing_rates(config_file, output_dir)
        isi_stats = self._measure_isi_stats(output_dir, config_file)
        rate_over_time = self._measure_rate_over_time(
            config_file, output_dir, float(p["rate_bin_ms"]))

        populations = list(p["populations"]) or self._detect_populations(output_dir, str(p["report_name"]))
        node_ids    = list(p["trace_node_ids"]) or None
        figures: Dict[str, Any] = {"raster": None, "traces": None, "rate": None}

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

        if bool(p["plot_rate"]) and rate_over_time:
            # Its own figure: this is a population measure over time, not a per-neuron
            # view, so it does not belong on the raster or the trace axes.
            requested = list(p["populations"])
            rate_paths = []
            for pop, measured in rate_over_time.items():
                if requested and pop not in requested:
                    continue
                try:
                    plt.figure()
                    plt.plot(measured["t_ms"], measured["rate_hz"])
                    plt.xlabel("time (ms)")
                    plt.ylabel("rate (Hz)")
                    # A rate axis that does not reach zero turns a small fluctuation
                    # into a dramatic-looking one.
                    plt.ylim(bottom=0)
                    plt.title(f"{pop} population rate ({measured['bin_ms']:g} ms bins)")
                    if bool(p["save_figures"]):
                        path = os.path.join(output_dir, f"rate_{pop}.png")
                        plt.savefig(path, bbox_inches="tight")
                        rate_paths.append(path)
                    plt.show()
                except Exception as e:
                    print(f"[NW_Analysis] rate plot skipped for {pop!r}: {e}")
            figures["rate"] = rate_paths or None

        self._print_summary(firing_rate_hz, isi_stats, rate_over_time)

        return {"figures": figures, "firing_rate_hz": firing_rate_hz,
                "isi_stats": isi_stats, "rate_over_time": rate_over_time}
