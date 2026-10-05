import os
from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition,
)
from neuroworkflow.core.port import PortType


class NW_NetworkView(Node):
    """
    Measures and draws the network that was built, rather than what it did.

    NW_Analysis answers "how did the cells fire"; this answers "what did my
    connection rules actually produce". Both read the same results dict from
    NW_SimConfig and can sit side by side.

    It reads the SONATA files directly - the same files the simulator reads - so
    what it reports is the network that ran, not the parameters that were meant to
    produce it. The two differ more often than expected: a 10% rule over twenty
    source neurons leaves some targets with no input at all, and that silence is
    indistinguishable from a modelling mistake once the simulation is running.
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_network_view",
        stage="analysis",
        tool="BMTK",
        model_source="https://alleninstitute.github.io/bmtk/",
        description=(
            "Reads the SONATA network files a simulation ran on and reports what the "
            "connection rules produced: how many pairs each projection connected, how "
            "many synapses they carry, the resulting density, and how many inputs each "
            "target received. Draws a connectivity matrix and in-degree distributions. "
            "Connect to NW_SimConfig.results, alongside NW_Analysis."
        ),
        parameters={
            "plot_matrix": ParameterDefinition(
                default_value=True,
                description=(
                    "Draw the population-by-population connectivity matrix, coloured by "
                    "the number of synapses from each source population to each target."
                ),
            ),
            "plot_in_degree": ParameterDefinition(
                default_value=True,
                description=(
                    "Draw, for each projection, how many source neurons contact each "
                    "target. A bar at zero is the one to look for: those targets receive "
                    "nothing through this projection."
                ),
            ),
            "save_figures": ParameterDefinition(
                default_value=True,
                description="Save figure PNG files next to the simulation output.",
            ),
        },
        inputs={
            "results": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict from NW_SimConfig with config_file (str), output_dir (str), "
                    "simulator (str). The network files are located through the config's "
                    "manifest, so this works wherever the run wrote them."
                ),
            ),
        },
        outputs={
            "figures": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dict with keys 'matrix' and 'in_degree', each the path to a saved "
                    "PNG (or None when that plot is disabled)."
                ),
            ),
            "populations": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Neurons per population, keyed by population name, read from the "
                    "SONATA node files rather than from the parameters that built them."
                ),
            ),
            "projections": PortDefinition(
                type=PortType.DICT,
                description=(
                    "One entry per projection, keyed 'source->target': "
                    "{'pairs', 'synapses', 'density_percent', 'in_degree_min', "
                    "'in_degree_mean', 'in_degree_max', 'targets_reached', "
                    "'edge_types', 'edge_types_share_wiring'}. density_percent is the "
                    "share of all possible source-target pairs that are connected. "
                    "edge_types counts the distinct synapse types on the projection - "
                    "more than one means a multi-receptor pathway, and "
                    "edge_types_share_wiring says whether they contacted the same pairs, "
                    "which is what makes them components of one synapse rather than "
                    "separate connections."
                ),
            ),
            "unconnected": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Per projection, how many target neurons received nothing through "
                    "it. A population silent for this reason looks exactly like a "
                    "modelling error, so it is reported as a number rather than left to "
                    "be noticed in a raster."
                ),
            ),
        },
        methods={
            "inspect": MethodDefinition(
                description=(
                    "Read the SONATA network files, measure every projection, and draw "
                    "the connectivity matrix and in-degree distributions."
                ),
                inputs=["results"],
                outputs=["figures", "populations", "projections", "unconnected"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("inspect", self.inspect, method_key="inspect")

    @staticmethod
    def _network_dir(config_file: str) -> str:
        """Where the SONATA files are, according to the config that ran.

        The manifest is the simulator's own answer, so a run that wrote somewhere
        unusual - an optimization trial directory, a staged cluster job - is read
        correctly instead of guessed at from results_path.
        """
        import json

        with open(config_file) as fh:
            manifest = json.load(fh).get("manifest", {}) or {}
        network = manifest.get("$NETWORK_DIR", "")
        base = manifest.get("$BASE_DIR", "")
        if network and base:
            network = network.replace("$BASE_DIR", base)
        if network and os.path.isdir(network):
            return network
        return os.path.join(os.path.dirname(os.path.abspath(config_file)), "network")

    @staticmethod
    def _population_sizes(network_dir: str) -> Dict[str, int]:
        import glob

        import h5py

        sizes: Dict[str, int] = {}
        for path in sorted(glob.glob(os.path.join(network_dir, "*_nodes.h5"))):
            with h5py.File(path, "r") as f:
                for pop, group in f.get("nodes", {}).items():
                    if "node_id" in group:
                        sizes[pop] = len(group["node_id"])
        return sizes

    def _measure(self, network_dir: str, sizes: Dict[str, int]):
        """Every projection on disk, measured from its edges."""
        import glob

        import h5py
        import numpy as np

        projections: Dict[str, Dict[str, Any]] = {}
        unconnected: Dict[str, int] = {}

        for path in sorted(glob.glob(os.path.join(network_dir, "*_edges.h5"))):
            with h5py.File(path, "r") as f:
                for name, group in f.get("edges", {}).items():
                    if "source_node_id" not in group or "_to_" not in name:
                        continue
                    src_pop, tgt_pop = name.split("_to_", 1)
                    # SONATA stores ids unsigned; numpy refuses to mix those with
                    # the signed arithmetic below.
                    src = np.asarray(group["source_node_id"]).astype(np.int64)
                    tgt = np.asarray(group["target_node_id"]).astype(np.int64)
                    nsyns = (np.asarray(group["0/nsyns"]).astype(np.int64)
                             if "0/nsyns" in group else np.ones(src.size, np.int64))
                    type_ids = (np.asarray(group["edge_type_id"]).astype(np.int64)
                                if "edge_type_id" in group
                                else np.zeros(src.size, np.int64))

                    n_src = sizes.get(src_pop, int(src.max()) + 1 if src.size else 0)
                    n_tgt = sizes.get(tgt_pop, int(tgt.max()) + 1 if tgt.size else 0)
                    pairs = set(zip(src.tolist(), tgt.tolist()))
                    in_degree = np.bincount(tgt, minlength=max(n_tgt, 1))[:max(n_tgt, 1)]

                    # Two edge types over the same pairs are one synapse with two
                    # receptor components; over different pairs they are two separate
                    # connections, which is a different model and easy to produce by
                    # giving the projections different connection rules by mistake.
                    distinct_types = np.unique(type_ids)
                    wiring = {frozenset(zip(src[type_ids == t].tolist(),
                                            tgt[type_ids == t].tolist()))
                              for t in distinct_types}

                    key = f"{src_pop}->{tgt_pop}"
                    projections[key] = {
                        "pairs": len(pairs),
                        "synapses": int(nsyns.sum()),
                        "density_percent": (100.0 * len(pairs) / (n_src * n_tgt)
                                            if n_src and n_tgt else 0.0),
                        "in_degree_min": int(in_degree.min()) if in_degree.size else 0,
                        "in_degree_mean": float(in_degree.mean()) if in_degree.size else 0.0,
                        "in_degree_max": int(in_degree.max()) if in_degree.size else 0,
                        "targets_reached": int((in_degree > 0).sum()),
                        "edge_types": int(distinct_types.size),
                        "edge_types_share_wiring": len(wiring) == 1,
                        "_in_degree": in_degree,          # for the plot, dropped below
                    }
                    unconnected[key] = int((in_degree == 0).sum())

        return projections, unconnected

    def _plot_matrix(self, projections, sizes, output_dir, save):
        import matplotlib.pyplot as plt
        import numpy as np

        names = sorted(sizes)
        if not names:
            return None
        grid = np.zeros((len(names), len(names)))
        for key, data in projections.items():
            src, tgt = key.split("->")
            if src in names and tgt in names:
                grid[names.index(src), names.index(tgt)] = data["synapses"]

        plt.figure()
        plt.imshow(grid, cmap="viridis")
        plt.colorbar(label="synapses")
        plt.xticks(range(len(names)), names, rotation=45, ha="right")
        plt.yticks(range(len(names)), names)
        plt.xlabel("target population")
        plt.ylabel("source population")
        plt.title("connectivity")
        for i in range(len(names)):
            for j in range(len(names)):
                if grid[i, j]:
                    plt.text(j, i, f"{int(grid[i, j])}", ha="center", va="center",
                             color="w", fontsize=8)
        path = None
        if save:
            path = os.path.join(output_dir, "connectivity_matrix.png")
            plt.savefig(path, bbox_inches="tight")
        plt.show()
        return path

    def _plot_in_degree(self, projections, output_dir, save):
        import matplotlib.pyplot as plt

        items = [(k, v["_in_degree"]) for k, v in sorted(projections.items())]
        if not items:
            return None
        fig, axes = plt.subplots(len(items), 1, figsize=(6, 2.2 * len(items)),
                                 squeeze=False)
        for ax, (key, in_degree) in zip(axes[:, 0], items):
            ax.hist(in_degree, bins=range(int(in_degree.max()) + 2), align="left",
                    color="tab:blue")
            silent = int((in_degree == 0).sum())
            ax.set_title(f"{key}   ({silent} target(s) with no input)" if silent
                         else key, fontsize=9)
            ax.set_xlabel("inputs per target neuron")
            ax.set_ylabel("neurons")
        fig.tight_layout()
        path = None
        if save:
            path = os.path.join(output_dir, "in_degree.png")
            plt.savefig(path, bbox_inches="tight")
        plt.show()
        return path

    def inspect(self, results: Dict) -> Dict[str, Any]:
        p = self._parameters
        config_file = results["config_file"]
        output_dir = results["output_dir"]

        network_dir = self._network_dir(config_file)
        sizes = self._population_sizes(network_dir)
        if not sizes:
            print(f"[NW_NetworkView] no SONATA node files in {network_dir}")
            return {"figures": {"matrix": None, "in_degree": None},
                    "populations": {}, "projections": {}, "unconnected": {}}

        projections, unconnected = self._measure(network_dir, sizes)
        figures: Dict[str, Any] = {"matrix": None, "in_degree": None}

        if projections:
            if bool(p["plot_matrix"]):
                try:
                    figures["matrix"] = self._plot_matrix(
                        projections, sizes, output_dir, bool(p["save_figures"]))
                except Exception as e:
                    print(f"[NW_NetworkView] connectivity matrix skipped: {e}")
            if bool(p["plot_in_degree"]):
                try:
                    figures["in_degree"] = self._plot_in_degree(
                        projections, output_dir, bool(p["save_figures"]))
                except Exception as e:
                    print(f"[NW_NetworkView] in-degree plot skipped: {e}")

        # The arrays were only needed for the plots; the ports carry numbers.
        for data in projections.values():
            data.pop("_in_degree", None)

        self._print_summary(sizes, projections, unconnected)
        return {"figures": figures, "populations": sizes,
                "projections": projections, "unconnected": unconnected}

    @staticmethod
    def _print_summary(sizes, projections, unconnected) -> None:
        """The ports carry these, but a run that only calls execute() shows nothing."""
        print(f"[NW_NetworkView] populations: {sizes}")
        if not projections:
            print("[NW_NetworkView] no projections found")
            return
        for key, d in sorted(projections.items()):
            line = (f"  {key}: {d['pairs']} pairs, {d['synapses']} synapses, "
                    f"{d['density_percent']:.1f}% density, in-degree "
                    f"{d['in_degree_min']}-{d['in_degree_max']} "
                    f"(mean {d['in_degree_mean']:.1f})")
            if d["edge_types"] > 1:
                line += (f" | {d['edge_types']} edge types, "
                         f"{'same' if d['edge_types_share_wiring'] else 'DIFFERENT'} wiring")
            silent = unconnected.get(key, 0)
            if silent:
                line += f" | {silent} target(s) receive nothing"
            print(line)
