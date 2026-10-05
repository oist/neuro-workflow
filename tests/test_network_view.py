"""NW_NetworkView must report the network that was built, not the one intended.

The parameters say what the connection rules were asked for; the SONATA files say
what they produced, and the two part company more often than expected. A 10% rule
over twenty source neurons leaves some targets with no input at all - in the
repository's own balanced network, 12 of 80 excitatory cells receive no inhibition
- and once the simulation is running that silence is indistinguishable from a
modelling mistake.

The files are written here directly rather than by running a build, so these tests
need neither BMTK nor NEST: the node's job is reading SONATA, and SONATA is what it
is given.
"""

import h5py
import numpy as np
import pytest

from neuroworkflow.nodes.analysis.NW_NetworkView import NW_NetworkView


def _network(tmp_path, sizes, edges):
    """edges: {(src_pop, tgt_pop): [(src_id, tgt_id, edge_type_id, nsyns), ...]}"""
    import json

    net = tmp_path / "network"
    out = tmp_path / "output"
    net.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    for pop, n in sizes.items():
        with h5py.File(net / f"{pop}_nodes.h5", "w") as f:
            f.create_dataset(f"nodes/{pop}/node_id", data=np.arange(n))

    for (src_pop, tgt_pop), rows in edges.items():
        with h5py.File(net / f"{src_pop}_{tgt_pop}_edges.h5", "w") as f:
            g = f.create_group(f"edges/{src_pop}_to_{tgt_pop}")
            g.create_dataset("source_node_id", data=np.array([r[0] for r in rows], np.uint64))
            g.create_dataset("target_node_id", data=np.array([r[1] for r in rows], np.uint64))
            g.create_dataset("edge_type_id", data=np.array([r[2] for r in rows], np.int64))
            g.create_dataset("0/nsyns", data=np.array([r[3] for r in rows], np.int64))

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"manifest": {
        "$BASE_DIR": str(tmp_path), "$NETWORK_DIR": "$BASE_DIR/network"}}))
    return str(config), str(out)


def _inspect(tmp_path, sizes, edges):
    config, output = _network(tmp_path, sizes, edges)
    node = NW_NetworkView("view")
    node.configure(plot_matrix=False, plot_in_degree=False, save_figures=False)
    return node.inspect({"config_file": config, "output_dir": output,
                         "simulator": "pointnet"})


def test_it_counts_pairs_synapses_and_density(tmp_path):
    """2 of the 2x3 possible pairs are connected, one carrying three synapses."""
    out = _inspect(tmp_path, {"a": 2, "b": 3},
                   {("a", "b"): [(0, 0, 100, 1), (1, 2, 100, 3)]})

    proj = out["projections"]["a->b"]
    assert proj["pairs"] == 2
    assert proj["synapses"] == 4
    assert proj["density_percent"] == pytest.approx(100 * 2 / 6)


def test_it_names_the_targets_that_receive_nothing(tmp_path):
    """The reason a population can be silent without anything being wrong with the
    neurons. Reported as a number rather than left to be spotted in a raster."""
    out = _inspect(tmp_path, {"a": 2, "b": 4},
                   {("a", "b"): [(0, 0, 100, 1), (1, 1, 100, 1)]})

    assert out["unconnected"]["a->b"] == 2          # targets 2 and 3 get nothing
    assert out["projections"]["a->b"]["targets_reached"] == 2
    assert out["projections"]["a->b"]["in_degree_min"] == 0


def test_two_edge_types_over_the_same_pairs_are_one_synapse(tmp_path):
    """AMPA and NMDA on one pathway: two edge types, identical wiring. This is what
    makes them components of one synapse."""
    rows = [(0, 0, 100, 1), (1, 1, 100, 1),     # AMPA
            (0, 0, 101, 1), (1, 1, 101, 1)]     # NMDA, same pairs
    out = _inspect(tmp_path, {"a": 2, "b": 2}, {("a", "b"): rows})

    proj = out["projections"]["a->b"]
    assert proj["edge_types"] == 2
    assert proj["edge_types_share_wiring"] is True


def test_two_edge_types_over_different_pairs_are_flagged(tmp_path):
    """The same two projections given different connection rules by mistake. The
    network runs and looks plausible; only the wiring tells you it is wrong."""
    rows = [(0, 0, 100, 1), (1, 1, 100, 1),     # AMPA on the diagonal
            (0, 1, 101, 1), (1, 0, 101, 1)]     # NMDA on the opposite pairs
    out = _inspect(tmp_path, {"a": 2, "b": 2}, {("a", "b"): rows})

    proj = out["projections"]["a->b"]
    assert proj["edge_types"] == 2
    assert proj["edge_types_share_wiring"] is False


def test_population_sizes_come_from_the_files_not_the_parameters(tmp_path):
    out = _inspect(tmp_path, {"exc": 80, "inh": 20},
                   {("exc", "inh"): [(0, 0, 100, 1)]})

    assert out["populations"] == {"exc": 80, "inh": 20}


def test_the_network_is_found_through_the_config_manifest(tmp_path):
    """An optimization trial or a staged cluster job writes somewhere other than
    results_path; the manifest is the simulator's own answer to where."""
    config, _ = _network(tmp_path, {"a": 1}, {})

    assert NW_NetworkView._network_dir(config) == str(tmp_path / "network")


def test_a_missing_network_reports_instead_of_raising(tmp_path):
    """Reading results that were never built must not take the workflow down."""
    import json

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"manifest": {"$BASE_DIR": str(tmp_path)}}))
    node = NW_NetworkView("view")

    out = node.inspect({"config_file": str(config), "output_dir": str(tmp_path),
                        "simulator": "pointnet"})

    assert out["populations"] == {}
    assert out["projections"] == {}
