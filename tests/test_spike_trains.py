"""NW_SpikeTrains must produce spike statistics that are what they claim to be.

The node wraps BMTK's PoissonSpikeGenerator and GammaSpikeGenerator: a population
of virtual cells, a SONATA spikes file, and a config "inputs" entry. Nothing here
runs a simulator - these check the generated trains and the shape of the entry,
because a wrong entry fails inside NEST with an error that names neither this node
nor the key at fault.
"""

import numpy as np
import pytest

pytest.importorskip("bmtk")

from neuroworkflow.nodes.stimulus.NW_SpikeTrains import NW_SpikeTrains


def _node(tmp_path, **params):
    node = NW_SpikeTrains("drive")
    node._context["results_path"] = str(tmp_path)
    if params:
        node.configure(**params)
    return node


def test_the_measured_rate_matches_the_requested_rate(tmp_path):
    """20 cells at 40 Hz for 1 s is about 800 spikes. A wrong unit conversion
    between the node's milliseconds and BMTK's seconds would be 1000x off."""
    node = _node(tmp_path, pop_name="drive", n_trains=20, firing_rate_hz=40.0,
                 start_ms=0.0, stop_ms=1000.0, seed=3)
    node.build()

    import h5py
    with h5py.File(tmp_path / "inputs" / "drive_spikes.h5") as f:
        n_spikes = len(f["spikes/drive/timestamps"])

    assert 700 < n_spikes < 900


def test_each_virtual_cell_fires_its_own_train(tmp_path):
    """N cells must be N independent realisations, not N copies of one train."""
    node = _node(tmp_path, pop_name="drive", n_trains=5, firing_rate_hz=50.0,
                 start_ms=0.0, stop_ms=1000.0, seed=3)
    node.build()

    import h5py
    with h5py.File(tmp_path / "inputs" / "drive_spikes.h5") as f:
        ids = np.asarray(f["spikes/drive/node_ids"])
        times = np.asarray(f["spikes/drive/timestamps"])

    assert set(ids.tolist()) == {0, 1, 2, 3, 4}
    trains = [np.sort(times[ids == i]) for i in range(5)]
    assert not np.array_equal(trains[0], trains[1][: trains[0].size])


def test_a_rate_given_as_a_list_varies_over_time(tmp_path):
    """BMTK wants a time for every rate value, not a start/stop pair; passing the
    pair silently raises inside BMTK instead of producing a varying rate."""
    rates = [2.0] * 50 + [60.0] * 50
    node = _node(tmp_path, pop_name="drive", n_trains=10, firing_rate_hz=rates,
                 start_ms=0.0, stop_ms=1000.0, seed=3)
    node.build()

    import h5py
    with h5py.File(tmp_path / "inputs" / "drive_spikes.h5") as f:
        times = np.asarray(f["spikes/drive/timestamps"])

    first_half = int((times < 500).sum())
    second_half = int((times >= 500).sum())
    assert second_half > 5 * first_half


def test_a_higher_gamma_shape_fires_more_regularly(tmp_path):
    """Shape 1 is Poisson; above 1 the intervals tighten. If the shape were ignored
    the two would be statistically identical."""
    import h5py

    def cv(shape, path):
        node = _node(path, pop_name="drive", n_trains=20, distribution="gamma",
                     firing_rate_hz=40.0, gamma_shape=shape,
                     start_ms=0.0, stop_ms=2000.0, seed=11)
        node.build()
        with h5py.File(path / "inputs" / "drive_spikes.h5") as f:
            ids = np.asarray(f["spikes/drive/node_ids"])
            times = np.asarray(f["spikes/drive/timestamps"])
        intervals = np.concatenate(
            [np.diff(np.sort(times[ids == i])) for i in np.unique(ids)]
        )
        return intervals.std() / intervals.mean()

    poisson_like = cv(1.0, tmp_path / "a")
    regular = cv(8.0, tmp_path / "b")

    assert poisson_like > regular


def test_the_input_entry_names_the_population_as_a_dict(tmp_path):
    """BMTK registers each population name as a node set, so the bare name "drive"
    would work too. The dict is pinned because it states the selection outright
    rather than relying on that registration."""
    node = _node(tmp_path, pop_name="drive", n_trains=4, start_ms=1.0, stop_ms=100.0)
    population = node.build()["population"]

    entry = population["_sim_inputs"]["drive_spikes"]
    assert entry["node_set"] == {"population": "drive"}
    assert entry["input_type"] == "spikes"
    assert entry["module"] == "sonata"


def test_the_output_has_the_shape_NW_Connectivity_consumes(tmp_path):
    """NW_Connectivity looks up pop_name and calls add_edges on builder."""
    node = _node(tmp_path, pop_name="drive", n_trains=4, start_ms=1.0, stop_ms=100.0)
    population = node.build()["population"]

    assert population["pop_name"] == "drive"
    assert hasattr(population["builder"], "add_edges")
    assert population["network_dir"].endswith("network")


def _setup_config(tmp_path, with_virtual):
    """Build a tiny network through the real nodes and return the written config."""
    import json

    from neuroworkflow.nodes.network.NW_Population import NW_Population
    from neuroworkflow.nodes.simulation.NW_SimConfig import NW_SimConfig

    pop = NW_Population("v1")
    cfg = NW_SimConfig("cfg")
    for node in (pop, cfg):
        node._context["results_path"] = str(tmp_path)
    pop.configure(pop_name="v1", N=2, model_template="nest:iaf_psc_alpha",
                  nest_params={"C_m": 250.0, "tau_m": 10.0, "t_ref": 2.0,
                               "V_th": -55.0, "V_reset": -70.0, "E_L": -70.0})
    cfg.configure(simulator="pointnet", config_file="config.json", tstop_ms=50.0,
                  dt_ms=0.1, compile_mechanisms=False, overwrite=True)

    populations = {"v1": pop.build()["population"]}
    if with_virtual:
        drive = _node(tmp_path, pop_name="drive", n_trains=2,
                      start_ms=1.0, stop_ms=50.0)
        populations["drive"] = drive.build()["population"]

    cfg.setup(populations)
    return json.loads((tmp_path / "config.json").read_text())


def test_a_virtual_population_is_dropped_from_an_all_cells_report(tmp_path):
    """Virtual cells have no membrane potential. Left in, the simulator fails with a
    KeyError naming only the population, so the report is narrowed for the user."""
    config = _setup_config(tmp_path, with_virtual=True)

    assert config["reports"]["v_report"]["cells"] == {"population": ["v1"]}


def test_a_network_with_no_virtual_population_keeps_its_report_untouched(tmp_path):
    """Every existing network must write exactly the report it wrote before."""
    config = _setup_config(tmp_path, with_virtual=False)

    assert config["reports"]["v_report"]["cells"] == "all"


def test_changing_the_rate_does_not_invalidate_the_network(tmp_path):
    """The trains live in the spikes file, rewritten every build; only the cells
    themselves are in the network. If the rate were part of the signature, every
    optimization trial would rebuild the network into the same directory - and the
    second rebuild in one process fails on an HDF5 file that is still open."""
    first = _node(tmp_path, pop_name="drive", n_trains=8, firing_rate_hz=10.0,
                  start_ms=1.0, stop_ms=100.0).build()["population"]["_signature"]
    faster = _node(tmp_path, pop_name="drive", n_trains=8, firing_rate_hz=90.0,
                   distribution="gamma", gamma_shape=4.0, seed=99,
                   start_ms=1.0, stop_ms=100.0).build()["population"]["_signature"]

    assert first == faster


def test_changing_the_cell_count_does_invalidate_the_network(tmp_path):
    """Those cells are written into the SONATA node files, so the network is stale."""
    eight = _node(tmp_path, pop_name="drive", n_trains=8,
                  start_ms=1.0, stop_ms=100.0).build()["population"]["_signature"]
    twenty = _node(tmp_path, pop_name="drive", n_trains=20,
                   start_ms=1.0, stop_ms=100.0).build()["population"]["_signature"]

    assert eight != twenty


def test_a_window_that_covers_no_time_is_refused(tmp_path):
    """Silently generating nothing would look like a network that never fired."""
    node = _node(tmp_path, pop_name="drive", start_ms=500.0, stop_ms=500.0)

    with pytest.raises(ValueError, match="must be after"):
        node.build()
