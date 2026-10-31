"""NW_SpikeTrains with nwb_file: recorded spikes from an NWB file drive the network.

BMTK's ecephys_probe module does the reading when the simulation starts; the node
only creates the virtual cells and writes the config entry. These tests build a
tiny NWB file in the Allen Neuropixels layout BMTK reads (a units table with
spike_times, joined to the electrodes table through peak_channel_id), so every
recorded spike time is known, then check the entry and what NEST does with it.
"""

from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bmtk")
pytest.importorskip("pynwb")

from neuroworkflow.nodes.stimulus.NW_SpikeTrains import NW_SpikeTrains

#: recorded spike times in seconds, one list per recorded neuron, 200 ms apart
RECORDED_S = [[0.1, 0.3], [0.5], [0.7, 0.9]]
LOCATIONS = ["VISp", "VISl", "CA1"]
FIRING_RATES = [2.0, 1.0, 2.0]


def _write_nwb(path, spikes_s=RECORDED_S):
    from pynwb import NWBHDF5IO, NWBFile

    nwb = NWBFile(session_description="NW_SpikeTrains test", identifier="nw-test",
                  session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))
    device = nwb.create_device(name="probe")
    group = nwb.create_electrode_group(name="shank", description="test shank",
                                       location="brain", device=device)
    for location in LOCATIONS[:len(spikes_s)]:
        nwb.add_electrode(location=location, group=group)
    nwb.add_unit_column(name="peak_channel_id", description="electrode of the peak")
    nwb.add_unit_column(name="firing_rate", description="mean rate in Hz")
    for i, times in enumerate(spikes_s):
        nwb.add_unit(spike_times=times, peak_channel_id=i, firing_rate=FIRING_RATES[i])
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return str(path)


def _node(tmp_path, **params):
    tmp_path.mkdir(parents=True, exist_ok=True)
    node = NW_SpikeTrains("drive")
    node._context["results_path"] = str(tmp_path)
    node.configure(**{"pop_name": "drive", "nwb_file": _write_nwb(tmp_path / "rec.nwb"),
                      **params})
    return node


def _entry(population):
    return population["_sim_inputs"]["drive_spikes"]


def test_units_map_makes_one_cell_per_recorded_neuron(tmp_path):
    pop = _node(tmp_path, n_trains=50).build()["population"]
    entry = _entry(pop)

    assert pop["builder"].nodes().__len__() == 3, "n_trains must be replaced by the neuron count"
    assert pop["_signature"]["n_trains"] == 3
    assert entry["module"] == "ecephys_probe"
    assert entry["mapping"] == "units_map"
    assert entry["node_set"] == {"population": "drive"}
    assert entry["interval"] == [0.0, 1000.0] and entry["interval_offset"] == 0.0

    mapping = pd.read_csv(entry["units"], sep=" ")
    assert list(mapping.columns) == ["node_ids", "unit_ids"]
    assert mapping["node_ids"].tolist() == [0, 1, 2]
    assert mapping["unit_ids"].tolist() == [0, 1, 2]


def test_sample_keeps_n_trains_and_writes_no_map(tmp_path):
    entry = _entry(_node(tmp_path, n_trains=2, nwb_mapping="sample").build()["population"])

    assert entry["mapping"] == "sample"
    assert "units" not in entry


def test_sample_with_more_cells_than_neurons_is_refused(tmp_path):
    with pytest.raises(ValueError, match="more than the 3 selected neurons"):
        _node(tmp_path, n_trains=4, nwb_mapping="sample").build()


def test_sample_with_replacement_allows_more_cells_than_neurons(tmp_path):
    pop = _node(tmp_path, n_trains=10, nwb_mapping="sample_with_replacement").build()["population"]

    assert len(pop["builder"].nodes()) == 10


def test_the_piece_cut_from_the_recording_is_set_by_nwb_start_ms(tmp_path):
    entry = _entry(_node(tmp_path, nwb_start_ms=1800000.0, start_ms=0.0,
                         stop_ms=3000.0).build()["population"])

    assert entry["interval"] == [1800000.0, 1803000.0]
    assert entry["interval_offset"] == 0.0


def test_an_area_filter_keeps_only_that_area(tmp_path):
    pop = _node(tmp_path, nwb_units={"location": "VISl"}).build()["population"]

    assert len(pop["builder"].nodes()) == 1
    assert pd.read_csv(_entry(pop)["units"], sep=" ")["unit_ids"].tolist() == [1]


def test_a_list_of_areas_keeps_any_of_them(tmp_path):
    pop = _node(tmp_path, nwb_units={"location": ["VISp", "CA1"]}).build()["population"]

    assert pd.read_csv(_entry(pop)["units"], sep=" ")["unit_ids"].tolist() == [0, 2]


def test_a_comparison_filter_is_applied(tmp_path):
    pop = _node(tmp_path, nwb_units={"firing_rate": {"operation": ">", "value": 1.5}}
                ).build()["population"]

    assert pd.read_csv(_entry(pop)["units"], sep=" ")["unit_ids"].tolist() == [0, 2]


def test_sample_passes_the_filter_to_bmtk_and_counts_only_the_selected(tmp_path):
    entry = _entry(_node(tmp_path, n_trains=2, nwb_mapping="sample",
                         nwb_units={"location": ["VISp", "CA1"]}).build()["population"])
    assert entry["units"] == {"location": ["VISp", "CA1"]}

    with pytest.raises(ValueError, match="more than the 1 selected"):
        _node(tmp_path / "visl", n_trains=2, nwb_mapping="sample",
              nwb_units={"location": "VISl"}).build()


def test_a_filter_matching_nothing_is_named(tmp_path):
    with pytest.raises(ValueError, match="match nwb_units"):
        _node(tmp_path, nwb_units={"location": "LGd"}).build()


def test_a_recorded_spike_at_time_zero_is_refused_before_nest_sees_it(tmp_path):
    node = NW_SpikeTrains("drive")
    node._context["results_path"] = str(tmp_path)
    node.configure(pop_name="drive", nwb_file=_write_nwb(tmp_path / "z.nwb", [[0.0, 0.2]]))

    with pytest.raises(ValueError, match="t<=0"):
        node.build()


def test_a_spike_at_the_start_of_a_later_piece_is_refused_too(tmp_path):
    """The spike recorded at 300 ms is the first of a piece cut from 300 ms and would
    play at model time 0."""
    with pytest.raises(ValueError, match="t<=0"):
        _node(tmp_path, nwb_start_ms=300.0, start_ms=0.0, stop_ms=500.0).build()


def test_a_missing_file_is_named(tmp_path):
    node = NW_SpikeTrains("drive")
    node._context["results_path"] = str(tmp_path)
    node.configure(nwb_file=str(tmp_path / "absent.nwb"))

    with pytest.raises(ValueError, match="nwb_file not found"):
        node.build()


# --- through NEST -----------------------------------------------------------------

NEST_PARAMS = {"C_m": 250.0, "tau_m": 10.0, "t_ref": 2.0,
               "V_th": -55.0, "V_reset": -70.0, "E_L": -70.0}
DELAY = 1.5
#: one input spike alone drives the target over threshold
WEIGHT = 3000.0


def _target_spikes(tmp_path, **params):
    """Recorded neurons -> virtual cells -> one iaf_psc_alpha; its spike times in ms."""
    pytest.importorskip("nest")
    from neuroworkflow.nodes.network.NW_Connectivity import NW_Connectivity
    from neuroworkflow.nodes.network.NW_Population import NW_Population
    from neuroworkflow.nodes.simulation.NW_SimConfig import NW_SimConfig

    drive = _node(tmp_path, **params).build()["population"]
    target = NW_Population("target")
    target._context["results_path"] = str(tmp_path)
    target.configure(pop_name="target", N=1, model_template="nest:iaf_psc_alpha",
                     nest_params=NEST_PARAMS)
    conn = NW_Connectivity("conn")
    conn._context["results_path"] = str(tmp_path)
    conn.configure(connections=[{"source": "drive", "target": "target"}],
                   connection_rule=1, syn_weight=WEIGHT, delay=DELAY)
    payload = conn.connect([drive, target.build()["population"]])["network"]

    cfg = NW_SimConfig("sim")
    cfg._context["results_path"] = str(tmp_path)
    cfg.configure(simulator="pointnet", config_file="config.json", tstop_ms=1000.0,
                  dt_ms=0.1, compile_mechanisms=False, overwrite=True, reports={})
    cfg.run(**cfg.setup(payload))

    with h5py.File(tmp_path / "output" / "spikes.h5") as f:
        if "target" not in f["spikes"]:
            return np.array([])
        return np.sort(f["spikes/target/timestamps"][()])


def _answers(target_ms, input_ms):
    """Each input spike is answered by the target within 10 ms, and the target never
    fires otherwise - so the target's spikes date the recorded ones."""
    input_ms = np.asarray(input_ms)
    for t in input_ms:
        assert np.any((target_ms >= t + DELAY) & (target_ms <= t + DELAY + 10.0)), t
    for s in target_ms:
        assert np.any((s >= input_ms + DELAY) & (s <= input_ms + DELAY + 10.0)), s


def test_recorded_spikes_play_at_their_recorded_times(tmp_path):
    target = _target_spikes(tmp_path)

    _answers(target, [100.0, 300.0, 500.0, 700.0, 900.0])


def test_the_window_keeps_only_the_spikes_inside_it(tmp_path):
    """Recording [250, 750] ms played from model 250 ms keeps the recorded times."""
    target = _target_spikes(tmp_path, nwb_start_ms=250.0, start_ms=250.0, stop_ms=750.0)

    _answers(target, [300.0, 500.0, 700.0])


def test_a_later_piece_of_the_recording_plays_from_start_ms(tmp_path):
    """Recording [600, 900] ms played from model 100 ms: the spikes recorded at 700
    and 900 ms arrive at 200 and 400 ms."""
    target = _target_spikes(tmp_path, nwb_start_ms=600.0, start_ms=100.0, stop_ms=400.0)

    _answers(target, [200.0, 400.0])


def test_the_area_filter_reaches_the_simulation(tmp_path):
    """Only the VISl neuron (spike at 500 ms) drives the target."""
    target = _target_spikes(tmp_path, nwb_units={"location": "VISl"})

    _answers(target, [500.0])
