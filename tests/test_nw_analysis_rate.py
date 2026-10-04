"""NW_Analysis must report what it measured, not only draw it.

The node used to return firing rates and ISI statistics to its ports and print
nothing, so a run showed figures and no numbers. It also reported a single rate
for the whole run, which cannot show that a network fires in bursts or falls
silent halfway through.

These tests cover the time-resolved rate and the summary print. They build SONATA
files directly rather than running a simulator, so nothing here needs BMTK or NEST.
The central claim is the one the port description makes: ``rate_over_time``
averaged over the run reproduces ``firing_rate_hz`` exactly. If that ever stops
holding, one of the two measurements is wrong and the pair cannot both be trusted.
"""

import json

import h5py
import numpy as np
import pytest

from neuroworkflow.nodes.analysis.NW_Analysis import NW_Analysis


def _build(root, *, n_neurons=4, spike_times=None, node_ids=None, tstop=1000.0):
    """Write the node file, spikes file and config a measurement reads."""
    network = root / "network"
    output = root / "output"
    network.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    with h5py.File(network / "popA_nodes.h5", "w") as f:
        f.create_dataset("nodes/popA/node_id", data=np.arange(n_neurons))

    times = np.array([] if spike_times is None else spike_times, dtype=float)
    ids = np.arange(times.size) % n_neurons if node_ids is None else np.array(node_ids)
    with h5py.File(output / "spikes.h5", "w") as f:
        f.create_dataset("spikes/popA/timestamps", data=times)
        f.create_dataset("spikes/popA/node_ids", data=ids)

    config = root / "config.json"
    config.write_text(json.dumps({"run": {"tstart": 0.0, "tstop": tstop}}))
    return str(config), str(output)


def _node(root):
    node = NW_Analysis("analysis")
    node._context["results_path"] = str(root)
    return node


def test_the_rate_curve_averages_to_the_single_rate(tmp_path):
    """The two rate measurements must agree, or neither can be relied on."""
    spikes = np.linspace(1.0, 999.0, 40)
    config, output = _build(tmp_path, n_neurons=4, spike_times=spikes)
    node = _node(tmp_path)

    whole_run = node._measure_firing_rates(config, output)
    over_time = node._measure_rate_over_time(config, output, 10.0)

    assert whole_run["popA"] == pytest.approx(10.0)
    assert np.mean(over_time["popA"]["rate_hz"]) == pytest.approx(whole_run["popA"])


def test_a_silent_population_reports_zeros_not_an_absent_key(tmp_path):
    """A missing key reads as a failed measurement; zeros read as a silent network."""
    config, output = _build(tmp_path, n_neurons=4, spike_times=[])
    over_time = _node(tmp_path)._measure_rate_over_time(config, output, 10.0)

    assert over_time["popA"]["rate_hz"] == [0.0] * 100
    assert over_time["popA"]["bin_ms"] == 10.0


def test_bins_tile_the_run_window(tmp_path):
    """Bin centres must sit inside the run, so a rate is never read off a partial bin."""
    config, output = _build(tmp_path, spike_times=[10.0, 20.0], tstop=1000.0)
    curve = _node(tmp_path)._measure_rate_over_time(config, output, 10.0)["popA"]

    assert len(curve["t_ms"]) == len(curve["rate_hz"]) == 100
    assert curve["t_ms"][0] == pytest.approx(5.0)
    assert curve["t_ms"][-1] == pytest.approx(995.0)


def test_a_bin_wider_than_the_run_is_refused(tmp_path):
    """Returning one bin covering a fraction of the run would overstate the rate."""
    config, output = _build(tmp_path, spike_times=[1.0], tstop=100.0)
    assert _node(tmp_path)._measure_rate_over_time(config, output, 5000.0) == {}


@pytest.mark.parametrize("run_section", [{}, {"tstart": 500.0, "tstop": 100.0}])
def test_an_unusable_run_window_measures_nothing(tmp_path, run_section):
    """No tstop, or a window that runs backwards: skip rather than invent bins."""
    config, output = _build(tmp_path, spike_times=[1.0])
    (tmp_path / "config.json").write_text(json.dumps({"run": run_section}))
    assert _node(tmp_path)._measure_rate_over_time(config, output, 10.0) == {}


def test_a_missing_spikes_file_measures_nothing(tmp_path):
    """A simulation that produced no output must not raise out of the node."""
    config, output = _build(tmp_path, spike_times=[1.0])
    (tmp_path / "output" / "spikes.h5").unlink()
    assert _node(tmp_path)._measure_rate_over_time(config, output, 10.0) == {}


def test_a_network_that_never_fired_measures_zero_not_nothing(tmp_path):
    """NEST writes a spikes file with no population groups when nothing fires. An
    empty result breaks optimization: a target addressing firing_rate_hz.<pop> stops
    resolving, and the study aborts with "No objectives" instead of scoring 0 Hz."""
    config, output = _build(tmp_path, n_neurons=4, spike_times=[])
    with h5py.File(tmp_path / "output" / "spikes.h5", "w") as f:
        f.create_group("spikes")           # the file exists, but holds no population
    node = _node(tmp_path)

    assert node._measure_firing_rates(config, output) == {"popA": 0.0}
    assert node._measure_rate_over_time(config, output, 10.0)["popA"]["rate_hz"] == [0.0] * 100
    assert node._measure_isi_stats(output, config)["popA"]["n_intervals"] == 0


def _add_driving_population(root, *, n_neurons=10, n_spikes=300, tstop=1000.0):
    """A virtual population whose spikes live in an input file, as BMTK writes them.

    BMTK never records virtual cells to the output: its spike recorder attaches to
    the real cells only. The driver's spikes are the input file instead, which the
    config names.
    """
    import json

    (root / "inputs").mkdir(exist_ok=True)
    with h5py.File(root / "network" / "drive_nodes.h5", "w") as f:
        f.create_dataset("nodes/drive/node_id", data=np.arange(n_neurons))

    spikes_file = root / "inputs" / "drive_spikes.h5"
    with h5py.File(spikes_file, "w") as f:
        f.create_dataset("spikes/drive/timestamps",
                         data=np.linspace(1.0, tstop - 1.0, n_spikes))
        f.create_dataset("spikes/drive/node_ids",
                         data=np.arange(n_spikes) % n_neurons)

    config = json.loads((root / "config.json").read_text())
    config["inputs"] = {
        "drive_spikes": {
            "input_type": "spikes",
            "module": "sonata",
            "input_file": str(spikes_file),
            "node_set": {"population": "drive"},
        }
    }
    (root / "config.json").write_text(json.dumps(config))


def test_a_driving_population_is_measured_from_its_input_file(tmp_path):
    """Its spikes are what drives the network. Reported as 0 Hz, they read as a dead
    input, which is the opposite of the truth."""
    config, output = _build(tmp_path, n_neurons=4, spike_times=np.linspace(1.0, 999.0, 40))
    _add_driving_population(tmp_path, n_neurons=10, n_spikes=300)

    rates = _node(tmp_path)._measure_firing_rates(config, output)

    assert rates["popA"] == pytest.approx(10.0)
    assert rates["drive"] == pytest.approx(30.0)   # 300 spikes / (10 cells x 1 s)


def test_the_driving_population_also_gets_isi_and_a_rate_curve(tmp_path):
    """Measured like any other population, not a special case."""
    config, output = _build(tmp_path, n_neurons=4, spike_times=np.linspace(1.0, 999.0, 40))
    _add_driving_population(tmp_path, n_neurons=10, n_spikes=300)
    node = _node(tmp_path)

    isi = node._measure_isi_stats(output, config)
    curve = node._measure_rate_over_time(config, output, 10.0)

    assert isi["drive"]["n_intervals"] > 0
    assert np.mean(curve["drive"]["rate_hz"]) == pytest.approx(30.0)


def test_without_a_config_only_the_output_file_is_read(tmp_path):
    """The call shape older code uses must behave exactly as it did before."""
    config, output = _build(tmp_path, n_neurons=4, spike_times=np.linspace(1.0, 999.0, 40))
    _add_driving_population(tmp_path, n_neurons=10, n_spikes=300)
    node = _node(tmp_path)

    trains_without = node._spike_trains(output)
    trains_with = node._spike_trains(output, config)

    assert set(trains_without) == {"popA"}
    assert set(trains_with) == {"popA", "drive"}
    assert node._measure_isi_stats(output).get("drive", {}).get("n_intervals") == 0


def test_a_current_clamp_input_is_not_read_as_spikes(tmp_path):
    """Every existing clamp workflow has an "inputs" section too. It holds no spike
    file, and trying to open it as one would break those workflows."""
    import json

    config, output = _build(tmp_path, n_neurons=4, spike_times=np.linspace(1.0, 999.0, 40))
    conf = json.loads((tmp_path / "config.json").read_text())
    conf["inputs"] = {
        "current_clamp": {
            "input_type": "current_clamp", "module": "IClamp",
            "node_set": "all", "amp": 376.0, "delay": 500.0, "duration": 2000.0,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(conf))

    assert set(_node(tmp_path)._spike_trains(output, config)) == {"popA"}


def test_the_summary_prints_the_measured_values(tmp_path, capsys):
    """The values live on ports; without this print a run displays only figures."""
    _node(tmp_path)._print_summary(
        {"popA": 12.5},
        {"popA": {"mean_ms": 80.0, "std_ms": 40.0, "cv": 0.5, "n_intervals": 24}},
        {"popA": {"t_ms": [5.0, 15.0], "rate_hz": [10.0, 15.0], "bin_ms": 10.0}},
    )
    printed = capsys.readouterr().out

    assert "12.50 Hz" in printed
    assert "80.0 ms" in printed and "0.50" in printed
    assert "15.00 Hz" in printed


def test_the_summary_says_so_when_nothing_was_measured(tmp_path, capsys):
    """Silence after a failed measurement looks like a node that did not run."""
    _node(tmp_path)._print_summary({}, {}, {})
    assert "no values measured" in capsys.readouterr().out
