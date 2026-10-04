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
