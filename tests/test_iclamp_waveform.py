"""A current clamp must be able to carry an arbitrary waveform, not only a step.

BMTK reads a waveform only from the config's ``inputs`` section. Its helper,
``create_environment()``, writes that case as module ``FileIClamp`` - a module
neither PointNet nor BioNet dispatches, so the stimulus is dropped with a log
line and the simulation runs unstimulated. The combination both simulators do
read is module ``IClamp`` with ``input_type: csv``.

So NW_IClamp emits two shapes: the amp/delay/duration dict that has always gone
to ``create_environment()``, and - only when a waveform file is set - a complete
config entry. These tests pin both, because the step shape reaching the wrong
path would silently change every existing network.
"""

import pytest

from neuroworkflow.nodes.stimulus.NW_IClamp import NW_IClamp


def test_without_a_waveform_the_step_shape_is_unchanged():
    """Existing workflows send this dict to create_environment(); it must not gain
    a 'module' key, which is what routes a clamp away from that path."""
    clamp = NW_IClamp("clamp")
    clamp.configure(amp_na=376.0, delay_ms=500.0, duration_ms=2000.0)

    entry = clamp.build()["iclamp"]

    assert entry["amp"] == 376.0
    assert entry["delay"] == 500.0
    assert entry["duration"] == 2000.0
    assert "module" not in entry


def test_a_waveform_produces_the_entry_both_simulators_dispatch(tmp_path):
    """module IClamp + input_type csv. FileIClamp would be silently ignored."""
    csv = tmp_path / "ramp.csv"
    csv.write_text("timestamps amps\n0.0 0.0\n10.0 100.0\n")

    clamp = NW_IClamp("clamp")
    clamp.configure(waveform_csv=str(csv), node_set="v1")
    entry = clamp.build()["iclamp"]

    assert entry["module"] == "IClamp"
    assert entry["input_type"] == "csv"
    assert entry["node_set"] == "v1"
    # BMTK's CSVAmpReader reads args['file']; 'input_file' is a different key and
    # would raise at simulation time, long after the submit.
    assert entry["file"] == str(csv.resolve())
    assert "amp" not in entry


def test_a_missing_waveform_file_fails_at_build_not_at_simulation(tmp_path):
    """Found when the node runs, not twenty minutes into a cluster job."""
    clamp = NW_IClamp("clamp")
    clamp.configure(waveform_csv=str(tmp_path / "absent.csv"))

    with pytest.raises(ValueError, match="waveform_csv not found"):
        clamp.build()


def test_the_amplitude_default_can_make_a_default_nest_cell_fire():
    """NEST reads this as picoamperes, unscaled. The old default of 0.15 was
    roughly 2500x below the current an iaf_psc_alpha needs to spike."""
    amp = NW_IClamp.NODE_DEFINITION.parameters["amp_na"].default_value
    assert amp >= 376.0
