"""A current clamp must drive the population the graph wires it to - only that one.

A clamp carries no record of its target: two clamps on two populations produce
identical dictionaries, so the only thing linking one to a population is the
population dict holding it. NW_SimConfig used to read that link out into a bare
variable from ``pop_list[0]`` and lose it, which produced three failures at once:

  * a clamp on any population other than the first was silently dropped, so the
    physics depended on the order the graph's edges happened to be drawn;
  * a second clamp overwrote the first, because an anonymous clamp can only be
    written under one fixed config key;
  * the target fell back to every population in the network - including virtual
    spike sources, which have no membrane and make NEST fail.

These tests work on the config entries rather than running a simulator, because
that is where the association is either preserved or lost.
"""

import pytest

from neuroworkflow.nodes.simulation.NW_SimConfig import NW_SimConfig
from neuroworkflow.nodes.stimulus.NW_IClamp import NW_IClamp

STEP = {"amp": 500.0, "delay": 50.0, "duration": 900.0}


def _pop(name, clamp=None):
    pop = {"pop_name": name}
    if clamp is not None:
        pop["_current_clamp"] = clamp
    return pop


def _collect(pop_list):
    """What setup() writes into the config's "inputs" section."""
    entries = {}
    for pop in pop_list:
        entries.update(NW_SimConfig._clamp_entry(pop))
    return entries


def test_the_clamp_targets_its_own_population_not_every_one():
    """The graph says "clamp -> Neuron1", so Neuron2 must be left alone."""
    entries = _collect([_pop("Neuron1", STEP), _pop("Neuron2")])

    assert list(entries) == ["current_clamp_Neuron1"]
    assert entries["current_clamp_Neuron1"]["node_set"] == "Neuron1"


@pytest.mark.parametrize("order", [("Neuron1", "Neuron2"), ("Neuron2", "Neuron1")])
def test_the_result_does_not_depend_on_the_order_of_the_populations(order):
    """The order is whichever edge was drawn first in the GUI. It must not reach
    the simulation. This is the regression that prompted the fix."""
    by_name = {"Neuron1": _pop("Neuron1", STEP), "Neuron2": _pop("Neuron2")}

    entries = _collect([by_name[n] for n in order])

    assert entries["current_clamp_Neuron1"]["node_set"] == "Neuron1"
    assert entries["current_clamp_Neuron1"]["amp"] == 500.0


def test_two_clamps_both_survive():
    """One fixed config key meant the second clamp replaced the first, and the
    simulation ran with one of the two stimuli the user asked for."""
    entries = _collect([
        _pop("Neuron1", {"amp": 500.0, "delay": 50.0, "duration": 900.0}),
        _pop("Neuron2", {"amp": 600.0, "delay": 50.0, "duration": 900.0}),
    ])

    assert sorted(entries) == ["current_clamp_Neuron1", "current_clamp_Neuron2"]
    assert entries["current_clamp_Neuron1"]["amp"] == 500.0
    assert entries["current_clamp_Neuron2"]["amp"] == 600.0


def test_one_clamp_wired_to_two_populations_drives_both_separately():
    """NW_Population hands every target the SAME dict object, so completing it in
    place would make the second population overwrite the first one's target."""
    shared = dict(STEP)

    entries = _collect([_pop("Neuron1", shared), _pop("Neuron2", shared)])

    assert entries["current_clamp_Neuron1"]["node_set"] == "Neuron1"
    assert entries["current_clamp_Neuron2"]["node_set"] == "Neuron2"
    assert "node_set" not in shared, "the clamp the populations share was mutated"


def test_an_explicit_target_is_respected():
    """Reaching part of a population stays possible."""
    entries = _collect([_pop("Neuron1", {**STEP, "node_set": "some_node_set"})])

    assert entries["current_clamp_Neuron1"]["node_set"] == "some_node_set"


def test_a_waveform_clamp_is_targeted_the_same_way():
    """The csv clamp already arrives as a config entry; it still needs its target."""
    waveform = {"input_type": "csv", "module": "IClamp", "file": "/tmp/w.csv"}

    entries = _collect([_pop("Neuron1", waveform), _pop("Neuron2")])

    assert entries["current_clamp_Neuron1"]["node_set"] == "Neuron1"
    assert entries["current_clamp_Neuron1"]["module"] == "IClamp"


def test_a_step_clamp_is_given_the_keys_bmtk_dispatches_on():
    """It arrives in create_environment()'s argument shape, which carries neither."""
    entry = _collect([_pop("Neuron1", STEP)])["current_clamp_Neuron1"]

    assert entry["module"] == "IClamp"
    assert entry["input_type"] == "current_clamp"


def test_a_population_with_no_clamp_contributes_nothing():
    assert _collect([_pop("Neuron1"), _pop("Neuron2")]) == {}


def test_the_clamp_node_leaves_the_target_open_by_default():
    """NW_IClamp cannot know which population it will be wired to, so it must not
    claim one. An omitted key is what lets NW_SimConfig fill it in."""
    assert "node_set" not in NW_IClamp("c").build()["iclamp"]
