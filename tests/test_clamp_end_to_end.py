"""Clamp targeting through a real simulation, from one neuron to a balanced network.

tests/test_clamp_targeting.py checks the config entries; these check what NEST
actually does with them, because an entry can be well-formed and still drive the
wrong cells. Each case asserts both halves of the claim: the clamped population
fires, and an unclamped one wired at weight zero stays silent.

The whole file simulates in well under a second - these are point neurons over one
second of model time - but it needs NEST, so it skips where NEST is absent.
"""

import h5py
import numpy as np
import pytest

pytest.importorskip("nest")

from neuroworkflow.nodes.network.NW_Connectivity import NW_Connectivity
from neuroworkflow.nodes.network.NW_Population import NW_Population
from neuroworkflow.nodes.simulation.NW_SimConfig import NW_SimConfig
from neuroworkflow.nodes.stimulus.NW_IClamp import NW_IClamp

NEST_PARAMS = {"C_m": 250.0, "tau_m": 10.0, "t_ref": 2.0,
               "V_th": -55.0, "V_reset": -70.0, "E_L": -70.0}
TSTOP = 1000.0
#: comfortably above the ~376 pA rheobase of the neuron above
AMP = 500.0


def _simulate(tmp_path, specs, connections=None, amps=None):
    """specs: [(pop_name, N, clamp_id_or_False)].

    A clamp id shared by two populations means ONE NW_IClamp node wired to both,
    which is how the GUI expresses "this stimulus drives these two populations".
    """
    amps = amps or {}
    clamps, members, sizes = {}, [], {}

    for pop_name, n, clamp_id in specs:
        pop = NW_Population(pop_name)
        pop._context["results_path"] = str(tmp_path)
        pop.configure(pop_name=pop_name, N=n,
                      model_template="nest:iaf_psc_alpha", nest_params=NEST_PARAMS)
        iclamp = None
        if clamp_id:
            if clamp_id not in clamps:
                clamp = NW_IClamp(f"clamp_{clamp_id}")
                clamp._context["results_path"] = str(tmp_path)
                clamp.configure(amp_na=amps.get(clamp_id, AMP), delay_ms=50.0,
                                duration_ms=TSTOP - 100.0)
                clamps[clamp_id] = clamp.build()["iclamp"]
            iclamp = clamps[clamp_id]
        members.append(pop.build(iclamp=iclamp)["population"])
        sizes[pop_name] = n

    cfg = NW_SimConfig("sim")
    cfg._context["results_path"] = str(tmp_path)
    cfg.configure(simulator="pointnet", config_file="config.json", tstop_ms=TSTOP,
                  dt_ms=0.1, compile_mechanisms=False, overwrite=True, reports={})

    if connections is None:
        payload = members[0]            # no connectivity node in the graph at all
    else:
        conn = NW_Connectivity("conn")
        conn._context["results_path"] = str(tmp_path)
        conn.configure(connections=connections, connection_rule=1,
                       syn_weight=0.0, delay=1.5)
        payload = conn.connect(members)["network"]

    cfg.run(**cfg.setup(payload))

    with h5py.File(tmp_path / "output" / "spikes.h5") as f:
        return {
            pop: (len(f[f"spikes/{pop}/timestamps"]) / (n * TSTOP / 1000.0)
                  if pop in f["spikes"] else 0.0)
            for pop, n in sizes.items()
        }


def _unconnected(*names):
    """Every pair joined at weight zero: independent populations still need a
    connectivity node, because NW_SimConfig's populations port is not fan-in."""
    return [{"source": a, "target": b, "syn_weight": 0.0}
            for a in names for b in names if a != b]


def test_one_clamp_one_neuron(tmp_path):
    rates = _simulate(tmp_path, [("v1", 1, "c")])

    assert rates["v1"] > 0


def test_one_clamp_one_population(tmp_path):
    """Every neuron of the population receives it, so the mean rate matches the
    single-neuron case rather than being diluted across the population."""
    single = _simulate(tmp_path / "one", [("v1", 1, "c")])
    many = _simulate(tmp_path / "many", [("v1", 5, "c")])

    assert many["v1"] == pytest.approx(single["v1"])


def test_one_clamp_two_neurons_leaves_the_other_silent(tmp_path):
    """The graph wires the clamp to Neuron1. Before the fix it reached both."""
    rates = _simulate(tmp_path, [("Neuron1", 1, "c"), ("Neuron2", 1, False)],
                      connections=_unconnected("Neuron1", "Neuron2"))

    assert rates["Neuron1"] > 0
    assert rates["Neuron2"] == 0.0


def test_one_clamp_two_populations_leaves_the_other_silent(tmp_path):
    rates = _simulate(tmp_path, [("Neuron1", 5, "c"), ("Neuron2", 5, False)],
                      connections=_unconnected("Neuron1", "Neuron2"))

    assert rates["Neuron1"] > 0
    assert rates["Neuron2"] == 0.0


def test_the_clamped_population_may_be_wired_last(tmp_path):
    """Which population is wired first is an accident of how the graph was drawn.
    It used to decide whose clamp survived - this is the reported bug."""
    first = _simulate(tmp_path / "first", [("Neuron1", 4, "c"), ("Neuron2", 4, False)],
                      connections=_unconnected("Neuron1", "Neuron2"))
    last = _simulate(tmp_path / "last", [("Neuron2", 4, False), ("Neuron1", 4, "c")],
                     connections=_unconnected("Neuron1", "Neuron2"))

    assert first == last
    assert first["Neuron1"] > 0


def test_one_clamp_wired_to_both_populations_drives_both(tmp_path):
    """One NW_IClamp, two edges. Each population gets its own config entry."""
    rates = _simulate(tmp_path, [("Neuron1", 4, "shared"), ("Neuron2", 4, "shared")],
                      connections=_unconnected("Neuron1", "Neuron2"))

    assert rates["Neuron1"] > 0
    assert rates["Neuron2"] > 0
    assert rates["Neuron1"] == pytest.approx(rates["Neuron2"])


def test_two_clamps_drive_their_own_populations_at_their_own_amplitudes(tmp_path):
    """A second clamp used to overwrite the first, so one population ran unstimulated
    or at the other's amplitude. Different amplitudes must give different rates."""
    rates = _simulate(tmp_path,
                      [("Neuron1", 4, "weak"), ("Neuron2", 4, "strong")],
                      amps={"weak": 420.0, "strong": 700.0},
                      connections=_unconnected("Neuron1", "Neuron2"))

    assert 0 < rates["Neuron1"] < rates["Neuron2"]
