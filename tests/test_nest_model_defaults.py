"""Every entry in NEST_MODEL_DEFAULTS must be a dictionary NEST will actually take.

These defaults are written into the model JSON for every population, so a key NEST
does not recognise is not a typo that lints away - it makes that model unusable, and
the failure appears inside the simulator long after the node ran. "G" sat in the
glif_cond entry doing exactly that: NEST rejects a dictionary containing an unknown
key outright, so glif_cond could never be simulated.

Skipped where NEST is not installed, which is the only way to answer the question.
"""

import pytest

nest = pytest.importorskip("nest")

from neuroworkflow.nodes.network.NW_Population import NEST_MODEL_DEFAULTS


@pytest.mark.parametrize("model", sorted(NEST_MODEL_DEFAULTS))
def test_nest_accepts_the_declared_defaults(model):
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()

    nest.Create(model, params=NEST_MODEL_DEFAULTS[model])


@pytest.mark.parametrize("model", sorted(NEST_MODEL_DEFAULTS))
def test_declared_values_match_nest_where_they_are_meant_to_be_neutral(model):
    """I_e is listed only so it shows up in the node panel and can be optimized. It
    must equal NEST's own default, or adding it would silently change every existing
    simulation that uses the model."""
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    declared = NEST_MODEL_DEFAULTS[model]

    if "I_e" not in declared:
        pytest.skip(f"{model} has no I_e")

    assert declared["I_e"] == nest.GetDefaults(model)["I_e"]
