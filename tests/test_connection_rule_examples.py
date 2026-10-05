"""Every connection rule shown in NW_Connectivity's documentation must actually run.

These examples are written to be copied into the GUI's parameter field, so a typo in
one of them is not a documentation slip - it is a rule a user pastes in and a network
that fails, or worse, builds the wrong wiring. The examples reach the workflow by two
different routes and both have to work:

  * set on the node itself, the code generator emits the lambda bare into the
    generated script (code_generation_service.py: ``if trimmed.startswith("lambda ")``),
    where it is ordinary Python and ``np`` comes from the script's own import;
  * set inside a ``connections`` entry, it stays a string and is compiled by
    ``safe_callable``, which supplies np, math, random, statistics and functools.

So the test reads the examples out of the live description rather than repeating them,
and checks both routes. Adding an example that does not run will fail here.
"""

import numpy as np
import pytest

from neuroworkflow.nodes.network.NW_Connectivity import NW_Connectivity

SRC = {"node_id": 3, "ei_type": "exc", "pop_name": "src"}
TGT = {"node_id": 7, "ei_type": "inh", "pop_name": "tgt"}


def _documented_rules():
    """The right-hand side of every 'connection_rule = ...' line in the description."""
    description = NW_Connectivity.NODE_DEFINITION.parameters["connection_rule"].description
    return [
        line.strip().split("=", 1)[1].strip()
        for line in description.splitlines()
        if line.strip().startswith("connection_rule =")
    ]


def test_the_description_actually_contains_examples():
    """If the extraction silently found nothing, every test below would pass vacuously."""
    assert len(_documented_rules()) >= 10


@pytest.mark.parametrize("rule_text", _documented_rules())
def test_the_rule_compiles_and_returns_a_synapse_count(rule_text):
    """The route taken when the rule sits inside a connections entry."""
    rule = NW_Connectivity._coerce_connection_rule(rule_text)

    count = rule if isinstance(rule, int) else rule(SRC, TGT)

    assert isinstance(count, (int, np.integer))
    assert count >= 0


@pytest.mark.parametrize("rule_text", _documented_rules())
def test_the_rule_also_works_emitted_bare_into_a_script(rule_text):
    """The route taken when the rule is set on the node: the generator writes it
    unquoted, so it runs as plain Python with numpy imported as np."""
    obj = eval(rule_text, {"np": np})

    count = obj if isinstance(obj, int) else obj(SRC, TGT)

    assert count >= 0


def test_a_float_is_still_refused():
    """0.1 looks like a probability but BMTK reads it as a synapse count and rounds to
    zero, building nothing. The description warns about it; this keeps the warning true."""
    with pytest.raises(ValueError, match="connection_rule"):
        NW_Connectivity._coerce_connection_rule(0.1)
