"""Every connection_rule form the node documents must actually work.

The examples are read out of `NW_Connectivity`'s own parameter descriptions rather than
copied here, so documenting a rule the coercion rejects fails this test. That is the point:
the node's documentation is a promise, and `connection_rule` reaches the node as text from
the GUI (JSON cannot carry a function), so the text path has to honour every form the
description advertises.
"""

import re

import pytest

from neuroworkflow.nodes.network.NW_Connectivity import NW_Connectivity

np = pytest.importorskip("numpy")

SRC = {"node_id": 5, "ei_type": "exc", "pop_name": "exc"}
TGT = {"node_id": 6, "ei_type": "inh", "pop_name": "inh"}


def _node_level_examples():
    """`connection_rule = <form>` lines in the connection_rule description."""
    text = NW_Connectivity.NODE_DEFINITION.parameters["connection_rule"].description
    return [
        m.group(1).strip()
        for m in (
            re.match(r"connection_rule\s*=\s*(.+)$", line.strip())
            for line in text.splitlines()
        )
        if m
    ]


def _per_connection_examples():
    """`"connection_rule": "<form>"` entries in the connections description."""
    text = NW_Connectivity.NODE_DEFINITION.parameters["connections"].description
    return re.findall(r'"connection_rule":\s*"([^"]+)"', text)


DOCUMENTED = _node_level_examples() + _per_connection_examples()


def test_the_descriptions_still_carry_examples():
    """Guard against the extraction silently matching nothing."""
    assert len(_node_level_examples()) >= 7
    assert len(_per_connection_examples()) >= 3


@pytest.mark.parametrize("form", DOCUMENTED)
def test_every_documented_form_coerces_and_runs(form):
    """Sweep pairs, not one pair.

    BMTK calls a rule once per source→target pair, so it has to hold for every pair the
    populations produce — including the diagonal, where a distance of 0 breaks anything
    that divides by it. Checking a single pair would miss exactly that.
    """
    rule = NW_Connectivity._coerce_connection_rule(form)

    if not callable(rule):
        # An integer literal is passed through as BMTK's fixed synapse count.
        assert isinstance(rule, int)
        return

    for src_id in range(12):
        for tgt_id in range(12):
            src = {"node_id": src_id, "ei_type": "exc", "pop_name": "exc"}
            tgt = {"node_id": tgt_id, "ei_type": "inh", "pop_name": "inh"}
            try:
                count = rule(src, tgt)
            except Exception as exc:  # noqa: BLE001 - the failure is the finding
                pytest.fail(f"{src_id}->{tgt_id} raised {type(exc).__name__}: {exc}")
            assert isinstance(count, (int, np.integer)) and not isinstance(
                count, bool
            ), (
                f"{src_id}->{tgt_id} returned {count!r}; BMTK needs a synapse count, and "
                f"a float silently produces no connections"
            )
            assert count >= 0, f"{src_id}->{tgt_id} returned a negative count {count!r}"


def test_a_real_callable_passes_through():
    """The notebook path: a genuine lambda, not text."""
    rule = NW_Connectivity._coerce_connection_rule(lambda src, tgt: 1)
    assert rule(SRC, TGT) == 1


def test_an_integer_and_a_numeric_string_are_synapse_counts():
    assert NW_Connectivity._coerce_connection_rule(3) == 3
    assert NW_Connectivity._coerce_connection_rule("3") == 3
