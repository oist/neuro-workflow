"""Tests for the shared text-to-callable helper.

The guard exists because a callable parameter arrives from the GUI as text, so the two
things that matter are: a rule someone would reasonably write must work, and a rule that
reaches outside the expression must be refused — with the refusal happening when the text
is compiled, not thousands of calls later while a network is being built.
"""

import pytest

from neuroworkflow.utils.safe_callable import safe_callable

SRC = {"node_id": 5, "ei_type": "exc", "pop_name": "exc"}
TGT = {"node_id": 6, "ei_type": "inh", "pop_name": "inh"}

ACCEPTED = [
    # plain lambdas, with and without the modules a scientific rule reaches for
    "lambda src, tgt: 1 if src['node_id'] != tgt['node_id'] else 0",
    "lambda src, tgt: 1 if abs(src['node_id'] - tgt['node_id']) <= 2 else 0",
    "lambda src, tgt: 1 if math.exp(-1.0) > 0.3 else 0",
    "lambda src, tgt: 1 if random.random() < 0.5 else 0",
    "lambda src, tgt: 1 if statistics.mean([src['node_id'], tgt['node_id']]) > 4 else 0",
    # builtins the old 12-name allowlist refused
    "lambda src, tgt: 1 if bool(src['node_id'] % 2) else 0",
    "lambda src, tgt: 1 if str(src['pop_name']) == 'exc' else 0",
    "lambda src, tgt: sum(w for w in [1, 0, 1])",
    "lambda src, tgt: 1 if any([src['node_id'] > 3]) else 0",
    "lambda src, tgt: 1 if pow(src['node_id'], 2) > 9 else 0",
    # shapes other than a bare lambda
    "functools.partial(lambda a, s, t: a, 1)",
    "(lambda s, t: 1) if True else (lambda s, t: 0)",
]

REFUSED = [
    # the standard escape from a restricted eval: no builtin needed, just attributes
    "lambda src, tgt: ().__class__.__bases__[0].__subclasses__()[0].__name__",
    "lambda src, tgt: __import__('os').system('echo pwned')",
    "lambda src, tgt: eval('1+1')",
    "lambda src, tgt: exec('x=1')",
    "lambda src, tgt: open('/etc/passwd').read()",
    "lambda src, tgt: getattr(src, 'keys')()",
    "lambda src, tgt: globals()",
    "lambda src, tgt: type(1).__mro__",
    # a name nothing provides: caught now, not at call time
    "lambda src, tgt: 1 if undefined_thing else 0",
    # produces a value, not a callable
    "42.5",
    # not an expression at all
    "def rule(src, tgt): return 1",
]


@pytest.mark.parametrize("text", ACCEPTED)
def test_reasonable_rules_are_accepted_and_callable(text):
    rule = safe_callable(text, what="connection_rule")
    assert callable(rule)
    assert isinstance(rule(SRC, TGT), int)


@pytest.mark.parametrize("text", REFUSED)
def test_rules_reaching_outside_the_expression_are_refused(text):
    with pytest.raises(ValueError):
        safe_callable(text, what="connection_rule")


def test_numpy_is_available_when_installed():
    pytest.importorskip("numpy")
    rule = safe_callable("lambda src, tgt: int(np.random.rand() < 1.0)")
    assert rule(SRC, TGT) == 1


def test_a_comprehension_variable_is_not_mistaken_for_a_missing_name():
    """`w` is bound by the comprehension; without that, this looks like a typo."""
    assert safe_callable("lambda src, tgt: sum(w for w in [1, 0, 1])")(SRC, TGT) == 2


def test_extra_globals_are_offered_to_the_expression():
    rule = safe_callable(
        "lambda src, tgt: kernel(src['node_id'])",
        extra_globals={"kernel": lambda n: n * 2},
    )
    assert rule(SRC, TGT) == 10


def test_an_unavailable_name_is_reported_when_the_text_is_compiled():
    """The message must name the value and say what it could have used instead."""
    with pytest.raises(ValueError) as caught:
        safe_callable("lambda src, tgt: helper(src)", what="connection_rule")
    message = str(caught.value)
    assert "connection_rule" in message
    assert "helper" in message
    assert "math" in message  # lists what is available


def test_a_name_missing_from_the_outer_expression_is_still_a_compile_time_error():
    """`helper` is bound by one lambda's arguments, so the name check lets it through;
    the outer conditional still has no `helper`. That must surface as the documented
    ValueError, not as a NameError escaping from eval."""
    with pytest.raises(ValueError, match="helper"):
        safe_callable("(lambda helper: 1) if helper else (lambda src, tgt: 0)")
