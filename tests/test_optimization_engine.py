"""Toy-node tests for the outer-loop optimization engine (no NEST)."""

import functools
import math
import sys
import types

import pytest

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)
from neuroworkflow.core.workflow import Workflow
from neuroworkflow.nodes.network.NW_Connectivity import NW_Connectivity
from neuroworkflow.optimization.addressing import _is_number
from neuroworkflow.optimization.engine import (
    _apply_control,
    _evaluate,
    decode_candidate,
    objective_fitness,
    optimize,
    target_ranges_off,
)
from neuroworkflow.optimization.optimizers import RandomSearch, sampler_init_kwargs
from neuroworkflow.optimization.spec import (
    AlgorithmConfig,
    Dimension,
    Objective,
    OptimizationSpec,
    build_spec,
    collect_dimensions,
)


class Probe(Node):
    """Scalar probe: output y = 2 * x (or NaN / skip, for failure cases)."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="probe",
        description="Toy probe for optimization engine tests",
        parameters={
            "x": ParameterDefinition(
                default_value=1.0,
                optimizable=True,
                optimization_range=[0.0, 10.0],
            ),
            "z": ParameterDefinition(
                default_value=4.0,
                optimizable=True,
                optimization_range=[0.0, 10.0],
            ),
        },
        outputs={
            "y": PortDefinition(type=PortType.FLOAT, description="2x"),
        },
        methods={
            "run": MethodDefinition(description="emit y", outputs=["y"]),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, outputs=["y"])
        self.skip_emit = False
        self.emit_nan = False

    def run(self):
        if self.skip_emit:
            return {}
        value = float(self._parameters["x"]) * 2.0
        if self.emit_nan:
            value = float("nan")
        return {"y": value}


def _spec():
    return OptimizationSpec(
        dimensions=[
            Dimension(address="Probe.x", low=0.0, high=10.0),
            Dimension(address="Probe.z", low=0.0, high=10.0),
        ],
        objectives=[
            Objective(
                name="y",
                measures="Probe.y",
                low=0.0,
                high=20.0,
                goal="in_range",
            )
        ],
        algorithm=AlgorithmConfig(name="random", pop_size=2, max_generations=2, seed=0),
    )


def _workflow():
    probe = Probe("Probe")
    return Workflow("toy", {"Probe": probe}, []), probe


def test_random_optimize_round_trip(tmp_path):
    workflow, _ = _workflow()
    result = optimize(
        workflow,
        spec=_spec(),
        results_path=str(tmp_path),
        per_trial_results=False,
        stop_when_reached=False,
        verbose=False,
        run_id="roundtrip",
    )
    assert result.stop_reason
    assert len(result.trials) == 4
    assert all(t["status"] == "ok" for t in result.trials)
    assert result.best is not None


def test_nan_measurement_is_failed_trial():
    workflow, probe = _workflow()
    probe.emit_nan = True
    row = _evaluate(workflow, _spec(), {"Probe.x": 1.0, "Probe.z": 4.0}, None, 1, 1, 0)
    assert row["status"] == "failed"
    assert row["fitness"] is None
    with pytest.raises(ValueError, match="non-finite"):
        objective_fitness(_spec().objectives[0], float("nan"))


def test_partial_inject_completes_dims_and_continues():
    workflow, probe = _workflow()
    spec = _spec()
    optimizer = RandomSearch(spec.dimensions, 1, spec.algorithm)
    probe.configure(x=1.0, z=4.0)

    ack, stop, paused = _apply_control(
        {
            "command": "inject",
            "seq": 1,
            "args": {"candidates": [{"Probe.x": 3.0}]},
        },
        optimizer,
        lambda _m: None,
        "",
        False,
        workflow=workflow,
        spec=spec,
    )
    assert stop == ""
    assert "injected" in ack["result"]

    candidates = optimizer.ask()
    assert optimizer.last_ask_sources[0] == "inject"
    assert candidates[0]["Probe.x"] == 3.0
    assert candidates[0]["Probe.z"] == 4.0

    row = _evaluate(workflow, spec, candidates[0], None, 1, 1, 0, source="inject")
    assert row["source"] == "inject"
    assert row["status"] == "ok"
    assert row["params"]["Probe.z"] == 4.0


def test_stale_output_does_not_leak_between_trials():
    workflow, probe = _workflow()
    spec = _spec()
    first = _evaluate(workflow, spec, {"Probe.x": 2.0, "Probe.z": 4.0}, None, 1, 1, 0)
    assert first["status"] == "ok"
    assert first["measured"]["y"] == 4.0

    probe.skip_emit = True
    second = _evaluate(workflow, spec, {"Probe.x": 5.0, "Probe.z": 4.0}, None, 2, 1, 1)
    assert second["status"] == "failed"
    assert second["measured"] == {}
    assert "not numeric" in (second["error"] or "")


def test_is_number_numpy_bool_nan():
    assert _is_number(1) is True
    assert _is_number(1.5) is True
    assert _is_number(True) is False
    assert _is_number(float("nan")) is False
    numpy = pytest.importorskip("numpy")
    assert _is_number(numpy.float32(1.25)) is True
    assert _is_number(numpy.int64(3)) is True
    assert _is_number(numpy.array([1.0, 2.0])) is False


def _import_simconfig():
    if "pandas" not in sys.modules:
        fake = types.ModuleType("pandas")
        fake.set_option = lambda *args, **kwargs: None
        sys.modules["pandas"] = fake
    from neuroworkflow.nodes.simulation.NW_SimConfig import NW_SimConfig

    return NW_SimConfig


def test_stable_partial_does_not_raise():
    NW_SimConfig = _import_simconfig()

    def times(n, x):
        return n * x

    wrapped = functools.partial(times, 2)
    assert NW_SimConfig._stable(wrapped)


def test_instance_node_definition_is_isolated():
    a = Probe("a")
    b = Probe("b")
    a.NODE_DEFINITION.parameters["x"].optimizable = False
    assert b.NODE_DEFINITION.parameters["x"].optimizable is True
    assert Probe.NODE_DEFINITION.parameters["x"].optimizable is True


def test_cmaes_sampler_kwargs_include_popsize():
    config = AlgorithmConfig(name="cmaes", pop_size=16, seed=1)
    cma = sampler_init_kwargs("CmaEsSampler", config)
    assert cma["popsize"] == 16
    nsga = sampler_init_kwargs("NSGAIISampler", config)
    assert nsga["population_size"] == 16
    assert "popsize" not in nsga


def test_connection_rule_accepts_any_callable_expression():
    """The form is not restricted to a lambda; producing a callable is what matters.

    A rule arrives from the GUI as text, and a one-liner should be allowed to be whatever
    expresses the rule. What is refused is code reaching outside the expression, not an
    unfamiliar shape — see tests/test_safe_callable.py for that guard in full.
    """
    assert NW_Connectivity._coerce_connection_rule("lambda src, tgt: 1")(0, 1) == 1
    assert NW_Connectivity._coerce_connection_rule(3) == 3
    assert NW_Connectivity._coerce_connection_rule("3") == 3
    # a shape other than a bare lambda
    partial_rule = NW_Connectivity._coerce_connection_rule(
        "functools.partial(lambda a, s, t: a, 1)"
    )
    assert partial_rule(0, 1) == 1


def test_connection_rule_refuses_code_reaching_outside_itself():
    with pytest.raises(ValueError, match="__import__"):
        NW_Connectivity._coerce_connection_rule("__import__('os').system('x')")
    with pytest.raises(ValueError, match="not available"):
        NW_Connectivity._coerce_connection_rule("src + tgt")
    with pytest.raises(ValueError, match="did not produce a callable"):
        NW_Connectivity._coerce_connection_rule("42.5")


# ---------------------------------------------------------------------------
# Integer axes: a count of neurons is not a continuous quantity
# ---------------------------------------------------------------------------


class Counter(Node):
    """Emits its own neuron count, the way a population node builds int(N) cells."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="counter",
        description="Toy node with an integer parameter",
        parameters={
            "N": ParameterDefinition(
                default_value=2500,
                optimizable=True,
                optimization_range=[2000.0, 3000.0],
            ),
            "amp": ParameterDefinition(
                default_value=1.5,
                optimizable=True,
                optimization_range=[0.0, 5.0],
            ),
        },
        outputs={"built": PortDefinition(type=PortType.FLOAT, description="cells")},
        methods={"run": MethodDefinition(description="build", outputs=["built"])},
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, outputs=["built"])

    def run(self):
        # Exactly what NW_Population does with N.
        return {"built": float(int(self._parameters["N"]))}


def _counter_workflow():
    node = Counter("Counter")
    return Workflow("toy", {"Counter": node}, []), node


def test_integer_axis_inferred_from_the_current_value():
    workflow, _ = _counter_workflow()
    dimensions, skipped = collect_dimensions(workflow)
    by_address = {d.address: d for d in dimensions}
    assert by_address["Counter.N"].integer is True
    assert by_address["Counter.amp"].integer is False
    assert skipped == []


def test_integer_axis_can_be_declared_when_the_value_is_a_float():
    node = Counter("Counter")
    node.NODE_DEFINITION.parameters["amp"].constraints = {"integer": True}
    workflow = Workflow("toy", {"Counter": node}, [])
    dimensions, _ = collect_dimensions(workflow)
    assert {d.address: d.integer for d in dimensions}["Counter.amp"] is True


def test_decode_rounds_integer_axes_and_leaves_floats_alone():
    spec = OptimizationSpec(
        dimensions=[
            Dimension(address="Counter.N", low=2000.0, high=3000.0, integer=True),
            Dimension(address="Counter.amp", low=0.0, high=5.0),
        ],
        objectives=[],
        algorithm=AlgorithmConfig(name="random"),
    )
    decoded = decode_candidate(spec, {"Counter.N": 2500.37, "Counter.amp": 1.2345})
    assert decoded["Counter.N"] == 2500
    assert isinstance(decoded["Counter.N"], int)
    assert decoded["Counter.amp"] == 1.2345


def test_decode_clamps_inside_the_declared_range():
    """Rounding must not push a proposal past a bound configure() would reject."""
    spec = OptimizationSpec(
        dimensions=[Dimension(address="Counter.N", low=2.2, high=6.8, integer=True)],
        objectives=[],
        algorithm=AlgorithmConfig(name="random"),
    )
    assert decode_candidate(spec, {"Counter.N": 2.3})["Counter.N"] == 3
    assert decode_candidate(spec, {"Counter.N": 6.7})["Counter.N"] == 6


def test_integer_axis_with_no_whole_number_is_skipped():
    node = Counter("Counter")
    node.NODE_DEFINITION.parameters["N"].optimization_range = [10.2, 10.8]
    workflow = Workflow("toy", {"Counter": node}, [])
    dimensions, skipped = collect_dimensions(workflow)
    assert "Counter.N" not in {d.address for d in dimensions}
    assert any("no whole number" in s["reason"] for s in skipped)


def test_ledger_records_the_integer_that_actually_ran(tmp_path):
    workflow, _ = _counter_workflow()
    spec = OptimizationSpec(
        dimensions=[
            Dimension(address="Counter.N", low=2000.0, high=3000.0, integer=True)
        ],
        objectives=[
            Objective(name="built", measures="Counter.built", low=2400.0, high=2600.0)
        ],
        algorithm=AlgorithmConfig(name="random", pop_size=3, max_generations=1, seed=4),
    )
    result = optimize(
        workflow,
        spec=spec,
        results_path=str(tmp_path),
        per_trial_results=False,
        verbose=False,
    )
    assert result.trials, "no trial ran"
    for trial in result.trials:
        recorded = trial["params"]["Counter.N"]
        assert isinstance(recorded, int)
        # The node reports what it built; it must equal what the ledger claims.
        assert trial["measured"]["built"] == float(recorded)
    assert "N=" in result.configure_snippet()
    assert ".37" not in result.configure_snippet()


def test_integer_inference_ignores_a_user_typed_int_literal():
    """A continuous quantity configured as `200` must not become integer-only.

    The node author declared 0.15 nA; that a user typed an int literal says nothing
    about the parameter's nature, and inferring from it would silently restrict the
    search to whole nanoamps.
    """
    node = Counter("Counter")
    node.NODE_DEFINITION.parameters["amp"].default_value = 0.15
    node.configure(amp=200)
    workflow = Workflow("toy", {"Counter": node}, [])
    dimensions, _ = collect_dimensions(workflow)
    assert {d.address: d.integer for d in dimensions}["Counter.amp"] is False


class Dynamics(Node):
    """On-target rate with pathological regularity — the case reject_fn exists for."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="dynamics",
        description="Toy analysis node with an objective and a dynamics signal",
        parameters={
            "x": ParameterDefinition(
                default_value=1.0,
                optimizable=True,
                optimization_range=[0.0, 10.0],
            )
        },
        outputs={
            "rate_hz": PortDefinition(type=PortType.DICT, description="rate"),
            "isi_stats": PortDefinition(type=PortType.DICT, description="regularity"),
        },
        methods={
            "run": MethodDefinition(
                description="emit", outputs=["rate_hz", "isi_stats"]
            )
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, outputs=["rate_hz", "isi_stats"])

    def run(self):
        return {"rate_hz": {"exc": 10.0}, "isi_stats": {"exc": {"cv": 0.02}}}


def test_reject_fn_receives_measurements_beyond_the_objectives(tmp_path):
    """The dynamics check must see what the objective does not capture."""
    node = Dynamics("Ana")
    workflow = Workflow("toy", {"Ana": node}, [])
    spec = OptimizationSpec(
        dimensions=[Dimension(address="Ana.x", low=0.0, high=10.0)],
        objectives=[
            Objective(name="rate", measures="Ana.rate_hz.exc", low=8.0, high=12.0)
        ],
        algorithm=AlgorithmConfig(name="random", pop_size=2, max_generations=1, seed=0),
    )

    seen = {}

    def reject(_workflow, measured):
        seen["keys"] = set(measured)
        cv = measured.get("Ana.isi_stats.exc.cv")
        return f"clock-like firing (CV {cv})" if cv is not None and cv < 0.1 else None

    result = optimize(
        workflow,
        spec=spec,
        results_path=str(tmp_path),
        reject_fn=reject,
        per_trial_results=False,
        verbose=False,
    )

    # A non-objective measurement is reachable by its address...
    assert "Ana.isi_stats.exc.cv" in seen["keys"]
    # ...and an objective is still reachable by its declared name.
    assert "rate" in seen["keys"]
    assert result.trials, "no trial ran"
    assert all(t["status"] == "rejected" for t in result.trials)
    assert "clock-like" in result.trials[0]["reject_reason"]


def test_stable_detects_a_changed_array_global():
    """A rule reading an array must not look unchanged when the array changes.

    numpy prints only the first and last few elements of anything past its print
    threshold, so recording the array with repr() made two different arrays produce the
    same signature — and `rebuild_network="auto"` then reused a network built from the
    other one, silently and with the wrong connectivity.
    """
    np = pytest.importorskip("numpy")
    NW_SimConfig = _import_simconfig()

    def rule_reading(array):
        return eval("lambda s, t: 1 if w[0] > 0 else 0", {"w": array})

    a = np.arange(5000.0)
    b = np.arange(5000.0)
    b[2500] = -1.0  # a change repr() cannot show

    assert NW_SimConfig._stable(rule_reading(a)) != NW_SimConfig._stable(
        rule_reading(b)
    )
    # An unchanged array must still match, or nothing would ever be reused.
    assert NW_SimConfig._stable(rule_reading(a)) == NW_SimConfig._stable(
        rule_reading(np.arange(5000.0))
    )

    # An array reached through a container has the same problem.
    def rule_reading_nested(array):
        return eval("lambda s, t: 1 if d['p'][0] > 0 else 0", {"d": {"p": array}})

    assert NW_SimConfig._stable(rule_reading_nested(a)) != NW_SimConfig._stable(
        rule_reading_nested(b)
    )


def test_is_number_rejects_text_but_keeps_numpy_scalars_and_ints():
    """A measurement must be a number, not text that happens to look like one.

    `float(value)` is what admits numpy scalars, and it also converts text — so a node
    emitting "8.3" would be accepted silently, and "0" used as a placeholder for missing
    data would score as a real measurement of zero. Integers stay valid: a spike count is
    a perfectly good measurement.
    """
    for text in ("8.3", "  8.3 ", "0", b"8.3"):
        assert _is_number(text) is False, text
    for number in (5, 5.0, -2):
        assert _is_number(number) is True, number

    np = pytest.importorskip("numpy")
    for scalar in (np.float64(8.3), np.int32(7), np.float32(1.5)):
        assert _is_number(scalar) is True, scalar
    assert _is_number(np.array([1.0, 2.0])) is False


def test_best_of_a_maximize_objective_is_the_largest_value():
    """Ranking must keep the sign of a minimize/maximize fitness.

    A maximize goal scores a value as its negative, so the trial measuring 50 Hz has
    fitness -50 and the one measuring 10 Hz has -10. Taking the magnitude before
    ranking would call 10 Hz the better trial.
    """
    spec = OptimizationSpec(
        dimensions=[Dimension(address="Probe.x", low=0.0, high=10.0)],
        objectives=[Objective(name="rate", measures="Probe.y", goal="maximize")],
        baseline={"measured": {"rate": 20.0}},
    )
    high = target_ranges_off(spec, [objective_fitness(spec.objectives[0], 50.0)])
    low = target_ranges_off(spec, [objective_fitness(spec.objectives[0], 10.0)])
    assert high < low

    spec.objectives[0].goal = "minimize"
    better = target_ranges_off(spec, [objective_fitness(spec.objectives[0], -10.0)])
    worse = target_ranges_off(spec, [objective_fitness(spec.objectives[0], -1.0)])
    assert better < worse

    # An in_range miss is a distance, never negative, so it ranks exactly as before.
    spec.objectives[0].goal = "in_range"
    spec.objectives[0].low, spec.objectives[0].high = 40.0, 50.0
    assert target_ranges_off(spec, [objective_fitness(spec.objectives[0], 45.0)]) == 0.0
    assert target_ranges_off(spec, [objective_fitness(spec.objectives[0], 25.0)]) == 1.5


def test_validate_refuses_an_empty_budget_and_a_reversed_band():
    """max_generations=0 would leave the loop empty and the final status write
    reading a generation that was never assigned; a reversed band can never be hit."""
    spec = _spec()
    spec.algorithm.max_generations = 0
    with pytest.raises(ValueError, match="max_generations 0"):
        spec.validate()

    spec = _spec()
    spec.algorithm.pop_size = 0
    with pytest.raises(ValueError, match="pop_size 0"):
        spec.validate()

    spec = _spec()
    spec.objectives[0].low, spec.objectives[0].high = 60.0, 40.0
    with pytest.raises(ValueError, match="low 60.0 > high 40.0"):
        spec.validate()

    spec = _spec()
    spec.objectives[0].low = spec.objectives[0].high  # a point target is allowed
    spec.validate()


# ---------------------------------------------------------------------------
# An explicit study: build_spec(explore=..., objectives=...)
# ---------------------------------------------------------------------------


class Knobs(Node):
    """One of each kind of parameter a study entry may point at."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="knobs",
        description="Toy node for explicit-study tests",
        parameters={
            "amp": ParameterDefinition(
                default_value=1.5,
                unit="pA",
                constraints={"min": 0.0, "max": 5.0},
            ),
            "N": ParameterDefinition(default_value=20),
            "label": ParameterDefinition(default_value="probe"),
            "cell": ParameterDefinition(
                default_value={"tau": 10.0, "n_syn": 3},
                unit="ms",
            ),
        },
        outputs={"y": PortDefinition(type=PortType.FLOAT, description="amp x N")},
        methods={"run": MethodDefinition(description="emit y", outputs=["y"])},
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, outputs=["y"])

    def run(self):
        return {"y": float(self._parameters["amp"]) * float(self._parameters["N"])}


def _knobs_workflow():
    knobs = Knobs("Knobs")
    probe = Probe("Probe")
    return Workflow("toy", {"Knobs": knobs, "Probe": probe}, []), knobs


def _study_spec(explore, objectives=None, run_baseline=False):
    workflow, _ = _knobs_workflow()
    return build_spec(
        workflow,
        AlgorithmConfig(name="random", pop_size=2, max_generations=1, seed=0),
        run_baseline=run_baseline,
        explore=explore,
        objectives=objectives,
    )


def test_study_dimensions_come_from_explore_entries():
    """With a study, the optimizable flags on the nodes are not consulted."""
    spec = _study_spec(
        [{"address": "Knobs.amp", "low": 1.0, "high": 3.0, "node_id": "rf-1"}]
    )
    assert [d.address for d in spec.dimensions] == ["Knobs.amp"]
    dim = spec.dimensions[0]
    assert (dim.low, dim.high, dim.unit, dim.source) == (1.0, 3.0, "pA", "study")
    assert dim.integer is False
    assert "Probe.x" not in {d.address for d in spec.dimensions}  # optimizable=True
    assert spec.skipped == []


def test_study_entry_unit_overrides_the_schema_unit():
    spec = _study_spec([{"address": "Knobs.amp", "low": 1, "high": 3, "unit": "nA"}])
    assert spec.dimensions[0].unit == "nA"


def test_study_explore_is_clipped_to_constraints():
    spec = _study_spec([{"address": "Knobs.amp", "low": -1.0, "high": 10.0}])
    dim = spec.dimensions[0]
    assert (dim.low, dim.high, dim.source) == (0.0, 5.0, "clipped")
    assert "low raised to 0.0" in dim.note and "high lowered to 5.0" in dim.note


def test_study_explore_infers_and_overrides_integer():
    spec = _study_spec(
        [
            {"address": "Knobs.N", "low": 10, "high": 30},
            {"address": "Knobs.amp", "low": 1, "high": 3, "integer": True},
            {"address": "Knobs.cell.n_syn", "low": 1, "high": 4},
            {"address": "Knobs.cell.tau", "low": 5, "high": 15},
        ]
    )
    integer = {d.address: d.integer for d in spec.dimensions}
    assert integer == {
        "Knobs.N": True,
        "Knobs.amp": True,
        "Knobs.cell.n_syn": True,
        "Knobs.cell.tau": False,
    }
    spec = _study_spec(
        [{"address": "Knobs.N", "low": 10, "high": 30, "integer": False}]
    )
    assert spec.dimensions[0].integer is False


def test_study_integer_axis_without_a_whole_number_is_skipped():
    spec = _study_spec([{"address": "Knobs.N", "low": 10.2, "high": 10.8}])
    assert spec.dimensions == []
    assert "no whole number" in spec.skipped[0]["reason"]


def test_study_dict_key_entry_reads_unit_from_schema_and_is_not_clipped():
    spec = _study_spec([{"address": "Knobs.cell.tau", "low": -50.0, "high": 50.0}])
    dim = spec.dimensions[0]
    assert (dim.low, dim.high, dim.unit) == (-50.0, 50.0, "ms")


def test_study_explore_reports_unresolvable_entries():
    entries = [
        {"address": "Nope.x", "low": 0, "high": 1},
        {"address": "Knobs.nope", "low": 0, "high": 1},
        {"address": "Knobs", "low": 0, "high": 1},
        {"address": "Knobs.cell.tau.deeper", "low": 0, "high": 1},
        {"address": "Knobs.amp", "low": 3, "high": 3},
        {"address": "Knobs.N", "low": "ten", "high": 30},
        {"address": "Knobs.label", "low": 0, "high": 1},
        {"address": "Knobs.cell.missing", "low": 0, "high": 1},
        {"address": "Knobs.cell", "low": 0, "high": 1},
        {"low": 0, "high": 1},
        "Knobs.amp",
    ]
    spec = _study_spec(entries)
    assert spec.dimensions == []
    reasons = [s["reason"] for s in spec.skipped]
    assert len(reasons) == len(entries)
    assert all(reasons)
    assert "no node named 'Nope'" in reasons[0]
    assert "has no parameter 'nope'" in reasons[1]
    assert "Node.param" in reasons[2]
    assert "deeper than one level" in reasons[3]
    assert "empty range" in reasons[4]
    assert "must both be numbers" in reasons[5]
    assert "not a number" in reasons[6]
    assert "not present" in reasons[7]
    assert "per-key" in reasons[8]
    assert {s["address"] for s in spec.skipped[-2:]} == {"explore[9]", "explore[10]"}


def test_study_duplicate_address_is_reported_once():
    spec = _study_spec(
        [
            {"address": "Knobs.amp", "low": 1, "high": 2},
            {"address": "Knobs.amp", "low": 2, "high": 3},
        ]
    )
    assert [d.low for d in spec.dimensions] == [1.0]
    assert spec.skipped == [{"address": "Knobs.amp", "reason": "declared twice"}]


def test_study_objectives_go_through_add_objective():
    spec = _study_spec(
        [{"address": "Probe.x", "low": 0, "high": 10}],
        objectives=[
            {
                "name": "y",
                "measures": "Probe.y",
                "low": 0,
                "high": 20,
                "unit": "Hz",
                "node_id": "rf-2",
            }
        ],
        run_baseline=True,
    )
    assert len(spec.objectives) == 1
    obj = spec.objectives[0]
    assert (obj.name, obj.measures, obj.low, obj.high, obj.goal, obj.unit) == (
        "y",
        "Probe.y",
        0,
        20,
        "in_range",
        "Hz",
    )
    assert obj.source == "study"
    assert spec.baseline["measured"] == {"y": 2.0}
    assert spec.baseline["params"] == {"Probe.x": 1.0}


def test_study_objective_with_a_bad_address_names_what_is_available():
    with pytest.raises(ValueError, match="Probe.y"):
        _study_spec(
            [{"address": "Probe.x", "low": 0, "high": 10}],
            objectives=[{"name": "y", "measures": "Probe.nope", "low": 0, "high": 1}],
            run_baseline=True,
        )


def test_study_objective_without_name_or_measures_is_refused():
    with pytest.raises(ValueError, match=r"objectives\[0\] needs name"):
        _study_spec([], objectives=[{"measures": "Probe.y", "low": 0, "high": 1}])
    with pytest.raises(ValueError, match=r"objectives\[0\] needs measures"):
        _study_spec([], objectives=[{"name": "y", "low": 0, "high": 1}])


def test_study_entries_do_not_leak_node_id_into_the_manifest():
    spec = _study_spec(
        [{"address": "Probe.x", "low": 0, "high": 10, "node_id": "rf-1"}],
        objectives=[
            {
                "name": "y",
                "measures": "Probe.y",
                "low": 0,
                "high": 20,
                "node_id": "rf-2",
            }
        ],
        run_baseline=True,
    )
    as_dict = spec.to_dict()
    assert "node_id" not in str(as_dict)
    round_trip = OptimizationSpec.from_dict(as_dict)
    assert round_trip.dimensions == spec.dimensions
    assert round_trip.objectives == spec.objectives


def test_an_empty_study_is_reported_by_validate():
    spec = _study_spec([], objectives=[])
    with pytest.raises(ValueError, match="NW_Optimization.explore"):
        spec.validate()
    spec = _study_spec([{"address": "Probe.x", "low": 0, "high": 10}], objectives=[])
    with pytest.raises(ValueError, match="NW_Optimization.objectives"):
        spec.validate()


def test_cmaes_still_refuses_two_study_objectives(tmp_path):
    """The refusal comes before Optuna is imported, so it holds without it."""
    workflow, _ = _knobs_workflow()
    spec = build_spec(
        workflow,
        AlgorithmConfig(name="cmaes", pop_size=2, max_generations=1),
        explore=[{"address": "Probe.x", "low": 0, "high": 10}],
        objectives=[
            {"name": "y", "measures": "Probe.y", "low": 0, "high": 20},
            {"name": "k", "measures": "Knobs.y", "low": 0, "high": 100},
        ],
    )
    with pytest.raises(ValueError, match="single-objective"):
        optimize(workflow, spec=spec, results_path=str(tmp_path), verbose=False)


def test_study_round_trip_with_random_search(tmp_path):
    workflow, _ = _knobs_workflow()
    spec = build_spec(
        workflow,
        AlgorithmConfig(name="random", pop_size=3, max_generations=2, seed=1),
        explore=[
            {"address": "Knobs.amp", "low": 0.0, "high": 5.0},
            {"address": "Knobs.N", "low": 10, "high": 30},
        ],
        objectives=[{"name": "k", "measures": "Knobs.y", "low": 0.0, "high": 150.0}],
    )
    result = optimize(
        workflow,
        spec=spec,
        results_path=str(tmp_path),
        per_trial_results=False,
        verbose=False,
    )
    assert result.best is not None
    assert isinstance(result.best["params"]["Knobs.N"], int)
