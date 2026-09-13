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
)
from neuroworkflow.optimization.optimizers import RandomSearch, sampler_init_kwargs
from neuroworkflow.optimization.spec import (
    AlgorithmConfig,
    Dimension,
    Objective,
    OptimizationSpec,
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


def test_connection_rule_lambda_only():
    fn = NW_Connectivity._coerce_connection_rule("lambda src, tgt: 1")
    assert fn(0, 1) == 1
    assert NW_Connectivity._coerce_connection_rule(3) == 3
    with pytest.raises(ValueError, match="lambda"):
        NW_Connectivity._coerce_connection_rule("__import__('os').system('x')")
    with pytest.raises(ValueError, match="lambda"):
        NW_Connectivity._coerce_connection_rule("src + tgt")


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
