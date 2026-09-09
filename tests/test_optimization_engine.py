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
    objective_fitness,
    optimize,
)
from neuroworkflow.optimization.optimizers import RandomSearch, sampler_init_kwargs
from neuroworkflow.optimization.spec import (
    AlgorithmConfig,
    Dimension,
    Objective,
    OptimizationSpec,
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
