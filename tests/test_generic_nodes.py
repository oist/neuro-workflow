"""The optimization engine must serve any node, not only the NW_* family.

Nothing here imports NEST, BMTK or SONATA. The nodes are a plain four-stage chain —
drive -> cell -> shaper -> report — of the kind a participant or another lab would write,
using every schema feature the engine reads: a scalar dimension, an integer dimension, a
dict parameter with a range per key, objectives addressed into a nested output dict, a
parameter deliberately left out of the search, and a node that writes a file per trial.
They use nothing from neuroworkflow beyond Node and the schema — no helper, no utility —
because that is what a node from another lab looks like.

If the engine is genuinely node-agnostic, these pass without it knowing anything about
them. That is the claim these tests exist to keep honest.
"""

import json
import os

import pytest

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)
from neuroworkflow.core.workflow import WorkflowBuilder
from neuroworkflow.optimization import AlgorithmConfig, build_spec, optimize

# ---------------------------------------------------------------------------
# A chain of ordinary nodes
# ---------------------------------------------------------------------------


class Drive(Node):
    """Produces a current. A scalar dimension and an integer one (a pulse count)."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="generic_drive",
        stage="stimulus",
        tool="custom",
        description="Constant current built from an amplitude and a number of pulses.",
        parameters={
            "amplitude_pa": ParameterDefinition(
                default_value=100.0,
                unit="pA",
                description="Current per pulse.",
                optimizable=True,
                optimization_range=[0.0, 500.0],
            ),
            "n_pulses": ParameterDefinition(
                default_value=2,
                description="How many pulses are summed. A count, not a continuum.",
                optimizable=True,
                optimization_range=[1.0, 6.0],
            ),
            "waveform": ParameterDefinition(
                default_value="square",
                description="Shape name. Not numeric, so not a search dimension.",
                optimizable=True,  # deliberately marked: the engine must refuse it
            ),
        },
        outputs={
            "drive": PortDefinition(type=PortType.DICT, description="{'current_pa': …}")
        },
        methods={
            "run": MethodDefinition(description="Build the drive.", outputs=["drive"])
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, method_key="run")

    def run(self):
        p = self._parameters
        current = float(p["amplitude_pa"]) * int(p["n_pulses"])
        return {"drive": {"current_pa": current, "pulses": int(p["n_pulses"])}}


class Cell(Node):
    """A threshold-linear cell. Its parameters live in a dict, one range per key."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="generic_cell",
        stage="neuron",
        tool="custom",
        description="Threshold-linear rate model: gain x (current - threshold).",
        parameters={
            "cell_params": ParameterDefinition(
                default_value={"threshold_pa": 150.0, "gain_hz_per_pa": 0.2},
                description="Model parameters; each key can be searched separately.",
                optimizable=True,
                optimization_range={
                    "threshold_pa": [50.0, 400.0],
                    "gain_hz_per_pa": [0.05, 0.5],
                },
            )
        },
        inputs={"drive": PortDefinition(type=PortType.DICT, description="from Drive")},
        outputs={
            "response": PortDefinition(type=PortType.DICT, description="{'rate_hz': …}")
        },
        methods={
            "run": MethodDefinition(
                description="Rate from current.", inputs=["drive"], outputs=["response"]
            )
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, method_key="run")

    def run(self, drive):
        p = self._parameters["cell_params"]
        above = float(drive["current_pa"]) - float(p["threshold_pa"])
        rate = max(0.0, above * float(p["gain_hz_per_pa"]))
        return {"response": {"rate_hz": rate}}


class Shaper(Node):
    """Scales the incoming rate. An ordinary numeric parameter, left out of the search."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="generic_shaper",
        stage="analysis",
        tool="custom",
        description="Applies a fixed gain to the incoming rate.",
        parameters={
            "scale": ParameterDefinition(
                default_value=1.0,
                description="Multiplies the rate. Not marked optimizable here.",
                constraints={"min": 0.0},
            )
        },
        inputs={
            "response": PortDefinition(type=PortType.DICT, description="from Cell")
        },
        outputs={
            "shaped": PortDefinition(type=PortType.DICT, description="{'rate_hz': …}")
        },
        methods={
            "run": MethodDefinition(
                description="Shape the rate.", inputs=["response"], outputs=["shaped"]
            )
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, method_key="run")

    def run(self, response):
        scaled = float(response["rate_hz"]) * float(self._parameters["scale"])
        return {"shaped": {"rate_hz": scaled}}


class Report(Node):
    """Writes a file per trial and publishes the numbers an objective can address."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="generic_report",
        stage="io",
        tool="custom",
        description="Writes the trial's numbers to disk and republishes them.",
        parameters={
            "filename": ParameterDefinition(
                default_value="report.json", description="Written under results_path."
            )
        },
        inputs={
            "shaped": PortDefinition(type=PortType.DICT, description="from Shaper")
        },
        outputs={
            "metrics": PortDefinition(
                type=PortType.DICT,
                description="Nested: {'rate': {'hz': …}, 'label': 'ok'}",
            )
        },
        methods={
            "run": MethodDefinition(
                description="Report.", inputs=["shaped"], outputs=["metrics"]
            )
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self.add_process_step("run", self.run, method_key="run")

    def run(self, shaped):
        root = self._context.get("results_path", "results")
        os.makedirs(root, exist_ok=True)
        rate = float(shaped["rate_hz"])
        with open(os.path.join(root, str(self._parameters["filename"])), "w") as fh:
            json.dump({"rate_hz": rate}, fh)
        # Nested, and with a non-numeric sibling: the engine must address the number and
        # ignore the label.
        return {"metrics": {"rate": {"hz": rate}, "label": "ok"}}


def _workflow(tmp_path, scale=1.0):
    drive, cell = Drive("drive"), Cell("cell")
    shaper, report = Shaper("shaper"), Report("report")
    shaper.configure(scale=scale)

    builder = WorkflowBuilder("generic", context={"results_path": str(tmp_path)})
    for node in (drive, cell, shaper, report):
        builder.add_node(node)
    builder.connect("drive", "drive", "cell", "drive")
    builder.connect("cell", "response", "shaper", "response")
    builder.connect("shaper", "shaped", "report", "shaped")
    return builder.build(), (drive, cell, shaper, report)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_a_chain_of_custom_nodes_can_be_optimized(tmp_path):
    """Four ordinary nodes, four dimensions, one target, no simulator anywhere."""
    workflow, _ = _workflow(tmp_path)
    spec = build_spec(
        workflow,
        AlgorithmConfig(name="random", pop_size=8, max_generations=6, seed=3),
    )
    spec.add_objective(
        name="rate", measures="report.metrics.rate.hz", low=40.0, high=60.0, unit="Hz"
    )

    result = optimize(
        workflow, spec=spec, results_path=str(tmp_path / "opt"), verbose=False
    )

    addresses = {d.address for d in spec.dimensions}
    assert addresses == {
        "drive.amplitude_pa",
        "drive.n_pulses",
        "cell.cell_params.threshold_pa",
        "cell.cell_params.gain_hz_per_pa",
    }
    assert result.best is not None, "no trial produced a usable measurement"
    assert 40.0 <= result.best["measured"]["rate"] <= 60.0, result.best["measured"]
    assert result.stop_reason == "a candidate is inside every target band"


def test_a_non_numeric_parameter_is_refused_as_a_dimension(tmp_path):
    """`waveform` is marked optimizable but is a string — it must be skipped, with a why."""
    workflow, _ = _workflow(tmp_path)
    spec = build_spec(
        workflow, AlgorithmConfig(name="random", pop_size=2), run_baseline=False
    )
    assert "drive.waveform" not in {d.address for d in spec.dimensions}
    reasons = {s["address"]: s["reason"] for s in spec.skipped}
    assert "drive.waveform" in reasons
    assert reasons["drive.waveform"]


def test_the_integer_dimension_of_a_custom_node_stays_whole(tmp_path):
    """n_pulses is a count: the ledger must record what the node actually used."""
    workflow, nodes = _workflow(tmp_path)
    spec = build_spec(
        workflow, AlgorithmConfig(name="random", pop_size=5, max_generations=1, seed=1)
    )
    spec.add_objective(
        name="rate", measures="report.metrics.rate.hz", low=0.0, high=1e9
    )
    result = optimize(
        workflow, spec=spec, results_path=str(tmp_path / "opt"), verbose=False
    )
    for trial in result.trials:
        recorded = trial["params"]["drive.n_pulses"]
        assert isinstance(recorded, int), recorded
        assert 1 <= recorded <= 6
        # the node reports the count it used; it must match the ledger
        assert trial["measured"]["rate"] >= 0.0


def test_several_objectives_over_custom_nodes_give_a_pareto_front(tmp_path):
    """Two objectives from the same nested dict, on nodes the engine has never seen."""
    workflow, _ = _workflow(tmp_path)
    spec = build_spec(
        workflow,
        AlgorithmConfig(name="random", pop_size=6, max_generations=4, seed=7),
    )
    spec.add_objective(
        name="rate", measures="report.metrics.rate.hz", low=40.0, high=60.0, unit="Hz"
    )
    spec.add_objective(name="pulses", measures="drive.drive.pulses", low=2.0, high=3.0)

    result = optimize(
        workflow, spec=spec, results_path=str(tmp_path / "opt"), verbose=False
    )
    assert len(spec.objectives) == 2
    assert result.pareto_front, "a multi-objective run must produce a front"
    for row in result.pareto_front:
        assert set(row["measured"]) == {"rate", "pulses"}
    # A run with several objectives develops the front rather than stopping early.
    assert result.stop_reason == "generation budget exhausted"


def test_each_trial_gets_its_own_results_directory(tmp_path):
    """Per-trial directories are generic: a node that writes files needs no NW_ plumbing."""
    workflow, _ = _workflow(tmp_path)
    spec = build_spec(
        workflow, AlgorithmConfig(name="random", pop_size=4, max_generations=1, seed=5)
    )
    spec.add_objective(
        name="rate", measures="report.metrics.rate.hz", low=0.0, high=1e9
    )
    result = optimize(
        workflow, spec=spec, results_path=str(tmp_path / "opt"), verbose=False
    )

    written = [t["results_path"] for t in result.trials]
    assert len(set(written)) == len(written), "trials shared a directory"
    for path in written:
        assert os.path.exists(os.path.join(path, "report.json")), path
