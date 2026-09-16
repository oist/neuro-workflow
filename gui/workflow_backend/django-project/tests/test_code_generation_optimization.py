"""Code generation with an NW_Optimization node on the canvas.

The optimization node turns the generated script into a search: the parameters
marked optimizable in the editor are declared on the built nodes, the study's
objectives are added to the spec and the engine runs the workflow many times.
Without the node the script must be exactly what it was before.
"""

import ast
from types import SimpleNamespace

import pytest

from app.workflow.code_generation_service import (
    NORMAL_TAIL,
    OPTIMIZATION_IMPORT,
    CodeGenerationError,
    CodeGenerationService,
)
from app.workflow.models import FlowProject


def node(node_id, label, instance_name=None, parameters=None, outputs=None, **data):
    return {
        "id": node_id,
        "type": "calculationNode",
        "position": {"x": 0, "y": 0},
        "data": {
            "label": label,
            "instanceName": instance_name or label,
            "nodeType": "network" if label != "NW_Optimization" else "optimization",
            "schema": {
                "inputs": {},
                "outputs": outputs or {},
                "parameters": parameters or {},
                "methods": {},
            },
            **data,
        },
    }


def opt_node(node_id="opt1", instance_name="opt", study=None):
    n = node(
        node_id,
        "NW_Optimization",
        instance_name,
        parameters={"algorithm": {"default_value": "cmaes"}},
    )
    if study is not None:
        n["data"]["study"] = study
    return n


CLAMP = node(
    "clamp1",
    "NW_IClamp",
    "clamp",
    parameters={
        "amp_na": {
            "default_value": 0.15,
            "optimizable": True,
            "optimization_range": [100.0, 1000.0],
            "unit": "nA",
        },
        "delay_ms": {"default_value": 500.0},
    },
)
POP = node(
    "pop1",
    "NW_Population",
    "exc",
    parameters={
        "nest_params": {
            "default_value": {"tau_m": 10.0, "V_th": -55.0},
            "optimizable": True,
            "optimization_range": {"tau_m": ["6", "14"], "V_th": "not a range"},
        },
        "N": {"default_value": 20, "optimizable": True},
        "off": {
            "default_value": 1.0,
            "optimizable": False,
            "optimization_range": [0, 1],
        },
    },
)
ANA = node("ana1", "NW_Analysis", "ana", outputs={"firing_rate_hz": {"type": "dict"}})

EDGES = [
    {
        "source": "clamp1",
        "target": "pop1",
        "sourceHandle": "clamp1-iclamp-output-object",
        "targetHandle": "pop1-iclamp-input-object",
    },
    {
        "source": "pop1",
        "target": "opt1",
        "sourceHandle": "pop1-population-output-object",
        "targetHandle": "opt1-default_input-input-any",
    },
]


@pytest.fixture
def service():
    return CodeGenerationService()


# --- detection -------------------------------------------------------------


def test_no_optimization_node(service):
    assert service._find_optimization_node([CLAMP, POP]) is None


def test_one_optimization_node(service):
    assert service._find_optimization_node([CLAMP, opt_node()])["id"] == "opt1"


def test_two_optimization_nodes_is_an_error(service):
    with pytest.raises(CodeGenerationError) as exc:
        service._find_optimization_node(
            [opt_node("o1", "coarse"), CLAMP, opt_node("o2", "fine")]
        )
    assert "coarse" in str(exc.value) and "fine" in str(exc.value)


# --- builder commands --------------------------------------------------------


def test_skipped_node_is_neither_added_nor_connected(service):
    commands, node_id_to_var = service._build_workflow_commands_from_json(
        [CLAMP, POP, opt_node()], EDGES, skip={"opt1"}
    )
    assert node_id_to_var["opt1"] == "opt"
    assert "    workflow_builder.add_node(clamp)" in commands
    assert "    workflow_builder.add_node(exc)" in commands
    assert not any("add_node(opt)" in c for c in commands)
    assert not any('"opt"' in c for c in commands)
    assert (
        'workflow_builder.connect("clamp", "iclamp", "exc", "iclamp")' in commands[-2]
    )
    assert commands[-1] == "    workflow = workflow_builder.build()"


def test_without_skip_every_node_is_added(service):
    commands, _ = service._build_workflow_commands_from_json([CLAMP, POP], EDGES[:1])
    assert sum("add_node(" in c for c in commands) == 2


# --- explore ---------------------------------------------------------------


def test_explore_lines(service):
    node_id_to_var = {"clamp1": "clamp", "pop1": "exc", "opt1": "opt"}
    lines = service._explore_lines([CLAMP, POP, opt_node()], node_id_to_var, "opt1")
    assert lines == [
        '    clamp.NODE_DEFINITION.parameters["amp_na"].optimizable = True',
        '    clamp.NODE_DEFINITION.parameters["amp_na"].optimization_range = '
        "[100.0, 1000.0]",
        "    clamp.NODE_DEFINITION.parameters[\"amp_na\"].unit = 'nA'",
        "",
        '    exc.NODE_DEFINITION.parameters["nest_params"].optimizable = True',
        '    exc.NODE_DEFINITION.parameters["nest_params"].optimization_range = '
        "{'tau_m': [6.0, 14.0]}",
        '    exc.NODE_DEFINITION.parameters["N"].optimizable = True',
    ]


def test_explore_lines_ignores_a_node_without_flags(service):
    assert service._explore_lines([ANA], {"ana1": "ana"}, "opt1") == []


def test_coerce_range_accepts_a_json_string(service):
    assert service._coerce_range("[1, 2.5]") == [1, 2.5]
    assert service._coerce_range("[1]") is None
    assert service._coerce_range({"a": [1, "x"]}) is None


# --- objectives --------------------------------------------------------------


def test_objective_lines(service):
    study = {
        "objectives": [
            {
                "node_id": "ana1",
                "port": "firing_rate_hz",
                "key": "exc",
                "name": "exc_rate",
                "goal": "in_range",
                "low": 40,
                "high": "50",
                "unit": "Hz",
            },
            {
                "node_id": "ana1",
                "port": "isi_stats",
                "key": "exc.cv",
                "name": "regular",
                "goal": "minimize",
            },
            {"node_id": "gone", "port": "x", "name": "orphan"},
            {"node_id": "ana1", "port": "firing_rate_hz", "name": ""},
        ]
    }
    lines = service._objective_lines(study, {"ana1": "ana"})
    assert lines == [
        "    spec.add_objective(",
        "        name='exc_rate',",
        "        measures='ana.firing_rate_hz.exc',",
        "        low=40,",
        "        high=50.0,",
        "        unit='Hz'",
        "    )",
        "    spec.add_objective(",
        "        name='regular',",
        "        measures='ana.isi_stats.exc.cv',",
        "        goal='minimize'",
        "    )",
        "    # objective 'orphan' skipped: its node is no longer on the canvas",
        "    spec.add_objective(",
        "        name='ana_firing_rate_hz',",
        "        measures='ana.firing_rate_hz'",
        "    )",
    ]


def test_objective_lines_without_a_study(service):
    assert service._objective_lines(None, {}) == []
    assert service._objective_lines({}, {}) == []


# --- template ------------------------------------------------------------------

PROJECT = SimpleNamespace(
    name="P", description="", workflow_context={"results_path": "results/"}
)


def test_normal_template_ends_with_the_execute_tail(service):
    code = service._create_base_template(PROJECT)
    assert code.endswith("    print(workflow)\n\n" + NORMAL_TAIL)
    assert "success = workflow.execute()" in code
    assert "build_spec" not in code
    ast.parse(code.replace("    # Create nodes", "    workflow = None"))


def test_optimization_tail_parses(service):
    node_id_to_var = {"clamp1": "clamp", "ana1": "ana", "opt1": "opt"}
    study = {
        "objectives": [
            {
                "node_id": "ana1",
                "port": "firing_rate_hz",
                "key": "exc",
                "name": "r",
                "low": 40,
                "high": 50,
            }
        ]
    }
    tail = service._optimization_tail(
        "opt",
        service._explore_lines([CLAMP, ANA], node_id_to_var, "opt1"),
        service._objective_lines(study, node_id_to_var),
    )
    code = service._create_base_template(PROJECT, tail=tail)
    ast.parse(code.replace("    # Create nodes", "    workflow = None"))
    assert "success = workflow.execute()" not in code
    assert "    spec = build_spec(workflow, opt.algorithm_config())\n" in code
    assert (
        "    result = optimize(workflow, spec=spec, results_path=opt.results_path())\n"
        in code
    )
    assert (
        code.index(".optimizable = True")
        < code.index("build_spec(")
        < code.index("add_objective(")
    )
    assert code.endswith('if __name__ == "__main__":\n    sys.exit(main())\n')


def test_optimization_tail_without_explore_or_objectives(service):
    tail = service._optimization_tail("opt", [], [])
    assert "No parameter is marked optimizable" in tail
    assert "add_objective" not in tail
    ast.parse("def main():\n" + tail)


# --- end to end ----------------------------------------------------------------


@pytest.fixture
def project(db, settings, tmp_path, user_alice):
    settings.BASE_DIR = str(tmp_path)
    return FlowProject.objects.create(name="Gen", owner=user_alice)


def generate(project, nodes, edges=()):
    service = CodeGenerationService()
    assert service.generate_code_from_flow_data(
        str(project.id), project.name, nodes, list(edges)
    )
    return service.get_code_file_path(project).read_text()


def test_generation_switches_between_modes_and_is_idempotent(project):
    normal_1 = generate(project, [CLAMP, POP, ANA], EDGES[:1])
    assert "success = workflow.execute()" in normal_1
    assert OPTIMIZATION_IMPORT not in normal_1

    study = {
        "objectives": [
            {
                "node_id": "ana1",
                "port": "firing_rate_hz",
                "key": "exc",
                "name": "exc_rate",
                "low": 40,
                "high": 50,
                "unit": "Hz",
            }
        ]
    }
    opt_1 = generate(project, [CLAMP, POP, ANA, opt_node(study=study)], EDGES)
    assert OPTIMIZATION_IMPORT in opt_1
    assert "from nodes.optimization.NW_Optimization import NW_Optimization" in opt_1
    assert '    opt = NW_Optimization("opt")' in opt_1
    assert "add_node(opt)" not in opt_1
    assert not any(
        "connect(" in line and '"opt"' in line for line in opt_1.splitlines()
    )
    assert "measures='ana.firing_rate_hz.exc'" in opt_1
    assert "success = workflow.execute()" not in opt_1
    ast.parse(opt_1)

    opt_2 = generate(project, [CLAMP, POP, ANA, opt_node(study=study)], EDGES)
    assert opt_2 == opt_1

    normal_2 = generate(project, [CLAMP, POP, ANA], EDGES[:1])
    assert normal_2 == normal_1


def test_view_rejects_two_optimization_nodes(project, auth_client, user_alice):
    client = auth_client(user_alice)
    res = client.post(
        f"/api/workflow/{project.id}/generate-code/",
        {"nodes": [opt_node("o1", "coarse"), opt_node("o2", "fine")], "edges": []},
        format="json",
    )
    assert res.status_code == 400
    assert "coarse" in res.json()["error"]
