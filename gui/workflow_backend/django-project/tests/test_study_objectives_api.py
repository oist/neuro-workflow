"""The study-objective endpoint behind the chat tools set/remove_study_objective.

An objective of the optimization study is always measured on an OUTPUT port of
a node, never on a parameter. The endpoint is where that rule is enforced, so
the LLM cannot talk its way around it; it also owns the write to
``data.study.objectives`` on the NW_Optimization node so nothing else on that
node is touched.
"""

import pytest
from django.urls import reverse

from app.workflow.code_generation_service import CodeGenerationService
from app.workflow.models import FlowNode, FlowProject

pytestmark = pytest.mark.django_db


def make_node(project, node_id, label, instance_name, outputs=None, parameters=None):
    return FlowNode.objects.create(
        id=node_id,
        project=project,
        position_x=0,
        position_y=0,
        node_type="calculationNode",
        data={
            "label": label,
            "instanceName": instance_name,
            "nodeType": "optimization" if label == "NW_Optimization" else "analysis",
            "schema": {
                "inputs": {},
                "outputs": outputs or {},
                "parameters": parameters or {},
                "methods": {},
            },
        },
    )


@pytest.fixture
def project(user_alice):
    p = FlowProject.objects.create(name="opt", owner=user_alice)
    make_node(
        p,
        "opt1",
        "NW_Optimization",
        "opt",
        parameters={"algorithm": {"default_value": "cmaes"}},
    )
    make_node(
        p,
        "ana1",
        "NW_FiringRate",
        "ana",
        outputs={"firing_rate_hz": {"type": "dict"}, "isi_stats": {"type": "dict"}},
        parameters={"mean_firing_rate": {"default_value": 0, "is_objective": True}},
    )
    return p


def set_url(project):
    return reverse("workflow:study-objective-set", args=[project.id])


def delete_url(project, name):
    return reverse("workflow:study-objective-delete", args=[project.id, name])


def objectives(project):
    return FlowNode.objects.get(id="opt1", project=project).data["study"]["objectives"]


def test_output_port_objective_is_stored_and_generated(
    auth_client, user_alice, project
):
    client = auth_client(user_alice)
    res = client.put(
        set_url(project),
        {
            "node_id": "ana1",
            "port": "firing_rate_hz",
            "key": "exc",
            "low": 40,
            "high": 50,
            "unit": "Hz",
        },
        format="json",
    )
    assert res.status_code == 200, res.content
    body = res.json()
    assert body["optimization_node_id"] == "opt1"
    assert body["objective"]["name"] == "ana_firing_rate_hz_exc"
    assert "ana.firing_rate_hz.exc" in body["message"]

    opt = FlowNode.objects.get(id="opt1", project=project)
    assert opt.data["schema"]["parameters"]["algorithm"]["default_value"] == "cmaes"
    stored = opt.data["study"]["objectives"]
    assert stored == [
        {
            "node_id": "ana1",
            "port": "firing_rate_hz",
            "key": "exc",
            "name": "ana_firing_rate_hz_exc",
            "goal": "in_range",
            "low": 40,
            "high": 50,
            "unit": "Hz",
        }
    ]
    # What the generator emits for it
    lines = CodeGenerationService()._objective_lines(opt.data["study"], {"ana1": "ana"})
    assert "        measures='ana.firing_rate_hz.exc'," in lines


def test_parameter_is_rejected_with_the_valid_output_ports(
    auth_client, user_alice, project
):
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "ana1", "port": "mean_firing_rate", "low": 8, "high": 12},
        format="json",
    )
    assert res.status_code == 400
    error = res.json()["error"]
    assert "OUTPUT port" in error
    assert "firing_rate_hz" in error and "isi_stats" in error
    assert "study" not in FlowNode.objects.get(id="opt1", project=project).data


def test_missing_port_is_rejected(auth_client, user_alice, project):
    res = auth_client(user_alice).put(
        set_url(project), {"node_id": "ana1", "low": 8, "high": 12}, format="json"
    )
    assert res.status_code == 400
    assert "firing_rate_hz" in res.json()["error"]


def test_node_without_outputs_is_rejected(auth_client, user_alice, project):
    make_node(project, "io1", "NW_Writer", "writer")
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "io1", "port": "x", "low": 1, "high": 2},
        format="json",
    )
    assert res.status_code == 400
    assert "no output ports" in res.json()["error"]


def test_optimization_node_itself_is_rejected(auth_client, user_alice, project):
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "opt1", "port": "algorithm", "low": 1, "high": 2},
        format="json",
    )
    assert res.status_code == 400
    assert "NW_Optimization node has no outputs" in res.json()["error"]


def test_unknown_node_is_rejected(auth_client, user_alice, project):
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "nope", "port": "x", "low": 1, "high": 2},
        format="json",
    )
    assert res.status_code == 400
    assert "not found" in res.json()["error"]


def test_without_optimization_node(auth_client, user_alice, project):
    FlowNode.objects.get(id="opt1", project=project).delete()
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "ana1", "port": "firing_rate_hz", "low": 1, "high": 2},
        format="json",
    )
    assert res.status_code == 400
    assert "No NW_Optimization" in res.json()["error"]


def test_two_optimization_nodes(auth_client, user_alice, project):
    make_node(project, "opt2", "NW_Optimization", "opt_b")
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "ana1", "port": "firing_rate_hz", "low": 1, "high": 2},
        format="json",
    )
    assert res.status_code == 400
    error = res.json()["error"]
    assert "Only one" in error and "opt" in error and "opt_b" in error


@pytest.mark.parametrize(
    "body",
    [
        {"goal": "smallest", "low": 1, "high": 2},
        {"goal": "in_range", "low": 5, "high": 5},
        {"goal": "in_range", "low": 5},
        {"goal": "in_range", "low": "abc", "high": 2},
    ],
)
def test_bad_goal_or_range(auth_client, user_alice, project, body):
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "ana1", "port": "firing_rate_hz", **body},
        format="json",
    )
    assert res.status_code == 400


def test_minimize_needs_no_range(auth_client, user_alice, project):
    res = auth_client(user_alice).put(
        set_url(project),
        {"node_id": "ana1", "port": "isi_stats", "key": "cv", "goal": "minimize"},
        format="json",
    )
    assert res.status_code == 200, res.content
    obj = res.json()["objective"]
    assert obj["goal"] == "minimize" and obj["low"] is None and obj["high"] is None
    assert "unit" not in obj


def test_same_name_replaces(auth_client, user_alice, project):
    client = auth_client(user_alice)
    base = {"node_id": "ana1", "port": "firing_rate_hz", "name": "rate", "low": 40}
    assert (
        client.put(set_url(project), {**base, "high": 50}, format="json").status_code
        == 200
    )
    res = client.put(set_url(project), {**base, "high": 60}, format="json")
    assert res.status_code == 200
    assert [o["high"] for o in objectives(project)] == [60]
    assert len(res.json()["objectives"]) == 1


def test_delete(auth_client, user_alice, project):
    client = auth_client(user_alice)
    client.put(
        set_url(project),
        {"node_id": "ana1", "port": "firing_rate_hz", "low": 40, "high": 50},
        format="json",
    )
    res = client.delete(delete_url(project, "ana_firing_rate_hz"))
    assert res.status_code == 200
    assert res.json()["objectives"] == []
    assert objectives(project) == []

    res = client.delete(delete_url(project, "ana_firing_rate_hz"))
    assert res.status_code == 404
    assert "No objective named" in res.json()["error"]


def test_other_users_private_project_is_hidden(auth_client, user_bob, project):
    res = auth_client(user_bob).put(
        set_url(project),
        {"node_id": "ana1", "port": "firing_rate_hz", "low": 40, "high": 50},
        format="json",
    )
    assert res.status_code == 404


# --- the study panel's whole-list save ------------------------------------------


def test_replace_list(auth_client, user_alice, project):
    client = auth_client(user_alice)
    items = [
        {
            "node_id": "ana1",
            "port": "firing_rate_hz",
            "key": "exc",
            "name": "rate",
            "goal": "in_range",
            "low": 40,
            "high": 50,
            "unit": "Hz",
        },
        {
            "node_id": "ana1",
            "port": "isi_stats",
            "key": "cv",
            "name": "cv",
            "goal": "minimize",
            "low": None,
            "high": None,
        },
    ]
    res = client.put(set_url(project), {"objectives": items}, format="json")
    assert res.status_code == 200, res.content
    assert [o["name"] for o in res.json()["objectives"]] == ["rate", "cv"]
    assert [o["name"] for o in objectives(project)] == ["rate", "cv"]

    res = client.put(set_url(project), {"objectives": []}, format="json")
    assert res.status_code == 200
    assert objectives(project) == []


def test_replace_list_keeps_a_row_whose_node_left_the_canvas(
    auth_client, user_alice, project
):
    items = [
        {"node_id": "gone", "port": "rate", "name": "old", "goal": "minimize"},
        {
            "node_id": "ana1",
            "port": "firing_rate_hz",
            "name": "new",
            "goal": "maximize",
        },
    ]
    res = auth_client(user_alice).put(
        set_url(project), {"objectives": items}, format="json"
    )
    assert res.status_code == 200, res.content
    assert [o["node_id"] for o in objectives(project)] == ["gone", "ana1"]


@pytest.mark.parametrize(
    "items, fragment",
    [
        (
            [{"node_id": "ana1", "port": "mean_firing_rate", "goal": "minimize"}],
            "OUTPUT port",
        ),
        (
            [
                {
                    "node_id": "ana1",
                    "port": "firing_rate_hz",
                    "name": "x",
                    "goal": "minimize",
                },
                {
                    "node_id": "ana1",
                    "port": "isi_stats",
                    "name": "x",
                    "goal": "minimize",
                },
            ],
            "unique",
        ),
        ("not a list", "must be a list"),
        (["not an object"], "must be an object"),
    ],
)
def test_replace_list_rejections(auth_client, user_alice, project, items, fragment):
    res = auth_client(user_alice).put(
        set_url(project), {"objectives": items}, format="json"
    )
    assert res.status_code == 400
    assert fragment in res.json()["error"]
    assert "study" not in FlowNode.objects.get(id="opt1", project=project).data


def test_delete_name_containing_a_slash(auth_client, user_alice, project):
    client = auth_client(user_alice)
    res = client.put(
        set_url(project),
        {
            "node_id": "ana1",
            "port": "firing_rate_hz",
            "name": "rate/exc",
            "low": 40,
            "high": 50,
        },
        format="json",
    )
    assert res.status_code == 200
    res = client.delete(delete_url(project, "rate/exc"))
    assert res.status_code == 200, res.content
    assert objectives(project) == []


# --- the study endpoint is the only writer of data.study ----------------------------


def stored_study(project):
    return FlowNode.objects.get(id="opt1", project=project).data.get("study")


def test_general_node_update_keeps_the_stored_study(auth_client, user_alice, project):
    client = auth_client(user_alice)
    client.put(
        set_url(project),
        {"node_id": "ana1", "port": "firing_rate_hz", "low": 40, "high": 50},
        format="json",
    )
    before = stored_study(project)
    opt = FlowNode.objects.get(id="opt1", project=project)
    forged = dict(opt.data)
    forged["study"] = {
        "objectives": [{"node_id": "ana1", "port": "mean_firing_rate", "name": "bad"}]
    }
    forged["instanceName"] = "renamed"
    res = client.put(
        f"/api/workflow/{project.id}/nodes/opt1/",
        {"position": {"x": 1, "y": 2}, "type": "calculationNode", "data": forged},
        format="json",
    )
    assert res.status_code == 200, res.content
    opt.refresh_from_db()
    assert opt.data["instanceName"] == "renamed"
    assert opt.data["study"] == before


def test_general_node_update_cannot_introduce_a_study(auth_client, user_alice, project):
    opt = FlowNode.objects.get(id="opt1", project=project)
    forged = dict(opt.data, study={"objectives": [{"node_id": "ana1", "port": "x"}]})
    res = auth_client(user_alice).put(
        f"/api/workflow/{project.id}/nodes/opt1/",
        {"position": {"x": 0, "y": 0}, "type": "calculationNode", "data": forged},
        format="json",
    )
    assert res.status_code == 200, res.content
    assert stored_study(project) is None


def test_node_create_drops_a_study(auth_client, user_alice, project):
    res = auth_client(user_alice).post(
        f"/api/workflow/{project.id}/nodes/",
        {
            "id": "opt_new",
            "position": {"x": 0, "y": 0},
            "type": "calculationNode",
            "data": {
                "label": "Other",
                "nodeType": "analysis",
                "schema": {},
                "study": {"objectives": [{"node_id": "ana1", "port": "x"}]},
            },
        },
        format="json",
    )
    assert res.status_code in (200, 201), res.content
    assert "study" not in FlowNode.objects.get(id="opt_new", project=project).data


def test_flow_save_keeps_the_stored_study(auth_client, user_alice, project):
    client = auth_client(user_alice)
    client.put(
        set_url(project),
        {"node_id": "ana1", "port": "firing_rate_hz", "low": 40, "high": 50},
        format="json",
    )
    before = stored_study(project)
    flow = client.get(f"/api/workflow/{project.id}/flow/").json()
    for node in flow["nodes"]:
        if node["id"] == "opt1":
            node["data"]["study"] = {"objectives": []}
            node["position"] = {"x": 5, "y": 5}
        else:
            node["data"]["study"] = {"objectives": [{"node_id": "x", "port": "y"}]}
    res = client.put(f"/api/workflow/{project.id}/flow/", flow, format="json")
    assert res.status_code == 200, res.content
    assert stored_study(project) == before
    assert FlowNode.objects.get(id="opt1", project=project).position_x == 5
    assert "study" not in FlowNode.objects.get(id="ana1", project=project).data
