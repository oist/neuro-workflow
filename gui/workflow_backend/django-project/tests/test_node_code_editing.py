"""Editing node code from the GUI.

Library files: guarded save and save-as-copy (palette). Canvas nodes: a
project copy under project_nodes/, schema refresh, generated imports.
"""

import copy
import hashlib

import pytest
from app.box.models import PythonFile
from app.box.services.python_file_service import PythonFileService
from app.workflow.code_generation_service import CodeGenerationService
from app.workflow.models import FlowEdge, FlowNode, FlowProject
from app.workflow.path_utils import projects_root
from django.core.files.base import ContentFile
from django.urls import reverse

pytestmark = pytest.mark.django_db


def _source(class_name="EditProbeNode", params=None, outputs=("ok",)):
    params = params if params is not None else {"rate": 10, "size": 3}
    param_lines = "".join(
        f"            '{k}': ParameterDefinition(default_value={v!r}, "
        f"description='{k}'),\n"
        for k, v in params.items()
    )
    output_lines = "".join(
        f"            '{o}': PortDefinition(type=PortType.STR, description='{o}'),\n"
        for o in outputs
    )
    return f"""\
from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)
from neuroworkflow.core.port import PortType


class {class_name}(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type='edit_probe',
        description='probe node for code editing tests',
        parameters={{
{param_lines}        }},
        inputs={{}},
        outputs={{
{output_lines}        }},
        methods={{}},
    )
"""


@pytest.fixture
def nodes_dir(settings, tmp_path):
    root = tmp_path / "nodes"
    (root / "analysis").mkdir(parents=True)
    settings.MEDIA_ROOT = str(root)
    # Project directories (codes/projects/<id>/) go under tmp_path too
    settings.BASE_DIR = str(tmp_path)
    return root


def _create(owner, code, name="EditProbeNode.py"):
    return PythonFileService().create_python_file(
        ContentFile(code.encode("utf-8"), name=name),
        user=owner,
        name=name,
        category="analysis",
    )


def _hash(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def code_url(identifier):
    return reverse("box:python-file-code", args=[str(identifier)])


def test_get_by_id_reports_disk_content_and_edit_right(
    auth_client, user_alice, user_bob, nodes_dir
):
    node = _create(user_alice, _source())
    node.status = PythonFile.Status.PUBLIC
    node.save()
    # A JupyterLab edit changes only the file on disk.
    disk_code = _source(params={"rate": 99})
    (nodes_dir / "analysis" / "EditProbeNode.py").write_text(disk_code)

    body = auth_client(user_alice).get(code_url(node.id)).json()
    assert body["code"] == disk_code
    assert body["content_hash"] == _hash(disk_code)
    assert body["file_id"] == str(node.id)
    assert body["class_names"] == ["EditProbeNode"]
    assert body["can_edit"] is True

    assert auth_client(user_bob).get(code_url(node.id)).json()["can_edit"] is False


def test_put_saves_disk_db_and_schema(auth_client, user_alice, nodes_dir):
    old = _source()
    node = _create(user_alice, old)
    new = _source(params={"rate": 10, "size": 3, "gain": 1.5})

    res = auth_client(user_alice).put(
        code_url(node.id),
        {"code": new, "base_hash": _hash(old), "expected_class": "EditProbeNode"},
        format="json",
    )
    assert res.status_code == 200, res.content
    assert res.json()["content_hash"] == _hash(new)
    node.refresh_from_db()
    assert node.file_content == new
    assert "gain" in node.node_classes["EditProbeNode"]["parameters"]
    assert (nodes_dir / "analysis" / "EditProbeNode.py").read_text() == new


@pytest.mark.parametrize(
    "code, line",
    [
        ("class EditProbeNode(Node:\n    pass\n", 1),
        (_source(class_name="Renamed"), None),
    ],
)
def test_put_rejects_unusable_code_without_writing(
    auth_client, user_alice, nodes_dir, code, line
):
    old = _source()
    node = _create(user_alice, old)

    res = auth_client(user_alice).put(
        code_url(node.id),
        {"code": code, "base_hash": _hash(old), "expected_class": "EditProbeNode"},
        format="json",
    )
    assert res.status_code == 400
    assert res.json().get("line") == line
    node.refresh_from_db()
    assert node.file_content == old
    assert (nodes_dir / "analysis" / "EditProbeNode.py").read_text() == old


def test_put_conflicts_when_file_changed_elsewhere(auth_client, user_alice, nodes_dir):
    old = _source()
    node = _create(user_alice, old)
    jupyter_edit = _source(params={"rate": 42})
    (nodes_dir / "analysis" / "EditProbeNode.py").write_text(jupyter_edit)

    res = auth_client(user_alice).put(
        code_url(node.id),
        {"code": _source(params={"rate": 1}), "base_hash": _hash(old)},
        format="json",
    )
    assert res.status_code == 409
    assert (nodes_dir / "analysis" / "EditProbeNode.py").read_text() == jupyter_edit


def test_put_forbidden_for_someone_elses_node(
    auth_client, user_alice, user_bob, nodes_dir
):
    node = _create(user_alice, _source())
    node.status = PythonFile.Status.PUBLIC
    node.save()

    res = auth_client(user_bob).put(
        code_url(node.id), {"code": _source(params={})}, format="json"
    )
    assert res.status_code == 403


def test_fork_creates_private_copy_with_matching_class_and_file(
    auth_client, user_alice, user_bob, nodes_dir
):
    node = _create(user_alice, _source())
    node.status = PythonFile.Status.PUBLIC
    node.save()
    edited = _source(params={"rate": 7})

    res = auth_client(user_bob).post(
        reverse("box:python-file-fork", args=[node.id]),
        {
            "code": edited,
            "class_name": "EditProbeNode",
            "new_class_name": "EditProbeNodeBob",
        },
        format="json",
    )
    assert res.status_code == 201, res.content
    body = res.json()
    assert body["file_name"] == "EditProbeNodeBob.py"
    assert body["class_name"] == "EditProbeNodeBob"

    copied = PythonFile.objects.get(pk=body["file_id"])
    assert copied.uploaded_by == user_bob
    assert copied.status == PythonFile.Status.PRIVATE
    assert list(copied.node_classes) == ["EditProbeNodeBob"]
    assert (
        copied.node_classes["EditProbeNodeBob"]["parameters"]["rate"]["default_value"]
        == 7
    )
    on_disk = (nodes_dir / "analysis" / "EditProbeNodeBob.py").read_text()
    assert "class EditProbeNodeBob(Node):" in on_disk
    # The original is untouched.
    node.refresh_from_db()
    assert node.file_content == _source()


@pytest.mark.parametrize(
    "new_class_name, expected",
    [("EditProbeNode", 400), ("bad-name", 400), ("Taken", 409)],
)
def test_fork_rejects_bad_or_taken_names(
    auth_client, user_alice, nodes_dir, new_class_name, expected
):
    node = _create(user_alice, _source())
    (nodes_dir / "analysis" / "Taken.py").write_text("# unsynced file\n")

    res = auth_client(user_alice).post(
        reverse("box:python-file-fork", args=[node.id]),
        {"class_name": "EditProbeNode", "new_class_name": new_class_name},
        format="json",
    )
    assert res.status_code == expected


def _flow_node(project, node_id, label, schema, modifications=None):
    data = {
        "label": label,
        "instanceName": node_id,
        "nodeType": "analysis",
        "file_name": f"{label}.py",
        "schema": schema,
    }
    if modifications:
        data["parameter_modifications"] = modifications
        data["has_parameter_modifications"] = True
    return FlowNode.objects.create(
        id=node_id,
        project=project,
        position_x=0,
        position_y=0,
        node_type="analysis",
        data=data,
    )


def _modified(original, current):
    return {
        "is_modified": True,
        "field_modifications": {
            "default_value_original": original,
            "default_value": {"current_value": current, "is_modified": True},
        },
    }


@pytest.fixture
def edited_project(user_alice, nodes_dir):
    """A project using EditProbeNode, whose code then dropped `size` and `ok`."""
    node = _create(user_alice, _source())
    old_schema = node.get_node_classes_for_frontend(user_alice)[0]["schema"]
    project = FlowProject.objects.create(name="p", owner=user_alice)
    a = _flow_node(
        project,
        "calc_a",
        "EditProbeNode",
        old_schema,
        {"rate": _modified(10, 25), "size": _modified(3, 5)},
    )
    b = _flow_node(project, "calc_b", "EditProbeNode", old_schema)
    _flow_node(project, "calc_other", "OtherNode", {"parameters": {}})
    FlowEdge.objects.create(
        id="e1",
        project=project,
        source_node=a,
        target_node=b,
        source_handle="calc_a::ok::output::str",
        target_handle="calc_b::in::input::str",
    )
    PythonFileService().update_file_content(
        node, _source(params={"rate": 10, "gain": 2}, outputs=("result",))
    )
    return project, node


def test_refresh_schema_keeps_edits_of_remaining_parameters(
    auth_client, user_alice, edited_project
):
    project, node = edited_project

    res = auth_client(user_alice).post(
        reverse("workflow:node-schema-refresh", args=[project.id]),
        {"file_id": str(node.id), "class_name": "EditProbeNode"},
        format="json",
    )
    assert res.status_code == 200, res.content
    body = res.json()
    assert sorted(body["updated"]) == ["calc_a", "calc_b"]
    assert body["dropped_parameters"] == {"calc_a": ["size"]}
    assert body["removed_ports"] == {"calc_a": ["ok"], "calc_b": ["ok"]}
    assert body["dangling_edges"] == ["e1"]

    a = FlowNode.objects.get(id="calc_a")
    params = a.data["schema"]["parameters"]
    assert set(params) == {"rate", "gain"}
    assert params["rate"]["default_value"] == 25
    assert set(a.data["parameter_modifications"]) == {"rate"}
    assert set(a.data["schema"]["outputs"]) == {"result"}
    other = FlowNode.objects.get(id="calc_other")
    assert other.data["schema"] == {"parameters": {}}


def test_refresh_schema_needs_write_access(
    auth_client, user_alice, user_bob, edited_project
):
    project, node = edited_project

    res = auth_client(user_bob).post(
        reverse("workflow:node-schema-refresh", args=[project.id]),
        {"file_id": str(node.id), "class_name": "EditProbeNode"},
        format="json",
    )
    assert res.status_code in (403, 404)
    assert FlowNode.objects.get(id="calc_a").data["schema"]["parameters"].get("size")


# --- project copies ------------------------------------------------------------


@pytest.fixture
def library_project(user_alice, nodes_dir):
    """A project using the unchanged library node EditProbeNode."""
    library = _create(user_alice, _source())
    schema = library.get_node_classes_for_frontend(user_alice)[0]["schema"]
    project = FlowProject.objects.create(name="p", owner=user_alice)
    a = _flow_node(
        project,
        "calc_a",
        "EditProbeNode",
        copy.deepcopy(schema),
        {"rate": _modified(10, 25), "size": _modified(3, 5)},
    )
    a.data["schema"]["parameters"]["rate"]["default_value"] = 25
    a.save()
    b = _flow_node(project, "calc_b", "EditProbeNode", copy.deepcopy(schema))
    FlowEdge.objects.create(
        id="e1",
        project=project,
        source_node=a,
        target_node=b,
        source_handle="calc_a::ok::output::str",
        target_handle="calc_b::in::input::str",
    )
    return project, library


def node_code_url(project, library=None, class_name="EditProbeNode"):
    url = reverse("workflow:project-node-code", args=[project.id, class_name])
    return f"{url}?file_id={library.id}" if library else url


def copy_file(project, class_name="EditProbeNode"):
    return projects_root() / str(project.id) / "project_nodes" / f"{class_name}.py"


def test_project_copy_starts_from_library_code(
    auth_client, user_alice, user_bob, library_project
):
    project, library = library_project

    body = auth_client(user_alice).get(node_code_url(project, library)).json()
    assert body["source"] == "library"
    assert body["code"] == _source()
    assert body["content_hash"] == _hash(_source())
    assert body["can_edit"] is True
    assert body["warnings"] == []
    assert body["jupyter_path"] == (
        f"codes/projects/{project.id}/project_nodes/EditProbeNode.py"
    )
    # Alice's project is private.
    assert auth_client(user_bob).get(node_code_url(project, library)).status_code == 404


def test_saving_writes_project_copy_and_leaves_library_alone(
    auth_client, user_alice, library_project, nodes_dir
):
    project, library = library_project
    edited = _source(params={"rate": 10, "gain": 2}, outputs=("result",))

    res = auth_client(user_alice).put(
        node_code_url(project, library),
        {"code": edited, "base_hash": _hash(_source())},
        format="json",
    )
    assert res.status_code == 200, res.content
    body = res.json()
    assert body["source"] == "project"
    assert sorted(body["updated"]) == ["calc_a", "calc_b"]
    assert body["dropped_parameters"] == {"calc_a": ["size"]}
    assert body["dangling_edges"] == ["e1"]

    assert copy_file(project).read_text() == edited
    library.refresh_from_db()
    assert library.file_content == _source()
    assert (nodes_dir / "analysis" / "EditProbeNode.py").read_text() == _source()

    a = FlowNode.objects.get(id="calc_a")
    assert set(a.data["schema"]["parameters"]) == {"rate", "gain"}
    assert a.data["schema"]["parameters"]["rate"]["default_value"] == 25
    assert a.data["projectNodePath"].endswith("project_nodes/EditProbeNode.py")

    # From now on the project copy is what the editor loads.
    again = auth_client(user_alice).get(node_code_url(project)).json()
    assert again["source"] == "project"
    assert again["code"] == edited


@pytest.mark.parametrize(
    "code, base_hash, expected",
    [
        (_source(class_name="Renamed"), None, 400),
        ("class EditProbeNode(:\n", None, 400),
        (_source(params={"rate": 1}), _hash("something else"), 409),
    ],
)
def test_project_copy_save_is_checked(
    auth_client, user_alice, library_project, code, base_hash, expected
):
    project, library = library_project

    res = auth_client(user_alice).put(
        node_code_url(project, library),
        {"code": code, "base_hash": base_hash},
        format="json",
    )
    assert res.status_code == expected
    assert not copy_file(project).exists()


def test_generated_workflow_imports_the_project_copy(
    auth_client, user_alice, library_project
):
    project, library = library_project
    auth_client(user_alice).put(
        node_code_url(project, library),
        {"code": _source(params={"rate": 10})},
        format="json",
    )
    nodes = [
        {
            "id": n.id,
            "type": n.node_type,
            "position": {"x": 0, "y": 0},
            "data": n.data,
        }
        for n in FlowNode.objects.filter(project=project)
    ]

    service = CodeGenerationService()
    assert service.generate_code_from_flow_data(
        str(project.id), project.name, nodes, []
    )
    code = service.get_code_file_path(project).read_text()
    assert "sys.path.insert(0, os.getcwd())" in code
    assert "from project_nodes.EditProbeNode import EditProbeNode" in code
    assert "from nodes.analysis.EditProbeNode" not in code


def test_library_refresh_skips_nodes_on_a_project_copy(
    auth_client, user_alice, library_project
):
    project, library = library_project
    auth_client(user_alice).put(
        node_code_url(project, library),
        {"code": _source(params={"rate": 10, "gain": 2})},
        format="json",
    )
    PythonFileService().update_file_content(library, _source(params={"other": 1}))

    res = auth_client(user_alice).post(
        reverse("workflow:node-schema-refresh", args=[project.id]),
        {"file_id": str(library.id), "class_name": "EditProbeNode"},
        format="json",
    )
    assert res.json()["updated"] == []
    params = FlowNode.objects.get(id="calc_b").data["schema"]["parameters"]
    assert set(params) == {"rate", "gain"}


def test_reverting_drops_the_copy_and_restores_library_schema(
    auth_client, user_alice, library_project
):
    project, library = library_project
    client = auth_client(user_alice)
    client.put(
        node_code_url(project, library),
        {"code": _source(params={"rate": 10, "gain": 2})},
        format="json",
    )

    res = client.delete(node_code_url(project, library))
    assert res.status_code == 200, res.content
    assert not copy_file(project).exists()
    a = FlowNode.objects.get(id="calc_a")
    assert "projectNodePath" not in a.data
    assert set(a.data["schema"]["parameters"]) == {"rate", "size"}
    assert a.data["schema"]["parameters"]["rate"]["default_value"] == 25


def test_dropped_node_of_a_copied_class_gets_the_copy_schema(
    auth_client, user_alice, library_project
):
    project, library = library_project
    client = auth_client(user_alice)
    client.put(
        node_code_url(project, library),
        {"code": _source(params={"gain": 2})},
        format="json",
    )
    library_schema = library.get_node_classes_for_frontend(user_alice)[0]["schema"]

    res = client.post(
        reverse("workflow:workflow-node-list-create", args=[project.id]),
        {
            "id": "calc_new",
            "position": {"x": 0, "y": 0},
            "type": "analysis",
            "data": {
                "label": "EditProbeNode",
                "instanceName": "new",
                "nodeType": "analysis",
                "file_name": "EditProbeNode.py",
                "schema": library_schema,
            },
        },
        format="json",
    )
    assert res.status_code == 201, res.content
    data = FlowNode.objects.get(id="calc_new").data
    assert set(data["schema"]["parameters"]) == {"gain"}
    assert data["projectNodePath"].endswith("project_nodes/EditProbeNode.py")


def test_project_copy_warns_about_file_relative_paths(
    auth_client, user_alice, library_project
):
    project, library = library_project
    code = _source() + "\nHERE = __file__\n"
    copy_file(project).parent.mkdir(parents=True)
    copy_file(project).write_text(code)

    body = auth_client(user_alice).get(node_code_url(project)).json()
    assert body["source"] == "project"
    assert body["warnings"]
