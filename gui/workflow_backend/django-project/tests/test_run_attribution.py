import base64
import json

from app.workflow.run_attribution import NodeAttributor, load_var_to_node

VAR_TO_NODE = {
    "instance_TVBVisualizationNode_001": "calc_100",
    "instance_NW_Analysis_002": "calc_200",
}

PNG = base64.b64encode(b"fake-png-bytes").decode()


def stdout(content):
    return {"type": "stdout", "data": {"content": content}}


def image(content=PNG):
    return {"type": "image", "data": {"content": content, "mime": "image/png"}}


def test_marker_emits_node_executing_and_attributes_images():
    a = NodeAttributor(VAR_TO_NODE)

    events = a.process_event(
        stdout("Executing node: instance_TVBVisualizationNode_001\n")
    )
    assert [e["type"] for e in events] == ["stdout", "node_executing"]
    assert events[1]["data"] == {
        "node_name": "instance_TVBVisualizationNode_001",
        "node_id": "calc_100",
    }

    (img,) = a.process_event(image())
    assert img["data"]["node_id"] == "calc_100"
    assert img["data"]["figure_index"] == 0

    (img2,) = a.process_event(image())
    assert img2["data"]["figure_index"] == 1


def test_marker_split_across_chunks():
    a = NodeAttributor(VAR_TO_NODE)
    a.process_event(stdout("Executing node: instance_NW_"))
    events = a.process_event(stdout("Analysis_002\n"))
    assert events[-1]["type"] == "node_executing"
    assert events[-1]["data"]["node_id"] == "calc_200"


def test_error_marker_does_not_match():
    a = NodeAttributor(VAR_TO_NODE)
    a.process_event(stdout("Executing node: instance_TVBVisualizationNode_001\n"))
    events = a.process_event(stdout("Error executing node: instance_NW_Analysis_002\n"))
    assert [e["type"] for e in events] == ["stdout"]
    (img,) = a.process_event(image())
    assert img["data"]["node_id"] == "calc_100"


def test_unknown_var_and_no_marker_yield_null_node_id():
    a = NodeAttributor(VAR_TO_NODE)

    (img,) = a.process_event(image())
    assert img["data"]["node_id"] is None

    events = a.process_event(stdout("Executing node: instance_deleted_999\n"))
    assert events[1]["data"]["node_id"] is None
    (img2,) = a.process_event(image())
    assert img2["data"]["node_id"] is None
    # Unattributed figures share one counter bucket
    assert img2["data"]["figure_index"] == 1


def test_figures_teed_to_disk_with_manifest(tmp_path):
    figures_dir = tmp_path / "results" / "figures"
    # Pre-existing figures from an earlier run are wiped at construction
    stale = figures_dir / "calc_old"
    stale.mkdir(parents=True)
    (stale / "fig_000.png").write_bytes(b"stale")

    a = NodeAttributor(VAR_TO_NODE, figures_dir)
    assert not stale.exists()

    a.process_event(stdout("Executing node: instance_TVBVisualizationNode_001\n"))
    a.process_event(image())
    a.process_event(image())
    a.write_manifest("ok")

    saved = sorted(p.name for p in (figures_dir / "calc_100").iterdir())
    assert saved == ["fig_000.png", "fig_001.png"]
    assert (figures_dir / "calc_100" / "fig_000.png").read_bytes() == b"fake-png-bytes"

    manifest = json.loads((figures_dir / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    assert len(manifest["figures"]) == 2
    assert manifest["figures"][0]["path"] == "results/figures/calc_100/fig_000.png"
    assert manifest["figures"][0]["node_id"] == "calc_100"


def test_unattributed_figures_saved_separately(tmp_path):
    figures_dir = tmp_path / "figures"
    a = NodeAttributor({}, figures_dir)
    a.process_event(image())
    a.write_manifest("aborted")

    assert (figures_dir / "_unattributed" / "fig_000.png").exists()
    manifest = json.loads((figures_dir / "manifest.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["figures"][0]["node_id"] is None


def test_load_var_to_node_missing_and_corrupt(tmp_path):
    assert load_var_to_node(tmp_path) == {}

    (tmp_path / "node_map.json").write_text("not json")
    assert load_var_to_node(tmp_path) == {}

    (tmp_path / "node_map.json").write_text(
        json.dumps({"version": 1, "var_to_node": {"v": "n"}})
    )
    assert load_var_to_node(tmp_path) == {"v": "n"}
