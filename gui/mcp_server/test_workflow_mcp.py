"""Path identifiers passed to the MCP tools cannot leave their workflow.

The notebook proxy pins ``workflow_id`` to the relayed project, so another
argument interpolated into the URL (``node_id="../../<other>/nodes/x"``) must
not be able to reach a different project. Run with:

    uv run --with fastmcp --with pytest pytest gui/mcp_server
"""
import asyncio

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

import workflow_mcp

WF = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
BASE = f"{workflow_mcp.DJANGO_API_URL}/workflow/{WF}/"


@pytest.fixture
def requested(monkeypatch):
    """Record every URL the tools would request from Django."""
    urls = []

    async def fake(url, *args, **kwargs):
        urls.append(url)
        return {}

    async def fake_detail(method, url, *args, **kwargs):
        urls.append(url)
        return {}, None

    for name in (
        "_make_get_request",
        "_make_put_request",
        "_make_post_request",
        "_make_delete_request",
    ):
        monkeypatch.setattr(workflow_mcp, name, fake)
    monkeypatch.setattr(workflow_mcp, "_make_request_with_detail", fake_detail)
    return urls


def _call(tool, **arguments):
    async def run():
        async with Client(workflow_mcp.mcp) as client:
            return await client.call_tool(tool, {"workflow_id": WF, **arguments})

    return asyncio.run(run())


TRAVERSAL = f"../../{OTHER}/nodes/x"

CASES = [
    ("get_node", {"node_id": TRAVERSAL}),
    ("update_node", {"node_id": TRAVERSAL, "payload": {}}),
    ("delete_node", {"node_id": TRAVERSAL}),
    (
        "update_node_parameter",
        {"node_id": TRAVERSAL, "parameter_key": "k", "parameter_value": 1},
    ),
    ("update_node_instance_name", {"node_id": TRAVERSAL, "instance_name": "n"}),
    ("delete_edge", {"edge_id": TRAVERSAL}),
    ("add_edge", {"source_node_id": TRAVERSAL, "target_node_id": TRAVERSAL}),
    ("remove_study_objective", {"name": TRAVERSAL}),
]


@pytest.mark.parametrize("tool,arguments", CASES, ids=[c[0] for c in CASES])
def test_slashes_stay_inside_one_segment(requested, tool, arguments):
    try:
        _call(tool, **arguments)
    except ToolError:
        pass  # e.g. add_edge failing on the empty fake node; URLs are recorded
    assert requested
    for url in requested:
        # httpx resolves "/../" segments; an encoded slash is left alone.
        raw_path = httpx.URL(url).raw_path.decode()
        assert raw_path.startswith(f"/api/workflow/{WF}/")
        assert f"..%2F..%2F{OTHER}%2Fnodes%2Fx" in raw_path


@pytest.mark.parametrize("bad", ["..", ".", ""])
@pytest.mark.parametrize(
    "tool,field",
    [
        ("get_node", "node_id"),
        ("delete_node", "node_id"),
        ("delete_edge", "edge_id"),
        ("remove_study_objective", "name"),
    ],
)
def test_dot_segments_are_rejected(requested, tool, field, bad):
    with pytest.raises(ToolError, match="Invalid identifier"):
        _call(tool, **{field: bad})
    assert requested == []


def test_regular_ids_are_unchanged(requested):
    _call("get_node", node_id="calc_1700000000000_ab12")
    assert requested == [f"{BASE}nodes/calc_1700000000000_ab12/"]
