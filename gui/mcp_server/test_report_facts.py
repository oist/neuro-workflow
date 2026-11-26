import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _install_import_stubs() -> None:
    try:
        import httpx  # noqa: F401
    except ImportError:
        sys.modules["httpx"] = types.ModuleType("httpx")
    try:
        import fastmcp  # noqa: F401

        return
    except ImportError:
        pass

    fastmcp = types.ModuleType("fastmcp")

    class FastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self):
            def decorator(fn):
                return fn

            return decorator

    fastmcp.FastMCP = FastMCP
    server = types.ModuleType("fastmcp.server")
    dependencies = types.ModuleType("fastmcp.server.dependencies")
    dependencies.get_http_headers = lambda: {}
    sys.modules["fastmcp"] = fastmcp
    sys.modules["fastmcp.server"] = server
    sys.modules["fastmcp.server.dependencies"] = dependencies


_install_import_stubs()
import workflow_mcp


def _facts(responses: dict):
    async def fake_get(url: str):
        if url.endswith("/files/"):
            return responses.get("files")
        if url.endswith("/results/"):
            return responses.get("results")
        if url.endswith("/nodes/"):
            return responses.get("nodes")
        if url.endswith("/edges/"):
            return responses.get("edges")
        if url.endswith("/code/"):
            return responses.get("code")
        return responses.get("project")

    workflow_mcp._make_get_request = fake_get
    return asyncio.run(workflow_mcp.get_workflow_facts("wf"))


def test_facts_keep_a_doi_a_file_and_result_shapes():
    facts = _facts(
        {
            "project": {
                "name": "Visual cortex",
                "description": "A small simulation",
                "doi": "10.1000/example",
                "data_source": "",
                "license": "",
                "funding": "",
                "reference": "",
                "contributors": [],
                "links": [],
                "workflow_context": {
                    "simulation_io": {
                        "records": [{"path": "connectivity.npz", "source": "Allen Institute"}]
                    }
                },
            },
            "files": {"files": [{"filename": "connectivity.npz", "size_bytes": 12}]},
            "nodes": [],
            "edges": [],
            "results": {
                "results": [
                    {
                        "filename": "rates.npz",
                        "arrays": [{"name": "rates", "shape": [1000, 2]}],
                    }
                ]
            },
            "code": {"code": "pass", "notebook_outputs": []},
        }
    )
    assert facts["project"]["doi"] == "10.1000/example"
    assert facts["project"]["funding"] == ""
    assert facts["project_files"][0]["filename"] == "connectivity.npz"
    assert "inputs" not in facts
    assert facts["results"][0]["arrays"][0]["shape"] == [1000, 2]
    assert facts["simulation_io"]["records"][0]["source"] == "Allen Institute"


def test_a_missing_doi_stays_empty():
    facts = _facts(
        {
            "project": {"name": "Untitled", "description": None},
            "files": None,
            "nodes": None,
            "edges": None,
            "results": None,
            "code": None,
        }
    )
    assert facts["project"]["doi"] == ""
    assert facts["project_files"] == []
    assert facts["results"] == []
    assert facts["simulation_io"] == {}


def test_methods_instructions_are_in_the_prompt_and_the_button():
    prompt = (
        ROOT
        / "workflow_backend/django-project/app/chat/services/chat_orchestrator.py"
    ).read_text()
    button = (
        ROOT / "workflow_frontend/src/views/home/components/chatbotView.tsx"
    ).read_text()
    results_line = (
        "Name the result files the simulation writes, "
        "including array shapes when the facts include them."
    )
    project_files_line = (
        "Refer to the files in project_files as project files, and call one an input "
        "only when the facts confirm that the simulation reads it."
    )
    credit_line = (
        "When a fact includes a source, owner, paper, DOI, or acknowledgment, "
        "state that credit in the same sentence as the value, and do not invent a source."
    )
    for text in (prompt, button):
        assert results_line in text
        assert project_files_line in text
        assert credit_line in text
        assert "Name the files the simulation reads" not in text
