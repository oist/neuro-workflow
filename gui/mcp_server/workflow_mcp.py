import logging
import os
from typing import Any

import httpx
from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration via environment variables
DJANGO_API_URL = os.environ.get("DJANGO_API_URL", "http://localhost:8000/api")
DJANGO_API_TOKEN = os.environ.get("DJANGO_API_TOKEN")
USER_AGENT = os.environ.get("MCP_USER_AGENT", "workflow-mcp/1.0")


def _build_headers() -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if DJANGO_API_TOKEN:
        headers["Authorization"] = f"Bearer {DJANGO_API_TOKEN}"
    return headers


async def _make_get_request(url: str, timeout: float = 30.0) -> dict | None:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, headers=_build_headers(), timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"GET request failed for {url}: {e}")
            return None


async def _make_post_request(url: str, payload: dict | None = None, timeout: float = 60.0) -> dict | None:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(url, headers=_build_headers(), json=payload or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"POST request failed for {url}: {e}")
            return None


async def _make_put_request(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict | None:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.put(url, headers=_build_headers(), json=payload or {}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"PUT request failed for {url}: {e}")
            return None


async def _make_delete_request(url: str, timeout: float = 30.0) -> dict | None:
    async with httpx.AsyncClient() as client:
        try:
            r = await client.delete(url, headers=_build_headers(), timeout=timeout)
            r.raise_for_status()
            # Some DELETE endpoints may return empty body
            try:
                return r.json()
            except Exception:
                return {"status": "deleted"}
        except Exception as e:
            logger.error(f"DELETE request failed for {url}: {e}")
            return None


mcp = FastMCP("workflow")


# Project endpoints
@mcp.tool()
async def list_projects() -> dict[str, Any]:
    """List all workflows (projects).

    Returns a dict with status and the list of projects under the "projects" key.
    """
    url = f"{DJANGO_API_URL}/workflow/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to fetch projects"}
    return {"status": "success", "projects": data}


@mcp.tool()
async def create_project(payload: dict) -> dict[str, Any]:
    """Create a new workflow/project.

    Args:
        payload: JSON-serializable dict containing project fields expected by the backend.

    Returns a dict with status and the created project under the "project" key.
    """
    url = f"{DJANGO_API_URL}/workflow/"
    data = await _make_post_request(url, payload)
    if data is None:
        return {"status": "error", "error": "Failed to create project"}
    return {"status": "success", "project": data}


@mcp.tool()
async def get_project(workflow_id: str) -> dict[str, Any]:
    """Retrieve a single workflow/project by its ID.

    Args:
        workflow_id: The ID of the workflow to fetch.

    Returns a dict with status and the project under the "project" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch project {workflow_id}"}
    return {"status": "success", "project": data}


@mcp.tool()
async def update_project(workflow_id: str, payload: dict) -> dict[str, Any]:
    """Update an existing workflow/project.

    Args:
        workflow_id: The ID of the workflow to update.
        payload: JSON-serializable dict with updated project fields.

    Returns a dict with status and the updated project under the "project" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/"
    data = await _make_put_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update project {workflow_id}"}
    return {"status": "success", "project": data}


@mcp.tool()
async def delete_project(workflow_id: str) -> dict[str, Any]:
    """Delete a workflow/project by its ID.

    Args:
        workflow_id: The ID of the workflow to delete.

    Returns a dict with status and deletion result under the "result" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/"
    data = await _make_delete_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to delete project {workflow_id}"}
    return {"status": "success", "result": data}


# Flow endpoints
@mcp.tool()
async def get_flow(workflow_id: str) -> dict[str, Any]:
    """Fetch the flow definition for a given workflow.

    Args:
        workflow_id: The ID of the workflow whose flow to fetch.

    Returns a dict with status and the flow under the "flow" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/flow/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch flow for {workflow_id}"}
    return {"status": "success", "flow": data}


@mcp.tool()
async def update_flow(workflow_id: str, flow_payload: dict) -> dict[str, Any]:
    """Update the flow definition for a given workflow.

    Args:
        workflow_id: The ID of the workflow to update.
        flow_payload: JSON-serializable dict describing the new flow.

    Returns a dict with status and the updated flow under the "flow" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/flow/"
    data = await _make_put_request(url, flow_payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update flow for {workflow_id}"}
    return {"status": "success", "flow": data}


# Node endpoints
@mcp.tool()
async def list_nodes(workflow_id: str) -> dict[str, Any]:
    """List all nodes belonging to a workflow.

    Args:
        workflow_id: The ID of the workflow.

    Returns a dict with status and the list of nodes under the "nodes" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to list nodes for {workflow_id}"}
    return {"status": "success", "nodes": data}


@mcp.tool()
async def create_node(workflow_id: str, payload: dict) -> dict[str, Any]:
    """Create a new node within a workflow.

    Args:
        workflow_id: The ID of the workflow.
        payload: JSON-serializable dict containing node data.

    Returns a dict with status and the created node under the "node" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/"
    data = await _make_post_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to create node for {workflow_id}"}
    return {"status": "success", "node": data}


@mcp.tool()
async def get_node(workflow_id: str, node_id: str) -> dict[str, Any]:
    """Retrieve a single node by ID within a workflow.

    Args:
        workflow_id: The ID of the workflow.
        node_id: The ID of the node to retrieve.

    Returns a dict with status and the node under the "node" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch node {node_id} for {workflow_id}"}
    return {"status": "success", "node": data}


@mcp.tool()
async def update_node(workflow_id: str, node_id: str, payload: dict) -> dict[str, Any]:
    """Update a node within a workflow.

    Args:
        workflow_id: The ID of the workflow.
        node_id: The ID of the node to update.
        payload: JSON-serializable dict with updated node fields.

    Returns a dict with status and the updated node under the "node" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/"
    data = await _make_put_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update node {node_id} for {workflow_id}"}
    return {"status": "success", "node": data}


@mcp.tool()
async def delete_node(workflow_id: str, node_id: str) -> dict[str, Any]:
    """Delete a node from a workflow.

    Args:
        workflow_id: The ID of the workflow.
        node_id: The ID of the node to delete.

    Returns a dict with status and deletion result under the "result" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/"
    data = await _make_delete_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to delete node {node_id} for {workflow_id}"}
    return {"status": "success", "result": data}


# Edge endpoints
@mcp.tool()
async def list_edges(workflow_id: str) -> dict[str, Any]:
    """List all edges for a workflow.

    Args:
        workflow_id: The ID of the workflow.

    Returns a dict with status and the list of edges under the "edges" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/edges/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to list edges for {workflow_id}"}
    return {"status": "success", "edges": data}


@mcp.tool()
async def create_edge(workflow_id: str, payload: dict) -> dict[str, Any]:
    """Create a new edge in a workflow.

    Args:
        workflow_id: The ID of the workflow.
        payload: JSON-serializable dict containing edge details.

    Returns a dict with status and the created edge under the "edge" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/edges/"
    data = await _make_post_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to create edge for {workflow_id}"}
    return {"status": "success", "edge": data}


@mcp.tool()
async def delete_edge(workflow_id: str, edge_id: str) -> dict[str, Any]:
    """Delete an edge from a workflow.

    Args:
        workflow_id: The ID of the workflow.
        edge_id: The ID of the edge to delete.

    Returns a dict with status and deletion result under the "result" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/edges/{edge_id}/"
    data = await _make_delete_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to delete edge {edge_id} for {workflow_id}"}
    return {"status": "success", "result": data}


# Node parameter update (matches existing helper in workflow_mcp.py)
@mcp.tool()
async def update_node_parameter(workflow_id: str, node_id: str, parameter_key: str, parameter_value: Any, parameter_field: str = "value") -> dict[str, Any]:
    """Update a specific parameter for a node.

    Args:
        workflow_id: The ID of the workflow.
        node_id: The ID of the node.
        parameter_key: The key/name of the parameter to update.
        parameter_value: The new value for the parameter.
        parameter_field: The field of the parameter to update (defaults to "value").

    Returns a dict with status and the update result under the "result" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/parameters/"
    payload = {
        "parameter_key": parameter_key,
        "parameter_value": parameter_value,
        "parameter_field": parameter_field,
    }
    data = await _make_put_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update parameter {parameter_key} for node {node_id}"}
    return {"status": "success", "result": data}


# Batch code generation
@mcp.tool()
async def generate_code_batch(workflow_id: str, nodes: list[Any], edges: list[Any]) -> dict[str, Any]:
    """Trigger batch code generation for a workflow.

    Args:
        workflow_id: The ID of the workflow.
        nodes: A list of node representations for code generation.
        edges: A list of edge representations for code generation.

    Returns a dict with status and the generation result under the "result" key.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/generate-code/"
    payload = {"nodes": nodes, "edges": edges}
    data = await _make_post_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to trigger code generation for {workflow_id}"}
    return {"status": "success", "result": data}


# Sample flow / health
@mcp.tool()
async def get_sample_flow() -> dict[str, Any]:
    """Fetch a sample flow used for testing/health-checks.

    Returns a dict with status and the sample flow under the "sample_flow" key.
    """
    url = f"{DJANGO_API_URL}/workflow/sample-flow/"
    data = await _make_get_request(url, timeout=10.0)
    if data is None:
        return {"status": "error", "error": "Backend unreachable or sample flow not available"}
    return {"status": "success", "sample_flow": data}


@mcp.tool()
async def health() -> dict[str, Any]:
    """Simple health check for the backend by calling sample-flow.

    Returns a dict indicating whether the backend is reachable.
    """
    # Keep a convenience health tool that calls sample-flow
    resp = await get_sample_flow()
    if resp.get("status") != "success":
        return {"status": "error", "error": "Backend unreachable"}
    return {"status": "ok", "backend": "reachable"}


# File upload
@mcp.tool()
async def upload_python_file(file_payload: dict) -> dict[str, Any]:
    """Upload a Python file to the backend.

    Args:
        file_payload: Dict containing fields expected by the backend (e.g. file path, metadata or upload reference).

    Note:
        If the backend expects multipart/form-data, handle upload on the frontend and pass a JSON payload
        describing the uploaded file location or metadata to this tool.

    Returns:
        Dict with status and either "file" containing backend response or an "error" message.
    """
    url = f"{DJANGO_API_URL}/box/upload/"
    data = await _make_post_request(url, file_payload)
    if data is None:
        return {"status": "error", "error": "Failed to upload python file"}
    return {"status": "success", "file": data}


# File list / detail
@mcp.tool()
async def list_python_files() -> dict[str, Any]:
    """List uploaded Python files.

    Returns:
        Dict with status and "files" containing the list of files from the backend.
    """
    url = f"{DJANGO_API_URL}/box/files/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to list python files"}
    return {"status": "success", "files": data}


@mcp.tool()
async def get_python_file(pk: str) -> dict[str, Any]:
    """Fetch file metadata/detail by primary key.

    Args:
        pk: Primary key or identifier of the file.

    Returns:
        Dict with status and "file" containing file detail, or error on failure.
    """
    url = f"{DJANGO_API_URL}/box/files/{pk}/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch python file {pk}"}
    return {"status": "success", "file": data}


# list of node definitions
@mcp.tool()
async def list_nodes_definitions() -> dict[str, Any]:
    """Return list of node definitions.

    Returns:
        Dict with status and "nodes" containing node metadata.
    """
    url = f"{DJANGO_API_URL}/box/uploaded-nodes/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to fetch nodes"}
    return {"status": "success", "nodes": data}


# File code management (get/put)
@mcp.tool()
async def get_python_file_code(filename: str) -> dict[str, Any]:
    """Retrieve the source code for a given uploaded file.

    Args:
        filename: Name or identifier of the file whose code to fetch.

    Returns:
        Dict with status and "code" containing the file code or an error message.
    """
    url = f"{DJANGO_API_URL}/box/files/{filename}/code/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch code for {filename}"}
    return {"status": "success", "code": data}


@mcp.tool()
async def update_python_file_code(filename: str, code_payload: dict) -> dict[str, Any]:
    """Update the source code for an uploaded file.

    Args:
        filename: Name or identifier of the file to update.
        code_payload: Dict containing the new code or related fields as expected by the backend.

    Returns:
        Dict with status and "result" containing backend response or an error message.
    """
    url = f"{DJANGO_API_URL}/box/files/{filename}/code/"
    data = await _make_put_request(url, code_payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update code for {filename}"}
    return {"status": "success", "result": data}


# File copy
@mcp.tool()
async def copy_python_file(copy_payload: dict) -> dict[str, Any]:
    """Copy an existing uploaded file to create a new file.

    Args:
        copy_payload: Dict specifying source file and destination info as required by backend.

    Returns:
        Dict with status and backend "result" or an error message.
    """
    url = f"{DJANGO_API_URL}/box/copy/"
    data = await _make_post_request(url, copy_payload)
    if data is None:
        return {"status": "error", "error": "Failed to copy python file"}
    return {"status": "success", "result": data}


# Parameter update
@mcp.tool()
async def update_python_file_parameter(param_payload: dict) -> dict[str, Any]:
    """Update a parameter in an uploaded Python file's metadata.

    Args:
        param_payload: Dict containing parameter key/value information expected by the backend.

    Returns:
        Dict with status and backend "result" or an error message.
    """
    url = f"{DJANGO_API_URL}/box/parameters/update/"
    data = await _make_put_request(url, param_payload)
    if data is None:
        return {"status": "error", "error": "Failed to update python file parameter"}
    return {"status": "success", "result": data}


# Categories
@mcp.tool()
async def node_categories() -> dict[str, Any]:
    """Retrieve node categories available in the box.

    Returns:
        Dict with status and "categories" list from the backend.
    """
    url = f"{DJANGO_API_URL}/box/categories/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to fetch node categories"}
    return {"status": "success", "categories": data}


# Bulk sync nodes
@mcp.tool()
async def bulk_sync_nodes(sync_payload: dict | None = None) -> dict[str, Any]:
    """Synchronize multiple nodes in bulk with the backend.

    Args:
        sync_payload: Optional dict specifying nodes to sync; if omitted an empty payload is sent.

    Returns:
        Dict with status and backend "result" or an error message.
    """
    url = f"{DJANGO_API_URL}/box/sync/"
    data = await _make_post_request(url, sync_payload or {})
    if data is None:
        return {"status": "error", "error": "Failed to bulk sync nodes"}
    return {"status": "success", "result": data}


if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except Exception:
        logger.exception("Failed to run FastMCP server")

