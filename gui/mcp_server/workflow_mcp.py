import copy
import logging
import os
import secrets
import time
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration via environment variables
DJANGO_API_URL = os.environ.get("DJANGO_API_URL", "http://localhost:3000/api")
# DJANGO_API_TOKEN is a fallback used only when no end-user Authorization
# header is forwarded (e.g. CLI / stdio invocation, health checks).
# In production the user's JWT is propagated per-request via get_http_headers().
DJANGO_API_TOKEN = os.environ.get("DJANGO_API_TOKEN")
USER_AGENT = os.environ.get("MCP_USER_AGENT", "workflow-mcp/1.0")
MCP_PORT = int(os.environ.get("MCP_PORT", 8001))


def _build_headers() -> dict[str, str]:
    """Forward the caller's Authorization header (per-user JWT) to Django.

    Falls back to DJANGO_API_TOKEN env when called outside an HTTP request
    context or when the caller did not supply Authorization.
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    try:
        # FastMCP excludes "authorization" by default to avoid accidental
        # leakage; explicitly opt in so we can forward the user's JWT.
        incoming = get_http_headers(include={"authorization"}) or {}
    except Exception:
        incoming = {}
    auth = incoming.get("authorization")
    if auth:
        headers["Authorization"] = auth
    elif DJANGO_API_TOKEN:
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


async def _make_multipart_post_request(
    url: str, files: dict, data: dict | None = None, timeout: float = 60.0
) -> dict | None:
    # Reuse the standard header builder, then drop Content-Type so httpx
    # can set the multipart boundary itself.
    headers = _build_headers()
    headers.pop("Content-Type", None)
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                url, headers=headers, files=files, data=data or {}, timeout=timeout
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Multipart POST failed for {url}: {e}")
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
    """List all active workflow projects.

    Returns an array of project objects, each containing:
    - id (UUID): Project identifier.
    - name (str): Project name.
    - description (str): Project description.
    - workflow_context (dict): Arbitrary workflow metadata.
    - owner (dict): Owner user object with id, username, email, first_name, last_name.
    - created_at / updated_at (datetime): Timestamps.
    - nodes_count (int): Number of nodes in the project.
    - edges_count (int): Number of edges in the project.
    """
    url = f"{DJANGO_API_URL}/workflow/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to fetch projects"}
    return {"status": "success", "projects": data}


@mcp.tool()
async def create_project(payload: dict) -> dict[str, Any]:
    """Create a new workflow project.

    Args:
        payload: Project fields. Required: {"name": str}. Optional: {"description": str, "workflow_context": dict}.

    Returns the created project object with id, name, description, workflow_context,
    owner, created_at, updated_at, nodes_count, and edges_count.
    """
    url = f"{DJANGO_API_URL}/workflow/"
    data = await _make_post_request(url, payload)
    if data is None:
        return {"status": "error", "error": "Failed to create project"}
    return {"status": "success", "project": data}


@mcp.tool()
async def get_project(workflow_id: str) -> dict[str, Any]:
    """Retrieve a single workflow project by its UUID.

    Args:
        workflow_id: UUID of the project to fetch.

    Returns the full project object including id, name, description,
    workflow_context, owner, timestamps, nodes_count, and edges_count.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch project {workflow_id}"}
    return {"status": "success", "project": data}


@mcp.tool()
async def update_project(workflow_id: str, payload: dict) -> dict[str, Any]:
    """Update an existing workflow project (full replacement).

    Args:
        workflow_id: UUID of the project to update.
        payload: Fields to update. Accepted: {"name": str, "description": str, "workflow_context": dict}.

    Returns the updated project object.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/"
    data = await _make_put_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update project {workflow_id}"}
    return {"status": "success", "project": data}


# Flow endpoints
@mcp.tool()
async def get_flow(workflow_id: str) -> dict[str, Any]:
    """Fetch the complete flow definition (all nodes and edges) for a workflow.

    Returns a dict with "nodes" and "edges" arrays representing the entire
    React Flow graph. Each node contains id, position ({x, y}), type, and data
    (including label, schema with inputs/outputs/parameters/methods, and instanceName).
    Each edge contains id, source, target, sourceHandle, and targetHandle.

    Args:
        workflow_id: UUID of the workflow whose flow to fetch.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/flow/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch flow for {workflow_id}"}
    return {"status": "success", "flow": data}


@mcp.tool()
async def update_flow(workflow_id: str, flow_payload: dict) -> dict[str, Any]:
    """Save (overwrite) the flow definition for a workflow.

    Replaces all nodes and edges in the project with the provided data.
    Note: This does NOT trigger code generation. Call generate_code_batch separately.

    Args:
        workflow_id: UUID of the workflow to update.
        flow_payload: Dict with "nodes" and "edges" arrays. Each node requires:
            {"id": str, "position": {"x": float, "y": float},
             "data": {"label": str, "nodeType": str, ...}}.
            Each edge requires: {"id": str, "source": str, "target": str}.
            Optional edge fields: sourceHandle, targetHandle, data.

            IMPORTANT: Every node's data MUST include "nodeType" with a valid category
            name (e.g. "analysis", "io", "network", "optimization", "simulation",
            "stimulus"). Missing or invalid nodeType will cause a 400 error.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/flow/"
    data = await _make_put_request(url, flow_payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update flow for {workflow_id}"}
    return {"status": "success", "flow": data}


# Node endpoints
@mcp.tool()
async def list_nodes(workflow_id: str) -> dict[str, Any]:
    """List all nodes in a workflow project.

    Returns an array of node objects, each containing:
    - id (str): Node identifier (used in React Flow).
    - project (UUID): Parent project ID.
    - position_x / position_y (float): Canvas coordinates.
    - node_type (str): React Flow node type.
    - data (dict): Full React Flow node data including label, schema
      (inputs, outputs, parameters, methods), instanceName, and parameter_modifications.
    - has_parameter_modifications (bool): Whether any parameters have been modified.
    - modified_parameters (dict): Map of modified parameter keys to their values.
    - created_at / updated_at (datetime): Timestamps.

    Args:
        workflow_id: UUID of the workflow.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to list nodes for {workflow_id}"}
    return {"status": "success", "nodes": data}


@mcp.tool()
async def add_node(
    workflow_id: str,
    node_name: str,
    position_x: float = 0.0,
    position_y: float = 0.0,
    instance_name: str = "",
) -> dict[str, Any]:
    """High-level tool to add a node to a workflow by name, automatically resolving all metadata.

    Unlike create_node (which requires a fully constructed payload), this tool only needs
    the node class name and position. It automatically looks up the node definition,
    resolves the schema, category color, file name, and generates a unique ID — mirroring
    exactly what the frontend does when a user drags a node from the sidebar.

    Args:
        workflow_id: UUID of the workflow to add the node to.
        node_name: Node class name (e.g. "PoissonGenerator", "IafPscAlpha").
            Matching priority: exact class_name > exact label > exact id > case-insensitive class_name.
        position_x: X coordinate on the canvas. Defaults to 0.0.
        position_y: Y coordinate on the canvas. Defaults to 0.0.
        instance_name: Display name and variable name for the node instance.
            Defaults to the node's label (class name) if not provided.

    Returns a dict with status and the created node object on success,
    or status "error" with available node names on failure.
    """
    # 1. Fetch node definitions and category colors
    defs_url = f"{DJANGO_API_URL}/box/uploaded-nodes/"
    defs_data = await _make_get_request(defs_url)
    if defs_data is None:
        return {"status": "error", "error": "Failed to fetch node definitions from backend"}

    nodes_list = defs_data.get("nodes", [])
    categories = defs_data.get("categories", {})

    if not nodes_list:
        return {"status": "error", "error": "No node definitions available. Upload node files first."}

    # 2. Match node_name with priority: class_name > label > id > case-insensitive class_name
    matches = [n for n in nodes_list if n.get("class_name") == node_name]
    if not matches:
        matches = [n for n in nodes_list if n.get("label") == node_name]
    if not matches:
        matches = [n for n in nodes_list if n.get("id") == node_name]
    if not matches:
        name_lower = node_name.lower()
        matches = [n for n in nodes_list if (n.get("class_name") or "").lower() == name_lower]

    if not matches:
        available = sorted(set(n.get("class_name", n.get("label", "unknown")) for n in nodes_list))
        return {
            "status": "error",
            "error": f"Node '{node_name}' not found. Available nodes: {available}",
        }

    if len(matches) > 1:
        candidates = [n.get("class_name", n.get("label", "unknown")) for n in matches]
        return {
            "status": "error",
            "error": f"Ambiguous node name '{node_name}'. Matches: {candidates}",
        }

    node_def = matches[0]

    # 3. Resolve metadata from the matched definition
    label = node_def.get("label") or node_def.get("class_name", node_name)
    class_name = node_def.get("class_name", label)
    file_name = node_def.get("file_name", f"{class_name}.py")
    if not file_name.endswith(".py"):
        file_name += ".py"
    category = node_def.get("category", "analysis")
    # Normalize to match frontend WorkflowCanvas.tsx:260 (lowercase + remove '/')
    # so node.type matches the keys registered in homeView's nodeTypes.
    node_type = category.lower().replace("/", "")
    schema = copy.deepcopy(node_def.get("schema", {"inputs": {}, "outputs": {}, "parameters": {}, "methods": {}}))
    resolved_instance_name = instance_name or label

    # 4. Resolve color from categories (cat_settings is keyed by DB value, i.e. normalized)
    cat_info = categories.get(node_type, {})
    color = cat_info.get("color", "#6b46c1")

    # 5. Generate unique node ID (matching frontend format)
    timestamp_ms = int(time.time() * 1000)
    random_hex = secrets.token_hex(5)
    node_id = f"calc_{timestamp_ms}_{random_hex}"

    # 6. Build payload matching frontend structure
    payload = {
        "id": node_id,
        "position": {"x": position_x, "y": position_y},
        "type": node_type,
        "data": {
            "file_name": file_name,
            "label": label,
            "instanceName": resolved_instance_name,
            "schema": schema,
            "nodeType": node_type,
            "nodeParameters": {},
            "color": color,
        },
    }

    # 7. Create the node via API
    create_url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/"
    data = await _make_post_request(create_url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to create node '{node_name}' for workflow {workflow_id}"}
    return {"status": "success", "node": data}


@mcp.tool()
async def get_node(workflow_id: str, node_id: str) -> dict[str, Any]:
    """Retrieve a single node by its ID within a workflow.

    Returns the full node object including position, type, data (with schema,
    instanceName, parameter_modifications), and computed fields
    (has_parameter_modifications, modified_parameters, parameter_modification_count).

    NOTE: In each parameter object, always read "default_value" as the current value.
    The "value" field is legacy and may be stale — ignore it.

    Args:
        workflow_id: UUID of the workflow.
        node_id: String ID of the node (React Flow node ID).
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch node {node_id} for {workflow_id}"}
    return {"status": "success", "node": data}


@mcp.tool()
async def update_node(workflow_id: str, node_id: str, payload: dict) -> dict[str, Any]:
    """Update a node's properties within a workflow (full replacement).

    Args:
        workflow_id: UUID of the workflow.
        node_id: String ID of the node to update.
        payload: Fields to update. Accepted:
            {"position": {"x": float, "y": float}, "type": str, "data": dict}.
            The data dict replaces the entire node data including label, schema, and instanceName.

            IMPORTANT: If "data" is provided, it MUST include "nodeType" with a valid
            category name (e.g. "analysis", "io", "network", "optimization", "simulation",
            "stimulus"). Missing or invalid nodeType will cause a 400 error.

    Returns the updated node object.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/"
    data = await _make_put_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update node {node_id} for {workflow_id}"}
    return {"status": "success", "node": data}


@mcp.tool()
async def delete_node(workflow_id: str, node_id: str) -> dict[str, Any]:
    """Delete a node and all its associated edges from a workflow.

    Permanently removes the node and any edges connected to it (as source or target).

    Args:
        workflow_id: UUID of the workflow.
        node_id: String ID of the node to delete.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/"
    data = await _make_delete_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to delete node {node_id} for {workflow_id}"}
    return {"status": "success", "result": data}


# Edge endpoints
@mcp.tool()
async def list_edges(workflow_id: str) -> dict[str, Any]:
    """List all edges (connections between nodes) in a workflow.

    Returns an array of edge objects, each containing:
    - id (str): Edge identifier.
    - project (UUID): Parent project ID.
    - source_node (str): Source node ID.
    - target_node (str): Target node ID.
    - source_handle (str|null): Source port handle ID.
    - target_handle (str|null): Target port handle ID.
    - edge_data (dict): Additional edge metadata.
    - created_at (datetime): Timestamp.

    Args:
        workflow_id: UUID of the workflow.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/edges/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to list edges for {workflow_id}"}
    return {"status": "success", "edges": data}


@mcp.tool()
async def add_edge(
    workflow_id: str,
    source_node_id: str,
    target_node_id: str,
    source_port_name: str = "",
    target_port_name: str = "",
) -> dict[str, Any]:
    """High-level tool to connect two nodes in a workflow.

    Automatically resolves handle IDs from node schemas and generates a
    proper edge ID — mirroring what the frontend does when a user drags
    a connection between ports.

    If source/target nodes each have only one output/input port,
    port names can be omitted and will be auto-resolved.
    If multiple ports exist, specify port names to select the correct ones.

    Args:
        workflow_id: UUID of the workflow.
        source_node_id: ID of the source node (the node with the output port).
        target_node_id: ID of the target node (the node with the input port).
        source_port_name: Name of the output port on the source node.
            Optional if the source node has only one output.
        target_port_name: Name of the input port on the target node.
            Optional if the target node has only one input.

    Returns a dict with status and the created edge object on success.
    """
    HANDLE_SEPARATOR = "::"

    # 1. Fetch both nodes
    src_url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{source_node_id}/"
    tgt_url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{target_node_id}/"
    source_node = await _make_get_request(src_url)
    target_node = await _make_get_request(tgt_url)

    if source_node is None:
        return {"status": "error", "error": f"Source node '{source_node_id}' not found in workflow {workflow_id}"}
    if target_node is None:
        return {"status": "error", "error": f"Target node '{target_node_id}' not found in workflow {workflow_id}"}

    # 2. Extract schemas
    src_schema = (source_node.get("data") or {}).get("schema") or {}
    tgt_schema = (target_node.get("data") or {}).get("schema") or {}
    src_outputs = src_schema.get("outputs") or {}
    tgt_inputs = tgt_schema.get("inputs") or {}

    # 3. Resolve source output port
    if source_port_name:
        if source_port_name not in src_outputs:
            return {
                "status": "error",
                "error": f"Output port '{source_port_name}' not found on source node '{source_node_id}'. "
                         f"Available outputs: {list(src_outputs.keys())}",
            }
        src_port_key = source_port_name
    else:
        if len(src_outputs) == 0:
            return {"status": "error", "error": f"Source node '{source_node_id}' has no output ports."}
        if len(src_outputs) == 1:
            src_port_key = next(iter(src_outputs))
        else:
            return {
                "status": "error",
                "error": f"Source node '{source_node_id}' has multiple output ports. "
                         f"Specify source_port_name. Available: {list(src_outputs.keys())}",
            }

    # 4. Resolve target input port
    if target_port_name:
        if target_port_name not in tgt_inputs:
            return {
                "status": "error",
                "error": f"Input port '{target_port_name}' not found on target node '{target_node_id}'. "
                         f"Available inputs: {list(tgt_inputs.keys())}",
            }
        tgt_port_key = target_port_name
    else:
        if len(tgt_inputs) == 0:
            return {"status": "error", "error": f"Target node '{target_node_id}' has no input ports."}
        if len(tgt_inputs) == 1:
            tgt_port_key = next(iter(tgt_inputs))
        else:
            return {
                "status": "error",
                "error": f"Target node '{target_node_id}' has multiple input ports. "
                         f"Specify target_port_name. Available: {list(tgt_inputs.keys())}",
            }

    # 5. Resolve port types
    src_port_type = (src_outputs[src_port_key].get("type") or "any")
    tgt_port_type = (tgt_inputs[tgt_port_key].get("type") or "any")

    # 6. Build handle IDs (matching frontend generateHandleId)
    source_handle = f"{source_node_id}{HANDLE_SEPARATOR}{src_port_key}{HANDLE_SEPARATOR}output{HANDLE_SEPARATOR}{src_port_type}"
    target_handle = f"{target_node_id}{HANDLE_SEPARATOR}{tgt_port_key}{HANDLE_SEPARATOR}input{HANDLE_SEPARATOR}{tgt_port_type}"

    # 7. Generate edge ID (matching frontend generateEdgeId)
    edge_id = f"{source_node_id}{HANDLE_SEPARATOR}{source_handle}{HANDLE_SEPARATOR}to{HANDLE_SEPARATOR}{target_node_id}{HANDLE_SEPARATOR}{target_handle}"

    # 8. Create edge via API
    payload = {
        "id": edge_id,
        "source": source_node_id,
        "target": target_node_id,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
    }
    create_url = f"{DJANGO_API_URL}/workflow/{workflow_id}/edges/"
    data = await _make_post_request(create_url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to create edge from '{source_node_id}' to '{target_node_id}'"}
    return {"status": "success", "edge": data}


@mcp.tool()
async def delete_edge(workflow_id: str, edge_id: str) -> dict[str, Any]:
    """Delete an edge (connection) from a workflow.

    Permanently removes the edge. Does not affect the connected nodes.

    Args:
        workflow_id: UUID of the workflow.
        edge_id: String ID of the edge to delete.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/edges/{edge_id}/"
    data = await _make_delete_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to delete edge {edge_id} for {workflow_id}"}
    return {"status": "success", "result": data}


@mcp.tool()
async def update_node_parameter(workflow_id: str, node_id: str, parameter_key: str, parameter_value: Any, parameter_field: str = "default_value") -> dict[str, Any]:
    """Update a specific parameter value in a node's schema.

    Modifies node.data.schema.parameters[parameter_key][parameter_field] and records
    the modification in node.data.parameter_modifications for change tracking.
    The original value is preserved for comparison.

    Args:
        workflow_id: UUID of the workflow.
        node_id: String ID of the node.
        parameter_key: Key in schema.parameters to update (e.g. "learning_rate", "num_neurons").
        parameter_value: The new value to set (any JSON-serializable type).
        parameter_field: Which field of the parameter to update. Defaults to "default_value".
            Other options: "constraints", etc.

    IMPORTANT: Always use parameter_field="default_value" (the default) when reading or
    writing parameter values. The "value" field is legacy and may be stale — ignore it.
    The code generator and GUI both use "default_value" as the source of truth.

    Returns a dict with status, message, node_id, workflow_id, parameter_key,
    parameter_field, parameter_value, and updated_parameter (full parameter object after update).
    Returns error if parameter_key or schema is not found in the node.
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


@mcp.tool()
async def update_node_instance_name(
    workflow_id: str, node_id: str, instance_name: str
) -> dict[str, Any]:
    """Update the instance name (display label) of a node in the workflow.

    Sets node.data.instanceName, which is the user-facing name shown on the canvas
    and used as the variable name in generated Python code.

    Args:
        workflow_id: UUID of the workflow.
        node_id: String ID of the node.
        instance_name: New instance name for the node (e.g. "my_poisson_generator").

    Returns a dict with status, message, node_id, workflow_id, and updated_instance_name.
    """
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/{node_id}/instance_name/"
    payload = {"instance_name": instance_name}
    data = await _make_put_request(url, payload)
    if data is None:
        return {
            "status": "error",
            "error": f"Failed to update instance name for node {node_id}",
        }
    return {"status": "success", "result": data}


# Batch code generation
@mcp.tool()
async def generate_code_batch(workflow_id: str, nodes: list[Any], edges: list[Any]) -> dict[str, Any]:
    """Generate Python code and a Jupyter notebook from a workflow's React Flow graph.

    Converts the node graph into executable Python code. The generated files (.py and .ipynb)
    are saved on the server and can be executed via the workflow run endpoint.

    Args:
        workflow_id: UUID of the workflow.
        nodes: Array of React Flow node objects. Each must include id, position, type, and
            data (with label, schema, instanceName).
        edges: Array of React Flow edge objects. Each must include id, source, target,
            and optionally sourceHandle and targetHandle.

    Returns a dict with status, message, workflow_id, nodes_processed, edges_processed,
    code_status, and files (with python_file, notebook_file paths and existence flags).
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
    """Fetch a sample flow definition for testing and initialization.

    Returns a pre-defined sample flow with example nodes and edges
    that can be used to verify the backend is working or to seed a new project.
    Also useful as a health check since it confirms the backend API is reachable.
    """
    url = f"{DJANGO_API_URL}/workflow/sample-flow/"
    data = await _make_get_request(url, timeout=10.0)
    if data is None:
        return {"status": "error", "error": "Backend unreachable or sample flow not available"}
    return {"status": "success", "sample_flow": data}


@mcp.tool()
async def health() -> dict[str, Any]:
    """Check if the Django backend API is reachable.

    Pings the sample-flow endpoint and returns {"status": "ok"} if the backend
    responds successfully, or {"status": "error"} if it is unreachable.
    Use this before other operations to verify connectivity.
    """
    # Keep a convenience health tool that calls sample-flow
    resp = await get_sample_flow()
    if resp.get("status") != "success":
        return {"status": "error", "error": "Backend unreachable"}
    return {"status": "ok", "backend": "reachable"}


# File upload
@mcp.tool()
async def upload_python_file(
    code: str, name: str, description: str = "", category: str = "analysis"
) -> dict[str, Any]:
    """Upload a Python node file to the backend for analysis and registration.

    The backend parses the uploaded .py file to extract Node class definitions
    (inputs, outputs, parameters, methods) and stores the analyzed metadata.
    The file must contain valid Python code with classes extending the Node base class.
    Duplicate files (same content hash) are rejected.

    Args:
        code: Python source code content as a string.
        name: Filename with .py extension (e.g. "MyCustomNode.py").
        description: Optional human-readable description of the file/node.
        category: Node category folder. One of: "analysis", "io", "network",
            "optimization", "simulation", "stimulus". Defaults to "analysis".

    Returns the created file object with id (UUID), name, description, category,
    file (URL), file_size, is_analyzed, analysis_error, node_classes_count,
    uploaded_by, and timestamps.
    """
    url = f"{DJANGO_API_URL}/box/upload/"
    files = {"file": (name, code.encode("utf-8"), "text/x-python")}
    form_data = {"name": name, "description": description, "category": category}
    data = await _make_multipart_post_request(url, files=files, data=form_data)
    if data is None:
        return {"status": "error", "error": "Failed to upload python file"}
    return {"status": "success", "file": data}


# File list / detail
@mcp.tool()
async def list_python_files() -> dict[str, Any]:
    """List all uploaded Python node files.

    Returns an array of file objects, each containing id (UUID), name, description,
    category, file (URL), file_size, is_analyzed, analysis_error, node_classes_count,
    uploaded_by, uploaded_by_name, and timestamps.

    The backend supports query parameters for filtering (name, category, analyzed_only)
    but this tool fetches all files without filters.
    """
    url = f"{DJANGO_API_URL}/box/files/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to list python files"}
    return {"status": "success", "files": data}


@mcp.tool()
async def get_python_file(pk: str) -> dict[str, Any]:
    """Retrieve metadata and details for a single uploaded Python file.

    Returns the file object with id, name, description, category, file (URL),
    file_size, is_analyzed, analysis_error, node_classes_count, uploaded_by,
    uploaded_by_name, and timestamps.

    Args:
        pk: UUID primary key of the file.
    """
    url = f"{DJANGO_API_URL}/box/files/{pk}/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch python file {pk}"}
    return {"status": "success", "file": data}


# list of node definitions
@mcp.tool()
async def list_nodes_definitions() -> dict[str, Any]:
    """Retrieve all uploaded node class definitions formatted for the frontend.

    Scans all analyzed Python files and extracts their Node class definitions.
    Returns a dict with:
    - nodes: Array of node definitions, each containing id (format: "uploaded_{file_id}_{class_name}"),
      type ("uploadedNode"), label, description, category, file_id, class_name, file_name,
      and schema (with inputs, outputs, parameters, and methods describing port types,
      default values, constraints, widget types, and optimization settings).
    - total_files (int): Number of analyzed files.
    - total_nodes (int): Total number of node classes found.
    - categories (dict): Map of category names to their settings (color).
    """
    url = f"{DJANGO_API_URL}/box/uploaded-nodes/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to fetch nodes"}
    return {"status": "success", "nodes": data}


# File code management (get/put)
@mcp.tool()
async def get_python_file_code(filename: str) -> dict[str, Any]:
    """Retrieve the Python source code content of an uploaded file.

    Returns the raw source code along with file metadata.
    Response includes: status, code (str), filename, file_id (UUID),
    description, and uploaded_at timestamp.

    Args:
        filename: Filename with or without .py extension (e.g. "MyNode" or "MyNode.py").
    """
    url = f"{DJANGO_API_URL}/box/files/{filename}/code/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": f"Failed to fetch code for {filename}"}
    return {"status": "success", "code": data}


@mcp.tool()
async def update_python_file_code(filename: str, code_payload: dict) -> dict[str, Any]:
    """Update the Python source code of an uploaded file and re-analyze it.

    Replaces the file's source code and triggers automatic re-analysis to update
    the node class definitions (inputs, outputs, parameters, methods).

    Args:
        filename: Filename with or without .py extension (e.g. "MyNode" or "MyNode.py").
        code_payload: Dict with the new code. Required: {"code": str}.

    Returns a dict with status, message, filename, file_id, and code_length.
    """
    url = f"{DJANGO_API_URL}/box/files/{filename}/code/"
    data = await _make_put_request(url, code_payload)
    if data is None:
        return {"status": "error", "error": f"Failed to update code for {filename}"}
    return {"status": "success", "result": data}


# File copy
@mcp.tool()
async def copy_python_file(copy_payload: dict) -> dict[str, Any]:
    """Copy one or more uploaded Python files to create new files.

    Supports two copy modes:
    1. By file IDs: {"file_ids": ["uuid1", "uuid2"]} - creates copies with auto-generated names.
    2. By filename: {"source_filename": str, "target_filename": str} - copies and renames
       the file and its internal class names accordingly.

    When copying by filename, class names in the source code are automatically updated
    to match the new filename.

    Args:
        copy_payload: Either {"file_ids": [str]} or {"source_filename": str, "target_filename": str}.

    Returns a dict with copied_files (array of file objects), total_copied, total_requested,
    and errors (array of {file_id, error} for any failures).
    """
    url = f"{DJANGO_API_URL}/box/copy/"
    data = await _make_post_request(url, copy_payload)
    if data is None:
        return {"status": "error", "error": "Failed to copy python file"}
    return {"status": "success", "result": data}


# Parameter update
@mcp.tool()
async def update_python_file_parameter(param_payload: dict) -> dict[str, Any]:
    """Update a parameter value directly in an uploaded Python file's source code.

    Searches the file's source code for the parameter using three patterns:
    variable assignment, dictionary values, and function arguments, then replaces the value.
    The file is automatically re-analyzed after the update.

    Args:
        param_payload: Dict with required fields:
            - parameter_key (str): Parameter name to find in the source code.
            - parameter_value (any): New value to set (supports nested arrays/dicts).
            - parameter_field (str, optional): Field to update, defaults to "value".
            And one of:
            - file_id (UUID): Primary key of the file.
            - filename (str): Filename to look up.

    Returns a dict with status ("success" or "no_change"), message, filename,
    file_id, parameter_key, parameter_field, parameter_value, and is_analyzed.
    Returns "no_change" if the parameter was not found or already matches.
    """
    url = f"{DJANGO_API_URL}/box/parameters/update/"
    data = await _make_put_request(url, param_payload)
    if data is None:
        return {"status": "error", "error": "Failed to update python file parameter"}
    return {"status": "success", "result": data}


# Categories
@mcp.tool()
async def node_categories() -> dict[str, Any]:
    """Retrieve all available node categories and their display settings.

    Scans the media directory for category folders and returns their configuration.
    Returns a dict with:
    - categories: Array of {value: str, label: str, settings: {color: str (hex)}}.
      Built-in categories: analysis, io, network, optimization, simulation, stimulus.
    - default (str): The default category name ("analysis").
    """
    url = f"{DJANGO_API_URL}/box/categories/"
    data = await _make_get_request(url)
    if data is None:
        return {"status": "error", "error": "Failed to fetch node categories"}
    return {"status": "success", "categories": data}


# Bulk sync nodes
@mcp.tool()
async def bulk_sync_nodes(sync_payload: dict | None = None) -> dict[str, Any]:
    """Scan all category folders and bulk-import Python files into the database.

    Walks through each category directory, finds .py files not yet registered,
    and imports them. Duplicate files (by content hash or filename+category) are skipped.
    Each imported file is automatically analyzed to extract node class definitions.

    Args:
        sync_payload: Optional dict (currently unused; send {} or omit).

    Returns a dict with:
    - total_scanned (int): Total .py files found.
    - added (int): Number of newly imported files.
    - skipped (int): Number of already-registered files.
    - errors (int): Number of files that failed to import.
    - files: {added: [...], skipped: [...], errors: [...]}.
    - settings: Map of category names to their color settings.
    """
    url = f"{DJANGO_API_URL}/box/sync/"
    data = await _make_post_request(url, sync_payload or {})
    if data is None:
        return {"status": "error", "error": "Failed to bulk sync nodes"}
    return {"status": "success", "result": data}


@mcp.tool()
async def get_workflow_facts(workflow_id: str) -> dict[str, Any]:
    """Collect all facts about a workflow needed to write a scientific report.

    Fetches and consolidates:
    - Project metadata (name, description)
    - All nodes with their current parameter values (use default_value as the
      authoritative current value; ignore the legacy "value" field if present)
    - All edges (connections between nodes)
    - Simulation result files from the project's results/ folder, including
      array keys and shapes for .npz files

    Use this tool before writing a Methods section or simulation report.
    Pass the returned data to the language model with a prompt like:
    "Write a Methods section for a neuroscience paper based on these facts."

    Args:
        workflow_id: UUID of the workflow project.

    Returns a dict with: project, nodes (list with instance_name, class,
    category, parameters as {key: default_value}), edges, and results
    (list of {filename, size_bytes, arrays}).
    """
    project_data = await _make_get_request(f"{DJANGO_API_URL}/workflow/{workflow_id}/")
    nodes_data = await _make_get_request(f"{DJANGO_API_URL}/workflow/{workflow_id}/nodes/")
    edges_data = await _make_get_request(f"{DJANGO_API_URL}/workflow/{workflow_id}/edges/")
    results_data = await _make_get_request(f"{DJANGO_API_URL}/workflow/{workflow_id}/results/")
    code_data = await _make_get_request(f"{DJANGO_API_URL}/workflow/{workflow_id}/code/")

    if project_data is None:
        return {"status": "error", "error": f"Project {workflow_id} not found"}

    # Simplify node list: extract only instance name, class, category, and parameter default_values
    simplified_nodes = []
    for node in (nodes_data or []):
        data = node.get("data", {})
        schema = data.get("schema", {})
        params = schema.get("parameters", {})
        simplified_params = {k: v.get("default_value") for k, v in params.items() if isinstance(v, dict)}
        simplified_nodes.append({
            "instance_name": data.get("instanceName", node.get("id")),
            "class": data.get("label", ""),
            "category": data.get("nodeType", ""),
            "parameters": simplified_params,
        })

    # Simplify edges: instance names instead of raw IDs
    id_to_name = {n.get("id"): n.get("data", {}).get("instanceName", n.get("id")) for n in (nodes_data or [])}
    simplified_edges = []
    for edge in (edges_data or []):
        simplified_edges.append({
            "from": id_to_name.get(edge.get("source_node"), edge.get("source_node")),
            "to": id_to_name.get(edge.get("target_node"), edge.get("target_node")),
            "from_port": edge.get("source_handle", "").split("::")[1] if "::" in (edge.get("source_handle") or "") else edge.get("source_handle"),
            "to_port": edge.get("target_handle", "").split("::")[1] if "::" in (edge.get("target_handle") or "") else edge.get("target_handle"),
        })

    return {
        "status": "success",
        "project": {
            "name": project_data.get("name"),
            "description": project_data.get("description"),
        },
        "nodes": simplified_nodes,
        "edges": simplified_edges,
        "results": (results_data or {}).get("results", []),
        "generated_code": (code_data or {}).get("code"),
        "notebook_outputs": (code_data or {}).get("notebook_outputs", []),
    }


@mcp.tool()
async def save_report(workflow_id: str, report_text: str, filename: str = "report.md") -> dict[str, Any]:
    """Save a markdown report to the workflow project folder.

    Writes the report as a file inside codes/projects/{ProjectName}/{filename},
    making it accessible from JupyterLab alongside the workflow code and results.

    Call this after composing the report text with get_workflow_facts.

    Args:
        workflow_id: UUID of the workflow project.
        report_text: Full markdown text of the report to save.
        filename: Output filename. Defaults to "report.md".

    Returns a dict with status, message, path, and size_bytes.
    """
    # Use project name as filename if not overridden
    if filename == "report.md":
        project_data = await _make_get_request(f"{DJANGO_API_URL}/workflow/{workflow_id}/")
        if project_data:
            project_name = project_data.get("name", "report").replace(" ", "_")
            filename = f"{project_name}.md"

    watermark = "*Generated by NeuroWorkflow — Brain/MINDS 2.0 (draft)*\n\n---\n\n"
    url = f"{DJANGO_API_URL}/workflow/{workflow_id}/report/"
    payload = {"report_text": watermark + report_text, "filename": filename}
    data = await _make_post_request(url, payload)
    if data is None:
        return {"status": "error", "error": f"Failed to save report for workflow {workflow_id}"}
    return {"status": "success", "result": data}


if __name__ == "__main__":
    try:
        mcp.run(transport="http", host="0.0.0.0", port=MCP_PORT)
    except Exception:
        logger.exception("Failed to run FastMCP server")

