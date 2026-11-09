"""Shape the facts a methods report is allowed to use.

The chat server is not imported here, so the shaping can be tested on its own.
"""

from typing import Any


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_workflow_facts(
    project_data: dict,
    files_data: dict | None,
    nodes_data: list | None,
    edges_data: list | None,
    results_data: dict | None,
    code_data: dict | None,
) -> dict[str, Any]:
    """Return the report facts. Empty credits stay empty so they are not invented."""
    simplified_nodes = []
    for node in nodes_data or []:
        data = node.get("data", {})
        schema = data.get("schema", {})
        params = schema.get("parameters", {})
        simplified_params = {
            key: value.get("default_value")
            for key, value in params.items()
            if isinstance(value, dict)
        }
        simplified_nodes.append(
            {
                "instance_name": data.get("instanceName", node.get("id")),
                "class": data.get("label", ""),
                "category": data.get("nodeType", ""),
                "parameters": simplified_params,
            }
        )

    id_to_name = {
        node.get("id"): node.get("data", {}).get("instanceName", node.get("id"))
        for node in (nodes_data or [])
    }
    simplified_edges = []
    for edge in edges_data or []:
        source_handle = edge.get("source_handle") or ""
        target_handle = edge.get("target_handle") or ""
        simplified_edges.append(
            {
                "from": id_to_name.get(edge.get("source_node"), edge.get("source_node")),
                "to": id_to_name.get(edge.get("target_node"), edge.get("target_node")),
                "from_port": source_handle.split("::")[1] if "::" in source_handle else source_handle,
                "to_port": target_handle.split("::")[1] if "::" in target_handle else target_handle,
            }
        )

    return {
        "status": "success",
        "project": {
            "name": project_data.get("name"),
            "description": project_data.get("description"),
            "doi": _text(project_data.get("doi")),
            "data_source": _text(project_data.get("data_source")),
            "license": _text(project_data.get("license")),
            "funding": _text(project_data.get("funding")),
            "reference": _text(project_data.get("reference")),
            "contributors": project_data.get("contributors") or [],
            "links": project_data.get("links") or [],
        },
        "inputs": (files_data or {}).get("files") or [],
        "nodes": simplified_nodes,
        "edges": simplified_edges,
        "results": (results_data or {}).get("results") or [],
        "generated_code": (code_data or {}).get("code"),
        "notebook_outputs": (code_data or {}).get("notebook_outputs") or [],
    }
