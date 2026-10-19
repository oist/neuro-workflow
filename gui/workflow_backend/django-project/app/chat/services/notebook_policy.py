"""Admin ceilings for the notebook agent's tool permissions (Issue #123).

The in-kernel agent asks the user before risky tool calls (allow / ask / deny
per tool category, see ``neuroworkflow.agent.permissions``). Staff can cap
those modes for everyone: a ceiling is the loosest mode a user may choose. The
kernel enforces every ceiling itself (it is advisory there, the kernel being
user-controlled); a ``deny`` ceiling on a workflow (MCP) tool is also enforced
here, in the notebook MCP proxies, which is a real boundary.

MCP tools carry no read-only annotation, so the read-only ones are listed here
and sent to the kernel with the policy; any other workflow tool, including a
new one, counts as an edit.
"""

MODES = ("allow", "ask", "deny")  # loosest to strictest

CATEGORIES = (
    "read",
    "workflow-read",
    "workflow-edit",
    "file-write",
    "code-exec",
    "web",
    "other",
)

WORKFLOW_READ_TOOLS = (
    "list_projects",
    "get_project",
    "get_flow",
    "get_sample_flow",
    "list_nodes",
    "get_node",
    "list_edges",
    "health",
    "node_categories",
    "list_python_files",
    "get_python_file",
    "list_nodes_definitions",
    "get_python_file_code",
    "get_workflow_facts",
    "catalog_statistics",
    "catalog_search",
    "catalog_lookup",
    "catalog_datasets",
    "viewer_search_regions",
    "viewer_get_region",
    "viewer_list_groups",
    "viewer_get_connections",
    "viewer_node_strength",
    "viewer_list_signals",
    "viewer_get_activity",
    "viewer_compute_metrics",
    "viewer_explain_activity",
    "viewer_functional_connectivity",
)

MAX_TOOL_NAME = 255


def validate_ceilings(category_ceilings, tool_ceilings) -> dict:
    """Return ``{field: [error, ...]}`` for invalid ceilings (empty when valid)."""
    errors = {}
    if not isinstance(category_ceilings, dict):
        errors["category_ceilings"] = ["Must be an object."]
    else:
        bad = [
            f"{key!r}: unknown category"
            for key in category_ceilings
            if key not in CATEGORIES
        ] + [
            f"{key!r}: mode must be one of {', '.join(MODES)}"
            for key, mode in category_ceilings.items()
            if mode not in MODES
        ]
        if bad:
            errors["category_ceilings"] = bad
    if not isinstance(tool_ceilings, dict):
        errors["tool_ceilings"] = ["Must be an object."]
    else:
        bad = [
            f"{key!r}: tool names are 1-{MAX_TOOL_NAME} characters"
            for key in tool_ceilings
            if not key or len(key) > MAX_TOOL_NAME
        ] + [
            f"{key!r}: mode must be one of {', '.join(MODES)}"
            for key, mode in tool_ceilings.items()
            if mode not in MODES
        ]
        if bad:
            errors["tool_ceilings"] = bad
    return errors


def mcp_tool_category(name: str) -> str:
    return "workflow-read" if name in WORKFLOW_READ_TOOLS else "workflow-edit"


def is_mcp_denied(policy, name: str) -> bool:
    """True when the admin ceiling for workflow tool ``name`` is deny.

    A per-tool ceiling wins over the category one, in both directions.
    """
    if name in policy.tool_ceilings:
        return policy.tool_ceilings[name] == "deny"
    return policy.category_ceilings.get(mcp_tool_category(name)) == "deny"
