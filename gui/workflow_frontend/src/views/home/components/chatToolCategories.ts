import type { ChatTool } from "@/api/chatProfileApi";

// Display grouping for the MCP tool picker in the chat profile editor. The
// MCP tool list carries no tags, so the mapping is static; tools not listed
// here (e.g. newly added ones) are shown under "Other".
export interface ToolCategory {
  id: string;
  label: string;
  tools: string[];
}

export const TOOL_CATEGORIES: ToolCategory[] = [
  {
    id: "projects",
    label: "Projects & Flow",
    tools: [
      "list_projects", "create_project", "get_project", "update_project",
      "get_flow", "update_flow", "get_sample_flow",
    ],
  },
  {
    id: "nodes",
    label: "Nodes & Edges",
    tools: [
      "list_nodes", "add_node", "get_node", "update_node", "delete_node",
      "update_node_parameter", "update_node_instance_name",
      "list_edges", "add_edge", "delete_edge",
    ],
  },
  {
    id: "code",
    label: "Code & System",
    tools: ["generate_code_batch", "node_categories", "bulk_sync_nodes", "health"],
  },
  {
    id: "python",
    label: "Python Node Files",
    tools: [
      "upload_python_file", "list_python_files", "get_python_file",
      "list_nodes_definitions", "get_python_file_code", "update_python_file_code",
      "copy_python_file", "update_python_file_parameter",
    ],
  },
  {
    id: "reports",
    label: "Reports",
    tools: ["get_workflow_facts", "save_report"],
  },
  {
    id: "viewer_read",
    label: "Brain Viewer (read)",
    tools: [
      "viewer_search_regions", "viewer_get_region", "viewer_list_groups",
      "viewer_get_connections", "viewer_node_strength", "viewer_list_signals",
      "viewer_get_activity", "viewer_compute_metrics", "viewer_explain_activity",
      "viewer_functional_connectivity",
    ],
  },
  {
    id: "viewer_control",
    label: "Brain Viewer (control)",
    tools: [
      "viewer_highlight_region", "viewer_focus_region", "viewer_set_time_window",
      "viewer_show_trace", "viewer_clear_selection",
    ],
  },
];

export interface ToolGroup {
  id: string;
  label: string;
  tools: ChatTool[];
}

// Group the live tool catalog by category, dropping empty categories and
// collecting unknown tools under "Other".
export const groupToolsByCategory = (tools: ChatTool[]): ToolGroup[] => {
  const byName = new Map(tools.map((t) => [t.name, t]));
  const seen = new Set<string>();
  const groups: ToolGroup[] = [];

  for (const category of TOOL_CATEGORIES) {
    const present: ChatTool[] = [];
    for (const name of category.tools) {
      const tool = byName.get(name);
      if (tool) {
        present.push(tool);
        seen.add(name);
      }
    }
    if (present.length > 0) {
      groups.push({ id: category.id, label: category.label, tools: present });
    }
  }

  const other = tools.filter((t) => !seen.has(t.name));
  if (other.length > 0) {
    groups.push({ id: "other", label: "Other", tools: other });
  }
  return groups;
};
