import { API_BASE_URL } from "../config/urls";

const API_PREFIX = API_BASE_URL;

/**
 * Relay the browser's Keycloak access token to the backend for `projectId`.
 *
 * The in-notebook chat agent of that project never receives the token: the
 * backend keeps it and uses it server-side when the kernel calls the workflow
 * MCP proxies. Called while the project's Jupyter tab is open.
 */
export const postNotebookToken = async (
  projectId: string,
  token: string
): Promise<void> => {
  const res = await fetch(`${API_PREFIX}/chat/notebook-token/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ project_id: projectId }),
  });
  if (!res.ok) {
    throw new Error(`Failed to relay notebook token: ${res.status}`);
  }
};
