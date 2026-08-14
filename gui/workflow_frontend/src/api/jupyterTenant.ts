import { createAuthHeaders } from "./authHeaders";
import { JUPYTER_BASE_URL } from "../config/urls";

export type Tenant = "internal" | "hackathon";

export interface JupyterSession {
  tenant: Tenant;
  hub_user: string;
  jupyter_path: string;
  viewer_token: string;
  is_node_reviewer: boolean;
  notice: string;
}

let cached: JupyterSession | null = null;
let inflight: Promise<JupyterSession> | null = null;

export async function getJupyterSession(force = false): Promise<JupyterSession> {
  if (!force && cached) {
    return cached;
  }
  if (!force && inflight) {
    return inflight;
  }
  inflight = (async () => {
    const headers = await createAuthHeaders();
    const response = await fetch("/api/workflow/jupyter/session/", {
      credentials: "include",
      headers,
    });
    if (!response.ok) {
      throw new Error(`Failed to load Jupyter session (${response.status})`);
    }
    const data = (await response.json()) as JupyterSession;
    cached = data;
    return data;
  })();
  try {
    return await inflight;
  } finally {
    inflight = null;
  }
}

export function jupyterTreeUrl(
  treePath: string,
  session: JupyterSession,
): string {
  const trimmed = treePath.replace(/^\/+/, "");
  const base = `${JUPYTER_BASE_URL}/user/${session.hub_user}/lab/workspaces/auto-E/tree/${trimmed}`;
  if (!session.viewer_token) {
    return base;
  }
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}nw_viewer=${encodeURIComponent(session.viewer_token)}`;
}

export async function openJupyterTree(treePath: string): Promise<string> {
  const session = await getJupyterSession();
  return jupyterTreeUrl(treePath, session);
}
