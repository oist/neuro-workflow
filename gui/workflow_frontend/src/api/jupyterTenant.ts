import { createAuthHeaders } from "./authHeaders";
import { JUPYTER_BASE_URL } from "../config/urls";

export type Tenant = "project" | "community";

export function normalizeTenant(value?: string | null): Tenant {
  const key = (value || "").trim().toLowerCase();
  if (key === "community" || key === "hackathon") {
    return "community";
  }
  if (key === "internal") {
    return "project";
  }
  return "project";
}

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
    data.tenant = normalizeTenant(data.tenant);
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
  const base = `${JUPYTER_BASE_URL}/user/${session.hub_user}/lab/tree/${trimmed}`;
  if (!session.viewer_token) {
    return base;
  }
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}nw_viewer=${encodeURIComponent(session.viewer_token)}`;
}

// The project Lab mounts the community folder read-only under codes-community/.
export function projectTreePath(
  treePath: string,
  projectTenant: string | null | undefined,
  session: JupyterSession,
): string {
  const trimmed = treePath.replace(/^\/+/, "");
  if (
    session.tenant === "project" &&
    projectTenant &&
    normalizeTenant(projectTenant) === "community" &&
    trimmed.startsWith("codes/")
  ) {
    return `codes-community/${trimmed.slice("codes/".length)}`;
  }
  return trimmed;
}

export async function openJupyterTree(
  treePath: string,
  projectTenant?: string | null,
): Promise<string> {
  const session = await getJupyterSession();
  return jupyterTreeUrl(projectTreePath(treePath, projectTenant, session), session);
}
