import { createAuthHeaders } from "./authHeaders";
import { API_BASE_URL } from "../config/urls";

const API_PREFIX = API_BASE_URL;

export interface WorkflowRunSSEEvent {
  type: string;
  data: Record<string, unknown>;
}

/**
 * Start a workflow run via SSE streaming.
 *
 * The backend will execute the generated Python script on a Jupyter kernel
 * and stream stdout/stderr/errors back as Server-Sent Events.
 */
export const runWorkflowStream = async (
  workflowId: string,
  onEvent: (event: WorkflowRunSSEEvent) => void,
  signal?: AbortSignal
) => {
  const headers = await createAuthHeaders();

  const res = await fetch(`${API_PREFIX}/workflow/${workflowId}/run/`, {
    method: "POST",
    headers,
    signal,
  });

  if (!res.ok) {
    const body = await res.text();
    let errorMsg: string;
    try {
      const parsed = JSON.parse(body);
      errorMsg = parsed.error || body;
    } catch {
      errorMsg = body;
    }
    throw new Error(`Workflow run error ${res.status}: ${errorMsg}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  // Persists across read() chunks: a multi-MB `data:` line (e.g. a base64
  // image) can complete many chunks after its `event:` line arrived.
  let currentEventType = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        const dataStr = line.slice(6);
        try {
          const data = JSON.parse(dataStr);
          onEvent({ type: currentEventType, data });
        } catch (e) {
          console.warn("[workflowRunApi] Failed to parse SSE data:", dataStr, e);
        }
        currentEventType = "";
      }
    }
  }
};


export interface RunFigureManifestEntry {
  node_id: string | null;
  node_name: string | null;
  path: string;
  index: number;
}

export interface RunFigureManifest {
  version: number;
  finished_at: string;
  status: string;
  figures: RunFigureManifestEntry[];
}

/**
 * Fetch the figure manifest persisted by the last streamed run, or null if
 * the project has none. Served by the unauthenticated /api/viewer/ route
 * (same as the brain-viewer data), so no auth headers are needed and the
 * PNG paths it lists can be used directly as <img> sources.
 */
export const fetchRunFigureManifest = async (
  projectId: string
): Promise<RunFigureManifest | null> => {
  try {
    const res = await fetch(
      `/api/viewer/${projectId}/results/figures/manifest.json?t=${Date.now()}`
    );
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
};


// ---------------------------------------------------------------------------
// Async run management API (Phase 3)
// ---------------------------------------------------------------------------

export interface WorkflowRunRecord {
  id: string;
  workflow: string;
  user: string | null;
  backend: "local" | "slurm" | "jupyter";
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  slurm_job_id: string;
  exit_code: number | null;
  stdout: string;
  stderr: string;
  error_message: string;
  resource_requests: Record<string, unknown>;
  artifacts: Record<string, unknown>;
  submitted_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export const submitWorkflowRun = async (
  workflowId: string,
  backend: "local" | "slurm" | "jupyter" = "jupyter",
  resourceRequests: Record<string, unknown> = {}
): Promise<WorkflowRunRecord> => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/workflow/${workflowId}/runs/submit/`, {
    method: "POST",
    headers,
    body: JSON.stringify({ backend, resource_requests: resourceRequests }),
  });
  if (!res.ok) throw new Error(`Submit failed: ${res.status}`);
  return res.json();
};

export const getWorkflowRunStatus = async (
  workflowId: string,
  runId: string
): Promise<WorkflowRunRecord> => {
  const headers = await createAuthHeaders();
  const res = await fetch(
    `${API_PREFIX}/workflow/${workflowId}/runs/${runId}/`,
    { headers }
  );
  if (!res.ok) throw new Error(`Status fetch failed: ${res.status}`);
  return res.json();
};

export const listWorkflowRuns = async (
  workflowId: string
): Promise<WorkflowRunRecord[]> => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/workflow/${workflowId}/runs/`, {
    headers,
  });
  if (!res.ok) throw new Error(`List runs failed: ${res.status}`);
  return res.json();
};

export interface ArtifactFile {
  path: string;
  size: number;
}

/**
 * Download a single result artifact fetched back from a remote run.
 *
 * Auth is Bearer-token based, so a plain link can't be used — we fetch the
 * file with the auth headers, then trigger a browser download from the blob.
 */
export const downloadArtifact = async (
  workflowId: string,
  runId: string,
  path: string
): Promise<void> => {
  const headers = await createAuthHeaders();
  const res = await fetch(
    `${API_PREFIX}/workflow/${workflowId}/runs/${runId}/artifacts/?path=${encodeURIComponent(
      path
    )}`,
    { headers }
  );
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  // Prefix with the short run id so files from different runs (which share
  // names like spike_rasters.png) don't collide in the browser's downloads.
  const base = path.split("/").pop() || "artifact";
  a.download = `${runId.slice(0, 8)}_${base}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

export const deleteWorkflowRun = async (
  workflowId: string,
  runId: string
): Promise<void> => {
  const headers = await createAuthHeaders();
  const res = await fetch(
    `${API_PREFIX}/workflow/${workflowId}/runs/${runId}/`,
    { method: "DELETE", headers }
  );
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
};

export const cancelWorkflowRun = async (
  workflowId: string,
  runId: string
): Promise<WorkflowRunRecord> => {
  const headers = await createAuthHeaders();
  const res = await fetch(
    `${API_PREFIX}/workflow/${workflowId}/runs/${runId}/cancel/`,
    { method: "POST", headers }
  );
  if (!res.ok) throw new Error(`Cancel failed: ${res.status}`);
  return res.json();
};
