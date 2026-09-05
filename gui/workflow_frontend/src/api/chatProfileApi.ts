import { createAuthHeaders } from "./authHeaders";

// Use relative path so Vite proxy handles routing to the backend
const API_PREFIX = "/api";

// A per-user preset for the browser chat: which MCP tools the assistant may
// use, plus an optional system prompt override. Empty allowed_tools disables
// tools entirely; "no profile selected" means all tools + default prompt.
export interface ChatProfile {
  id: string;
  name: string;
  allowed_tools: string[];
  system_prompt: string;
  created_at: string;
  updated_at: string;
}

export interface ChatProfilePayload {
  name: string;
  allowed_tools: string[];
  system_prompt: string;
}

export interface ChatTool {
  name: string;
  description: string;
}

// Turn a DRF error body ({field: ["msg"]} or {error: "msg"}) into an Error.
const readError = async (res: Response, fallback: string): Promise<Error> => {
  const body = await res.json().catch(() => null);
  if (body && typeof body === "object") {
    const parts = Object.entries(body as Record<string, unknown>).map(
      ([key, value]) => {
        const text = Array.isArray(value) ? value.join(" ") : String(value);
        return key === "error" ? text : `${key}: ${text}`;
      }
    );
    if (parts.length > 0) return new Error(parts.join("; "));
  }
  return new Error(`${fallback}: ${res.status}`);
};

export const listChatProfiles = async (): Promise<ChatProfile[]> => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/profiles/`, { headers });
  if (!res.ok) throw await readError(res, "Failed to list chat profiles");
  return res.json();
};

export const createChatProfile = async (
  payload: ChatProfilePayload
): Promise<ChatProfile> => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/profiles/`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await readError(res, "Failed to create chat profile");
  return res.json();
};

export const updateChatProfile = async (
  id: string,
  payload: Partial<ChatProfilePayload>
): Promise<ChatProfile> => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/profiles/${id}/`, {
    method: "PUT",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await readError(res, "Failed to update chat profile");
  return res.json();
};

export const deleteChatProfile = async (id: string): Promise<void> => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/profiles/${id}/`, {
    method: "DELETE",
    headers,
  });
  if (!res.ok) throw await readError(res, "Failed to delete chat profile");
};

// The MCP tool catalog, via the existing OpenAI-function-shaped endpoint.
export const listChatTools = async (): Promise<ChatTool[]> => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/mcp-tools/`, { headers });
  if (!res.ok) throw await readError(res, "Failed to load MCP tools");
  const data = await res.json();
  const tools: { function: { name: string; description?: string } }[] =
    data.tools ?? [];
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description ?? "",
  }));
};
