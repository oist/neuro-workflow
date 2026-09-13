import { createAuthHeaders } from "./authHeaders";

// Use relative path so Vite proxy handles routing to the backend
const API_PREFIX = "/api";

// REST endpoints

export interface ConversationCreatePayload {
  title?: string;
  project?: string | null;
  system_prompt?: string;
}

export const listConversations = async () => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/conversations/`, { headers });
  if (!res.ok) throw new Error(`Failed to list conversations: ${res.status}`);
  return res.json();
};

export const getConversation = async (id: string) => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/conversations/${id}/`, {
    headers,
  });
  if (!res.ok) throw new Error(`Failed to get conversation: ${res.status}`);
  return res.json();
};

export const createConversation = async (
  payload: ConversationCreatePayload
) => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/conversations/`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to create conversation: ${res.status}`);
  return res.json();
};

export const deleteConversation = async (id: string) => {
  const headers = await createAuthHeaders();
  const res = await fetch(`${API_PREFIX}/chat/conversations/${id}/`, {
    method: "DELETE",
    headers,
  });
  if (!res.ok) throw new Error(`Failed to delete conversation: ${res.status}`);
  return res.json();
};

// SSE streaming

export interface SendMessagePayload {
  message: string;
  conversation_id?: string | null;
  project_id?: string | null;
  // Snapshot of the brain viewer the user is currently looking at (selection,
  // time window, toggles, data_path). Injected server-side as chat context.
  viewer_context?: string | null;
  // Selected chat profile (MCP tool allowlist + system prompt override).
  // An explicit id always wins. Null/omitted: staff get all tools + the
  // default prompt; non-staff get the admin default profile when one is set,
  // otherwise all tools.
  profile_id?: string | null;
}

// Thrown by sendMessageStream on a non-2xx response; status/body let callers
// react to specific failures (e.g. 404 "Chat profile not found").
export interface ChatStreamError extends Error {
  status?: number;
  body?: string;
}

export interface SSEEvent {
  type: string;
  data: Record<string, unknown>;
}

export const sendMessageStream = async (
  payload: SendMessagePayload,
  onEvent: (event: SSEEvent) => void,
  onConversationId: (id: string) => void,
  signal?: AbortSignal
) => {
  const headers = await createAuthHeaders();

  const res = await fetch(`${API_PREFIX}/chat/stream/`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    const body = await res.text();
    const error: ChatStreamError = new Error(
      `Chat stream error ${res.status}: ${body}`
    );
    error.status = res.status;
    error.body = body;
    throw error;
  }

  // Read the conversation ID from the response header
  const convId = res.headers.get("X-Conversation-Id");
  if (convId) {
    onConversationId(convId);
  }

  // Parse SSE from the ReadableStream (POST-based SSE)
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEventType = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEventType = line.slice(7).trim();

        // Also check for conversation_id event in the stream
        if (currentEventType === "conversation_id") {
          // The next data line will have the conversation id
        }
      } else if (line.startsWith("data: ")) {
        const dataStr = line.slice(6);
        try {
          const data = JSON.parse(dataStr);

          if (currentEventType === "conversation_id") {
            if (data.id) onConversationId(data.id);
          } else {
            onEvent({ type: currentEventType, data });
          }
        } catch {
          // Ignore JSON parse errors for partial data
        }
        currentEventType = "";
      }
    }
  }
};
