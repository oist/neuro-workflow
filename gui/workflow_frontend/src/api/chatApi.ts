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
  // Null/omitted means the default: all tools, default prompt.
  profile_id?: string | null;
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
    throw new Error(`Chat stream error ${res.status}: ${body}`);
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
