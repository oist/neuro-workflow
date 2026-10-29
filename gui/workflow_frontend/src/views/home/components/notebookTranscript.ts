import type {
  NotebookConversation,
  NotebookMessage,
} from "@/api/chatApi";
import type { ActiveToolCall } from "@/stores/chatStore";

export type TranscriptItem =
  | { type: "message"; data: NotebookMessage }
  | { type: "tool_call"; data: ActiveToolCall };

// Order a saved conversation for display: each tool call (with its result,
// matched by tool_call_id) follows the assistant text that made it. Tool
// messages are never shown on their own, and a call without a result is
// shown as failed (the turn was interrupted).
export function buildTranscriptItems(
  messages: NotebookMessage[]
): TranscriptItem[] {
  const results = new Map(
    messages
      .filter((m) => m.role === "tool")
      .map((m) => [m.tool_call_id ?? "", m])
  );
  const items: TranscriptItem[] = [];
  for (const message of messages) {
    if (message.role === "tool" || message.role === "system") continue;
    if (message.content) items.push({ type: "message", data: message });
    for (const call of message.tool_calls ?? []) {
      const result = results.get(call.id);
      items.push({
        type: "tool_call",
        data: {
          tool_call_id: call.id,
          tool_name: call.function.name,
          arguments: call.function.arguments,
          result: result?.content,
          status: result && !result.is_error ? "done" : "error",
        },
      });
    }
  }
  return items;
}

// A backtick fence longer than any backtick run inside the text.
function fence(text: string, lang = ""): string {
  const longest = Math.max(
    2,
    ...(text.match(/`+/g) ?? []).map((run) => run.length)
  );
  const ticks = "`".repeat(longest + 1);
  return `${ticks}${lang}\n${text}\n${ticks}`;
}

function formatArguments(args: string): string {
  try {
    return JSON.stringify(JSON.parse(args), null, 2);
  } catch {
    return args;
  }
}

export function transcriptToMarkdown(conversation: NotebookConversation): string {
  const lines = [`# ${conversation.title}`, ""];
  if (conversation.user_label) lines.push(`- User: ${conversation.user_label}`);
  if (conversation.model) lines.push(`- Model: ${conversation.model}`);
  lines.push(
    `- Started: ${conversation.created_at}`,
    `- Updated: ${conversation.updated_at}`,
    ""
  );
  for (const item of buildTranscriptItems(conversation.messages)) {
    if (item.type === "message") {
      const who = item.data.role === "user" ? "User" : "Assistant";
      lines.push(`## ${who}`, "", item.data.content, "");
    } else {
      const call = item.data;
      const failed = call.status === "error" ? " (error)" : "";
      lines.push(
        "<details>",
        `<summary>Tool: ${call.tool_name}${failed}</summary>`,
        "",
        fence(formatArguments(call.arguments), "json"),
        ""
      );
      if (call.result !== undefined) lines.push(fence(call.result), "");
      lines.push("</details>", "");
    }
  }
  return lines.join("\n");
}

export function transcriptToJSON(conversation: NotebookConversation): string {
  return JSON.stringify(conversation, null, 2);
}

export function transcriptFilename(
  conversation: NotebookConversation,
  extension: "md" | "json"
): string {
  const day = conversation.created_at.slice(0, 10);
  return `notebook-chat-${day}-${conversation.id.slice(0, 8)}.${extension}`;
}

export function downloadText(filename: string, text: string, mime: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
