import { describe, expect, it } from "vitest";
import type { NotebookConversation, NotebookMessage } from "@/api/chatApi";
import {
  buildTranscriptItems,
  transcriptFilename,
  transcriptToMarkdown,
} from "./notebookTranscript";

const message = (m: Partial<NotebookMessage>): NotebookMessage => ({
  id: Math.random().toString(),
  role: "assistant",
  content: "",
  is_error: false,
  ...m,
});

const call = (id: string, name: string, args: string) => ({
  id,
  type: "function" as const,
  function: { name, arguments: args },
});

const messages: NotebookMessage[] = [
  message({ role: "user", content: "Plot it" }),
  message({
    content: "Running.",
    tool_calls: [
      call("t1", "run_code", '{"code": "plot()"}'),
      call("t2", "get_flow", "{}"),
    ],
  }),
  message({ role: "tool", tool_call_id: "t1", content: "```done```" }),
  // t2 has no result: the turn was interrupted.
  message({ tool_calls: [call("t3", "run_code", "{}")] }),
  message({ role: "tool", tool_call_id: "t3", content: "boom", is_error: true }),
  message({ content: "It failed." }),
];

const conversation: NotebookConversation = {
  id: "0a1b2c3d-4e5f-6789-abcd-ef0123456789",
  title: "Plot it",
  project: "p1",
  user_label: "Alice",
  model: "claude-test",
  message_count: messages.length,
  created_at: "2026-10-20T01:02:03Z",
  updated_at: "2026-10-20T01:05:00Z",
  can_delete: true,
  messages,
};

describe("buildTranscriptItems", () => {
  it("pairs tool calls with their results and hides tool messages", () => {
    const items = buildTranscriptItems(messages);

    expect(
      items.map((i) => (i.type === "message" ? i.data.content : i.data.tool_call_id))
    ).toEqual(["Plot it", "Running.", "t1", "t2", "t3", "It failed."]);
    const calls = items.flatMap((i) => (i.type === "tool_call" ? [i.data] : []));
    expect(calls.map((c) => [c.status, c.result])).toEqual([
      ["done", "```done```"],
      ["error", undefined],
      ["error", "boom"],
    ]);
  });
});

describe("transcriptToMarkdown", () => {
  it("writes the header, turns and tool calls", () => {
    const md = transcriptToMarkdown(conversation);

    expect(md).toContain("# Plot it\n\n- User: Alice\n- Model: claude-test");
    expect(md).toContain("## User\n\nPlot it");
    expect(md).toContain("## Assistant\n\nRunning.");
    expect(md).toContain("<summary>Tool: run_code</summary>");
    expect(md).toContain('```json\n{\n  "code": "plot()"\n}\n```');
    // A result containing backticks gets a longer fence.
    expect(md).toContain("````\n```done```\n````");
    expect(md).toContain("<summary>Tool: get_flow (error)</summary>");
  });
});

describe("transcriptFilename", () => {
  it("uses the start date and a short id", () => {
    expect(transcriptFilename(conversation, "md")).toBe(
      "notebook-chat-2026-10-20-0a1b2c3d.md"
    );
  });
});
