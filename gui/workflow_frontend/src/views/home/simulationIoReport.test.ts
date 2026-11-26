import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../../../..");

function creditSentence(source: string): string {
  const match = source.match(/When a file note[\s\S]*?do not invent one\./);
  expect(match).not.toBeNull();
  return match![0].replace(/\s+/g, " ");
}

describe("simulation file credits in the report", () => {
  it("names simulation_io in the methods rules and the report button", () => {
    const rules = readFileSync(
      resolve(
        repoRoot,
        "gui/workflow_backend/django-project/app/chat/services/chat_orchestrator.py",
      ),
      "utf8",
    );
    const button = readFileSync(
      resolve(repoRoot, "gui/workflow_frontend/src/views/home/components/chatbotView.tsx"),
      "utf8",
    );
    expect(creditSentence(rules)).toContain("simulation_io");
    expect(creditSentence(button)).toContain("simulation_io");
  });
});
