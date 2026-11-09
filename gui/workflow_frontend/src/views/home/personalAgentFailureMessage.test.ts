import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { personalAgentFailureMessage } from "./personalAgentFailureMessage";

const modal = readFileSync(
  resolve(dirname(fileURLToPath(import.meta.url)), "components/PersonalAgentModal.tsx"),
  "utf8",
);

describe("personalAgentFailureMessage", () => {
  it("shows the server error or detail, otherwise the HTTP status", () => {
    expect(personalAgentFailureMessage({ error: "Invalid service token" }, 401)).toBe(
      "Invalid service token",
    );
    expect(personalAgentFailureMessage({ detail: "Authentication credentials were not provided." }, 401)).toBe(
      "Authentication credentials were not provided.",
    );
    expect(personalAgentFailureMessage({ error: "" }, 404)).toBe("HTTP 404");
    expect(personalAgentFailureMessage(null, 500)).toBe("HTTP 500");
  });
});

describe("PersonalAgentModal", () => {
  it("clears the saved status before the next read", () => {
    const effect = modal.slice(modal.indexOf("useEffect"), modal.indexOf("const save"));
    const clearAt = effect.indexOf('setStatus("")');
    expect(clearAt).toBeGreaterThanOrEqual(0);
    expect(clearAt).toBeLessThan(effect.indexOf("fetch("));
  });

  it("ignores a second save or clear while one is running", () => {
    const save = modal.slice(modal.indexOf("const save"), modal.indexOf("const clear"));
    const clear = modal.slice(modal.indexOf("const clear"), modal.lastIndexOf("return ("));
    expect(save).toContain("if (inFlight.current)");
    expect(clear).toContain("if (inFlight.current)");
    expect(modal).toContain("personalAgentFailureMessage");
  });
});
