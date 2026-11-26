import { describe, expect, it } from "vitest";
import { archiveFailureMessage } from "./archiveFailureMessage";

describe("archiveFailureMessage", () => {
  it("shows the server error or detail, otherwise the HTTP status", () => {
    expect(archiveFailureMessage({ error: "Project folder not found" }, 404)).toBe(
      "Project folder not found",
    );
    expect(archiveFailureMessage({ detail: "Project not found." }, 404)).toBe(
      "Project not found.",
    );
    expect(archiveFailureMessage({ error: "" }, 404)).toBe("HTTP 404");
    expect(archiveFailureMessage(null, 500)).toBe("HTTP 500");
  });
});
