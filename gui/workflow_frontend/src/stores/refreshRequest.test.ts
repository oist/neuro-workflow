import { describe, expect, it } from "vitest";
import { refreshFailureMessage, refreshResultIsCurrent, saveIsKeptDuringRefresh } from "./refreshRequest";

describe("refreshResultIsCurrent", () => {
  it("applies when the timestamp and project are unchanged", () => {
    expect(refreshResultIsCurrent(10, 10, "project-a", "project-a")).toBe(true);
  });

  it("does not apply after a newer refresh or a project change", () => {
    expect(refreshResultIsCurrent(10, 11, "project-a", "project-a")).toBe(false);
    expect(refreshResultIsCurrent(10, 10, "project-a", "project-b")).toBe(false);
    expect(refreshResultIsCurrent(10, 10, "project-a", null)).toBe(false);
  });
});

describe("refreshFailureMessage", () => {
  it("uses the response detail or error, and otherwise the status", () => {
    expect(refreshFailureMessage(404, { detail: "Not found." })).toBe("Not found.");
    expect(refreshFailureMessage(400, { error: "Missing project" })).toBe("Missing project");
    expect(refreshFailureMessage(404, {})).toBe("HTTP 404");
    expect(refreshFailureMessage(500, null)).toBe("HTTP 500");
  });
});

describe("saveIsKeptDuringRefresh", () => {
  it("keeps a button-refresh save and drops a chat refresh or an in-flight load", () => {
    expect(saveIsKeptDuringRefresh(false, true, true)).toBe(true);
    expect(saveIsKeptDuringRefresh(false, true, false)).toBe(false);
    expect(saveIsKeptDuringRefresh(true, true, true)).toBe(false);
    expect(saveIsKeptDuringRefresh(false, false, false)).toBe(true);
  });
});
