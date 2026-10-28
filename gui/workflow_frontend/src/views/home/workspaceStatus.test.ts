import { describe, expect, it } from "vitest";
import { workspaceStatus, writeFailureMarksOffline } from "./workspaceStatus";

describe("workspaceStatus", () => {
  it("reserves the connection-lost line for a real disconnect", () => {
    const status = workspaceStatus(false, false, true);
    expect(status.badge).toBe("Offline");
    expect(status.message).toBe("⚠️ Connection lost - changes not saved");
  });

  it("shows view only while the API is reachable and the project cannot be edited", () => {
    const status = workspaceStatus(true, false, true);
    expect(status.badge).toBe("View only");
    expect(status.message).toBe("This project is view only.");
    expect(status.message).not.toContain("Connection lost");
  });

  it("keeps auto-save for an editable project", () => {
    expect(workspaceStatus(true, true, true).badge).toBe("Auto-save");
    expect(workspaceStatus(true, true, false).badge).toBe("Manual save");
  });
});

describe("writeFailureMarksOffline", () => {
  it("does not treat 403 as a lost connection", () => {
    expect(writeFailureMarksOffline(403)).toBe(false);
    expect(writeFailureMarksOffline(500)).toBe(true);
    expect(writeFailureMarksOffline(0)).toBe(true);
  });
});
