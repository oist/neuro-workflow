import { describe, expect, it } from "vitest";
import {
  canvasAcceptsDrop,
  nodeUpdateForViewer,
  refusedSaveNotice,
  workspaceStatus,
  writeFailureMarksOffline,
} from "./workspaceStatus";

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

describe("canvasAcceptsDrop", () => {
  it("refuses a drop on a view-only project and accepts one that can be edited", () => {
    expect(canvasAcceptsDrop(false)).toBe(false);
    expect(canvasAcceptsDrop(true)).toBe(true);
  });
});

describe("nodeUpdateForViewer", () => {
  it("drops a view-only schema change and keeps folding the node", () => {
    expect(nodeUpdateForViewer(false, { schema: { parameters: {} }, instanceName: "Rate" })).toBeNull();
    expect(nodeUpdateForViewer(false, { isParamExpand: false })).toEqual({ isParamExpand: false });
    expect(nodeUpdateForViewer(true, { schema: { parameters: {} }, instanceName: "Rate" })).toEqual({
      schema: { parameters: {} },
      instanceName: "Rate",
    });
  });
});

describe("refusedSaveNotice", () => {
  it("says view only only when the project cannot be edited", () => {
    expect(refusedSaveNotice(false, "You do not have permission to perform this action.")).toEqual({
      title: "View only",
      description: "This project is view only.",
    });
    const editable = refusedSaveNotice(true, "You do not have permission to perform this action.");
    expect(editable).toEqual({
      title: "Not saved",
      description: "You do not have permission to perform this action.",
    });
    expect(editable.title).not.toContain("View only");
    expect(editable.description).not.toContain("View only");
    expect(refusedSaveNotice(true, null).description).toBe("This change was not saved.");
  });
});

describe("writeFailureMarksOffline", () => {
  it("does not treat 403 as a lost connection", () => {
    expect(writeFailureMarksOffline(403)).toBe(false);
    expect(writeFailureMarksOffline(500)).toBe(true);
    expect(writeFailureMarksOffline(0)).toBe(true);
  });
});
