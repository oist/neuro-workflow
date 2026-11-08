export type WorkspaceTone = "offline" | "view-only" | "auto-save" | "manual";

export function workspaceStatus(
  isConnected: boolean,
  canEdit: boolean,
  autoSaveEnabled: boolean,
): { tone: WorkspaceTone; badge: string; message: string } {
  if (!isConnected) {
    return {
      tone: "offline",
      badge: "Offline",
      message: "⚠️ Connection lost - changes not saved",
    };
  }
  if (!canEdit) {
    return {
      tone: "view-only",
      badge: "View only",
      message: "This project is view only.",
    };
  }
  if (autoSaveEnabled) {
    return {
      tone: "auto-save",
      badge: "Auto-save",
      message: "💾 Changes are automatically saved",
    };
  }
  return {
    tone: "manual",
    badge: "Manual save",
    message: "⚠️ Remember to save manually",
  };
}

/** A refused write is authorization, not a dropped connection. */
export function writeFailureMarksOffline(status: number): boolean {
  return status !== 403;
}
