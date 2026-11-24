export type WorkspaceTone = "offline" | "view-only" | "auto-save" | "manual";

export const VIEW_ONLY_TITLE = "View only";
export const VIEW_ONLY_MESSAGE = "This project is view only.";
export const REFUSED_SAVE_FALLBACK = "This change was not saved.";

export function canvasAcceptsDrop(canEdit: boolean): boolean {
  return canEdit;
}

export function nodeUpdateForViewer<T extends { isParamExpand?: boolean }>(
  canEdit: boolean,
  data: T,
): T | { isParamExpand: boolean } | null {
  if (canEdit) return data;
  if (typeof data.isParamExpand === "boolean") {
    return { isParamExpand: data.isParamExpand };
  }
  return null;
}

export function refusedSaveNotice(
  canEdit: boolean,
  detail: string | null,
): { title: string; description: string } {
  if (!canEdit) {
    return { title: VIEW_ONLY_TITLE, description: VIEW_ONLY_MESSAGE };
  }
  const text = detail?.trim() ? detail.trim() : REFUSED_SAVE_FALLBACK;
  return { title: "Not saved", description: text };
}

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
      badge: VIEW_ONLY_TITLE,
      message: VIEW_ONLY_MESSAGE,
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
