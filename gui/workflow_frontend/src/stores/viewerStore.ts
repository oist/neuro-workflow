import { create } from "zustand";

// The state snapshot brain_viewer.js posts to the parent (buildStateSnapshot()).
// Loosely typed — we forward it to the chat as context, we don't inspect it.
export type ViewerSnapshot = Record<string, unknown> & {
  dataset?: { data_url?: string; species?: string; [k: string]: unknown };
};

export type ViewerStore = {
  // latest snapshot per viewer tab id
  snapshots: Record<string, ViewerSnapshot>;
  // the viewer tab the chat should read/drive (last-activated viewer)
  activeViewerTabId: string | null;

  setSnapshot: (tabId: string, snapshot: ViewerSnapshot) => void;
  clearSnapshot: (tabId: string) => void;
  setActiveViewerTabId: (tabId: string | null) => void;
  getActiveSnapshot: () => ViewerSnapshot | null;
};

export const useViewerStore = create<ViewerStore>((set, get) => ({
  snapshots: {},
  activeViewerTabId: null,

  setSnapshot: (tabId, snapshot) =>
    set((s) => ({ snapshots: { ...s.snapshots, [tabId]: snapshot } })),

  clearSnapshot: (tabId) =>
    set((s) => {
      const next = { ...s.snapshots };
      delete next[tabId];
      return { snapshots: next };
    }),

  setActiveViewerTabId: (tabId) => set({ activeViewerTabId: tabId }),

  getActiveSnapshot: () => {
    const { snapshots, activeViewerTabId } = get();
    if (activeViewerTabId && snapshots[activeViewerTabId]) {
      return snapshots[activeViewerTabId];
    }
    // Fall back to the sole snapshot if exactly one viewer is open.
    const ids = Object.keys(snapshots);
    return ids.length === 1 ? snapshots[ids[0]] : null;
  },
}));

// Derive the project-relative data_path from the snapshot's data_url
// (/api/viewer/<projectId>/<relative-path>) so chat tools read the same run.
export const deriveDataPath = (snapshot: ViewerSnapshot | null): string | null => {
  const url = snapshot?.dataset?.data_url;
  if (typeof url !== "string") return null;
  // Strip the ?_ts=... cache-buster (and any fragment) the viewer appends —
  // it is not part of the on-disk path the backend resolves.
  const clean = url.split("?")[0].split("#")[0];
  const m = clean.match(/^\/api\/viewer\/[^/]+\/(.+)$/);
  return m ? m[1] : null;
};

// Build the CURRENT VIEWER STATE context string sent with a chat message.
export const buildViewerContext = (snapshot: ViewerSnapshot | null): string | null => {
  if (!snapshot) return null;
  const dataPath = deriveDataPath(snapshot);
  const payload = dataPath ? { ...snapshot, data_path: dataPath } : snapshot;
  return JSON.stringify(payload);
};
