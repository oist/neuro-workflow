import { create } from "zustand";

// A figure emitted by a node during a workflow run. `src` is a data: URI
// while streaming, or an /api/viewer/ URL when restored from disk.
export interface NodeFigure {
  src: string;
  mime: string;
  index: number;
}

// Memory cap per node; oldest figures are dropped first.
const MAX_FIGURES_PER_NODE = 20;

export type RunStore = {
  // Figures keyed by React Flow node id. Deliberately kept out of
  // flowStore: run output must not enter undo history or the debounced
  // node-persistence PUT (base64 in the DB).
  figuresByNode: Record<string, NodeFigure[]>;
  // Figures the backend could not attribute to a node (node_id null).
  unattributed: NodeFigure[];
  executingNodeId: string | null;

  addFigure: (nodeId: string | null, fig: Omit<NodeFigure, "index">) => void;
  setExecutingNodeId: (nodeId: string | null) => void;
  clearRunFigures: () => void;
  setAllFigures: (byNode: Record<string, NodeFigure[]>) => void;
};

export const useRunStore = create<RunStore>((set) => ({
  figuresByNode: {},
  unattributed: [],
  executingNodeId: null,

  addFigure: (nodeId, fig) =>
    set((s) => {
      if (nodeId === null) {
        const next = [...s.unattributed, { ...fig, index: s.unattributed.length }];
        return { unattributed: next.slice(-MAX_FIGURES_PER_NODE) };
      }
      const existing = s.figuresByNode[nodeId] ?? [];
      const next = [...existing, { ...fig, index: existing.length }];
      return {
        figuresByNode: {
          ...s.figuresByNode,
          [nodeId]: next.slice(-MAX_FIGURES_PER_NODE),
        },
      };
    }),

  setExecutingNodeId: (nodeId) => set({ executingNodeId: nodeId }),

  clearRunFigures: () =>
    set({ figuresByNode: {}, unattributed: [], executingNodeId: null }),

  setAllFigures: (byNode) =>
    set({
      // Same cap as addFigure — keep the newest figures per node.
      figuresByNode: Object.fromEntries(
        Object.entries(byNode).map(([nodeId, figs]) => [
          nodeId,
          figs.slice(-MAX_FIGURES_PER_NODE),
        ])
      ),
      unattributed: [],
    }),
}));
