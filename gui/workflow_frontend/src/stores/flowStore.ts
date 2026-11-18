import { create } from "zustand";
import { temporal } from "zundo";
import {
  Node,
  Edge,
  NodeChange,
  EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
} from "@xyflow/react";
import { CalculationNodeData } from "../views/home/type";

export const FLOW_STATE_KEY = 'reactflow-viewport';
export const PROJECT_ID_KEY = 'projectId';

// Viewport for each projectId
export type ProjectViewport = {
  projectId: string | null;
  x: number;
  y: number;
  zoom: number;
};

// Common State
export type FlowStore = {
  sharedNodes: Node<CalculationNodeData>[];
  sharedEdges: Edge[];
  setSharedNodes: (updater: Node<CalculationNodeData>[] | ((nds: Node<CalculationNodeData>[]) => Node<CalculationNodeData>[])) => void;
  setSharedEdges: (updater: Edge[] | ((eds: Edge[]) => Edge[])) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  addNode: (node: Node<CalculationNodeData>) => void;
  removeNode: (nodeId: string) => void;
  removeEdge: (edgeId: string) => void;
  updateNodeData: (nodeId: string, data: Partial<CalculationNodeData>) => void;
  flowRefreshRequestedAt: number;
  flowRefreshInProgress: boolean;
  flowRefreshFlushBeforeLoad: boolean;
  requestFlowRefresh: () => void;
  requestButtonFlowRefresh: () => void;
  setFlowRefreshInProgress: (v: boolean) => void;
  clearFlowRefreshRequest: () => void;
};

// Common State Store with temporal middleware for undo/redo
export const useFlowStore = create<FlowStore>()(
  temporal(
    (set) => ({
      sharedNodes: [],
      sharedEdges: [],
      flowRefreshRequestedAt: 0,
      flowRefreshInProgress: false,
      flowRefreshFlushBeforeLoad: false,
      requestFlowRefresh: () => set({
        flowRefreshRequestedAt: Date.now(),
        flowRefreshFlushBeforeLoad: false,
      }),
      requestButtonFlowRefresh: () => set({
        flowRefreshRequestedAt: Date.now(),
        flowRefreshFlushBeforeLoad: true,
      }),
      setFlowRefreshInProgress: (v) => set({ flowRefreshInProgress: v }),
      clearFlowRefreshRequest: () => set({
        flowRefreshRequestedAt: 0,
        flowRefreshFlushBeforeLoad: false,
      }),
      setSharedNodes: (valueOrFn) =>
        set((state) => ({
          sharedNodes: typeof valueOrFn === 'function'
            ? (valueOrFn as (prev: Node<CalculationNodeData>[]) => Node<CalculationNodeData>[])(state.sharedNodes)
            : valueOrFn
        })),
      setSharedEdges: (valueOrFn) =>
        set((state) => ({
          sharedEdges: typeof valueOrFn === 'function'
            ? (valueOrFn as (prev: Edge[]) => Edge[])(state.sharedEdges)
            : valueOrFn
        })),
      onNodesChange: (changes) =>
        set((state) => ({
          sharedNodes: applyNodeChanges(changes, state.sharedNodes) as Node<CalculationNodeData>[],
        })),
      onEdgesChange: (changes) =>
        set((state) => ({
          sharedEdges: applyEdgeChanges(changes, state.sharedEdges),
        })),
      addNode: (node) =>
        set((state) => ({
          sharedNodes: [...state.sharedNodes, node],
        })),
      removeNode: (nodeId) =>
        set((state) => ({
          sharedNodes: state.sharedNodes.filter((n) => n.id !== nodeId),
          sharedEdges: state.sharedEdges.filter(
            (e) => e.source !== nodeId && e.target !== nodeId
          ),
        })),
      removeEdge: (edgeId) =>
        set((state) => ({
          sharedEdges: state.sharedEdges.filter((e) => e.id !== edgeId),
        })),
      updateNodeData: (nodeId, data) =>
        set((state) => ({
          sharedNodes: state.sharedNodes.map((n) =>
            n.id === nodeId ? { ...n, data: { ...n.data, ...data } } : n
          ),
        })),
    }),
    { limit: 50 }
  )
);

export default useFlowStore;
