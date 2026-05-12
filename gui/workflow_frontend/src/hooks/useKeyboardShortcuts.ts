import { useEffect } from 'react';
import { Node, Edge } from '@xyflow/react';
import { useToast } from '@chakra-ui/react';
import { CalculationNodeData } from '../views/home/type';
import { useFlowStore, FlowStore } from '../stores/flowStore';

interface UseKeyboardShortcutsParams {
  toast: ReturnType<typeof useToast>;
  autoSaveEnabled: boolean;
  isViewOpen: boolean;
  isCodeOpen: boolean;
  onRequestDeleteNodes: (nodeIds: string[]) => void;
  deleteEdgeAPI: (edgeId: string) => Promise<void>;
}

// Reads the latest sharedNodes/sharedEdges from the Zustand store inside the
// keydown handler so callers do not need to subscribe at their level. A
// HomeView-level subscription would re-render the whole tree on every drag
// frame.
export const useKeyboardShortcuts = ({
  toast,
  autoSaveEnabled,
  isViewOpen,
  isCodeOpen,
  onRequestDeleteNodes,
  deleteEdgeAPI,
}: UseKeyboardShortcutsParams) => {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTypingTarget =
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        Boolean(target?.isContentEditable);

      // Disable deletion when modal is open
      if (isViewOpen || isCodeOpen || isTypingTarget) {
        return;
      }

      if (event.key === 'Delete' || event.key === 'Backspace') {
        const { sharedNodes, sharedEdges, setSharedEdges } = useFlowStore.getState() as FlowStore;

        const selectedNodes = sharedNodes.filter((node: Node<CalculationNodeData>) => node.selected);
        if (selectedNodes.length > 0) {
          event.preventDefault();
          onRequestDeleteNodes(selectedNodes.map((node: Node<CalculationNodeData>) => node.id));
          return;
        }

        const selectedEdges = sharedEdges.filter((edge: Edge) => edge.selected);
        if (selectedEdges.length > 0) {
          event.preventDefault();
          if (autoSaveEnabled) {
            selectedEdges.forEach((edge: Edge) => {
              deleteEdgeAPI(edge.id);
            });
          }
          setSharedEdges((eds: Edge[]) => eds.filter((edge: Edge) => !edge.selected));

          toast({
            title: "Deleted",
            description: `${selectedEdges.length} edge(s) deleted`,
            status: "info",
            duration: 2000,
            isClosable: true,
          });
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [toast, autoSaveEnabled, isViewOpen, isCodeOpen, onRequestDeleteNodes, deleteEdgeAPI]);
};

export default useKeyboardShortcuts;
