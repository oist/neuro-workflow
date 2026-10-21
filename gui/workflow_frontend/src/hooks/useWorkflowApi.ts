import { Node, Edge } from '@xyflow/react';
import { useToast } from '@chakra-ui/react';
import { CalculationNodeData } from '../views/home/type';
import { createAuthHeaders } from '../api/authHeaders';
import { useFlowStore } from '../stores/flowStore';

interface UseWorkflowApiParams {
  selectedProject: string | null;
  autoSaveEnabled: boolean;
  toast: ReturnType<typeof useToast>;
  setIsConnected: (connected: boolean) => void;
}

const createAuthHeadersLocal = async () => {
  return await createAuthHeaders();
};

export const useWorkflowApi = ({
  selectedProject,
  autoSaveEnabled,
  toast,
  setIsConnected,
}: UseWorkflowApiParams) => {
  // Creating individual nodes
  const createNodeAPI = async (nodeData: Node<CalculationNodeData>) => {
    if (!selectedProject || !autoSaveEnabled) {
      console.log('Skipping node creation API call:', { selectedProject, autoSaveEnabled });
      return;
    }

    console.log('Creating node via API:', nodeData);

    try {
      const headers = await createAuthHeadersLocal();
      const requestBody = {
        id: nodeData.id,
        position: nodeData.position,
        type: nodeData.type,
        data: nodeData.data,
      };

      console.log('Request body:', requestBody);

      const response = await fetch(`/api/workflow/${selectedProject}/nodes/`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const responseData = await response.json();
      console.log('Create node response:', responseData);

      if (!response.ok) {
        setIsConnected(false);
        throw new Error(`HTTP ${response.status}: ${responseData.error || 'Failed to create node'}`);
      }

      setIsConnected(true);
      console.log('Node created successfully:', responseData);

      // A class this project has its own copy of: show the copy's schema
      const saved = responseData.data?.data;
      if (saved?.projectNodePath) {
        useFlowStore.getState().updateNodeData(nodeData.id, {
          schema: saved.schema,
          projectNodePath: saved.projectNodePath,
        });
      }
    } catch (error) {
      console.error('Error creating node:', error);
      setIsConnected(false);
      toast({
        title: "Save Error",
        description: `Failed to save node: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  };

  // Individual node updates
  const updateNodeAPI = async (nodeId: string, nodeData: Partial<Node<CalculationNodeData>>) => {
    if (!selectedProject || !autoSaveEnabled) {
      console.log('Skipping node update API call:', { selectedProject, autoSaveEnabled });
      return;
    }

    console.log('Updating node via API:', { nodeId, nodeData });

    try {
      const headers = await createAuthHeadersLocal();
      const requestBody = {
        position: nodeData.position,
        type: nodeData.type,
        data: nodeData.data,
      };

      console.log('Update request body:', requestBody);

      const response = await fetch(`/api/workflow/${selectedProject}/nodes/${nodeId}/`, {
        method: 'PUT',
        credentials: 'include',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const responseData = await response.json();
      console.log('Update node response:', responseData);

      if (!response.ok) {
        setIsConnected(false);
        throw new Error(`HTTP ${response.status}: ${responseData.error || 'Failed to update node'}`);
      }

      setIsConnected(true);
    } catch (error) {
      console.error('Error updating node:', error);
      setIsConnected(false);
      toast({
        title: "Save Error",
        description: `Failed to update node: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 2000,
        isClosable: true,
      });
    }
  };

  // Deleting nodes individually
  const deleteNodeAPI = async (nodeId: string) => {
    if (!selectedProject || !autoSaveEnabled) {
      console.log('Skipping node deletion API call:', { selectedProject, autoSaveEnabled });
      return;
    }

    console.log('Deleting node via API:', nodeId);

    try {
      const headers = await createAuthHeadersLocal();
      const response = await fetch(`/api/workflow/${selectedProject}/nodes/${nodeId}/`, {
        method: 'DELETE',
        credentials: 'include',
        headers: {
          ...headers,
        },
      });

      let responseData;
      if (response.status !== 204) {
        responseData = await response.json();
        console.log('Delete node response:', responseData);
      }

      if (!response.ok) {
        setIsConnected(false);
        throw new Error(`HTTP ${response.status}: ${responseData?.error || 'Failed to delete node'}`);
      }

      setIsConnected(true);
      console.log('Node deleted successfully');
    } catch (error) {
      console.error('Error deleting node:', error);
      setIsConnected(false);
      toast({
        title: "Save Error",
        description: `Failed to delete node: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 2000,
        isClosable: true,
      });
    }
  };

  // Creating edges individually
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const createEdgeAPI = async (edgeData: Edge) => {
    if (!selectedProject || !autoSaveEnabled) {
      console.log('Skipping edge creation API call:', { selectedProject, autoSaveEnabled });
      return;
    }

    console.log('Creating edge via API:', edgeData);

    try {
      const headers = await createAuthHeadersLocal();
      const requestBody = {
        id: edgeData.id,
        source: edgeData.source,
        target: edgeData.target,
        sourceHandle: edgeData.sourceHandle,
        targetHandle: edgeData.targetHandle,
        data: edgeData.data || {},
      };

      console.log('Edge request body:', requestBody);

      const response = await fetch(`/api/workflow/${selectedProject}/edges/`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const responseData = await response.json();
      console.log('Create edge response:', responseData);

      if (!response.ok) {
        setIsConnected(false);
        throw new Error(`HTTP ${response.status}: ${responseData.error || 'Failed to create edge'}`);
      }

      setIsConnected(true);
    } catch (error) {
      console.error('Error creating edge:', error);
      setIsConnected(false);
      toast({
        title: "Save Error",
        description: `Failed to save connection: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 2000,
        isClosable: true,
      });
    }
  };

  // Deleting individual edges
  const deleteEdgeAPI = async (edgeId: string) => {
    if (!selectedProject || !autoSaveEnabled) {
      console.log('Skipping edge deletion API call:', { selectedProject, autoSaveEnabled });
      return;
    }

    console.log('Deleting edge via API:', edgeId);

    try {
      const headers = await createAuthHeadersLocal();
      const response = await fetch(`/api/workflow/${selectedProject}/edges/${edgeId}/`, {
        method: 'DELETE',
        credentials: 'include',
        headers: {
          ...headers,
        },
      });

      let responseData;
      if (response.status !== 204) {
        responseData = await response.json();
        console.log('Delete edge response:', responseData);
      }

      if (!response.ok) {
        setIsConnected(false);
        throw new Error(`HTTP ${response.status}: ${responseData?.error || 'Failed to delete edge'}`);
      }

      setIsConnected(true);
      console.log('Edge deleted successfully');
    } catch (error) {
      console.error('Error deleting edge:', error);
      setIsConnected(false);
      toast({
        title: "Save Error",
        description: `Failed to delete connection: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 2000,
        isClosable: true,
      });
    }
  };

  return {
    createNodeAPI,
    updateNodeAPI,
    deleteNodeAPI,
    createEdgeAPI,
    deleteEdgeAPI,
    createAuthHeadersLocal,
  };
};

export default useWorkflowApi;
