import { useCallback, useRef, useState, useEffect, useMemo } from 'react';
import {
  Node,
  ReactFlowInstance,
  NodeProps,
} from '@xyflow/react';
import {
  Box,
  Button,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  useDisclosure,
  Text,
  useToast,
  AlertDialog,
  AlertDialogBody,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogOverlay,
  VStack,
} from '@chakra-ui/react';
import { CodeEditorModal } from './components/codeEditorModal';
import { JUPYTER_BASE_URL, API_BASE_URL } from '../../config/urls';
import '@xyflow/react/dist/style.css';
import SideBoxArea from '../box/boxView';
import ChatbotArea from './components/chatbotView';
import { CalculationNodeData, Project, FlowData } from './type';
import { ProjectSelector } from './components/projectSelector';
import { EdgeMenu } from './components/edgeMenu';
import { NodeMenu } from './components/nodeMenu';
import { CalculationNode } from './components/calculationNode';
import { WorkflowCanvasProvider } from './WorkflowCanvas';
import { WorkflowToolbar } from './WorkflowToolbar';
import { createAuthHeaders } from '../../api/authHeaders';
import { useAuth } from '../../auth/authContext';
import { useUploadedNodes } from '../../hooks/useUploadedNodes';
import { useWorkflowApi } from '../../hooks/useWorkflowApi';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import NodeDetailsContent from './components/nodeDetailModal';
import { DeleteConfirmDialog } from './components/deleteConfirmDialog';
import RunStatusPanel from './components/runStatusPanel';
import { useTabContext } from '../../components/tabs/TabManager';
import { useFlowStore, PROJECT_ID_KEY } from '../../stores/flowStore';
import { convertToStrIncFloat } from '../../utils/typeConversion';

const HomeView = () => {
  const toast = useToast();
  const { user, setHasPendingSaves, registerSaveFlusher } = useAuth();
  const currentUserId = user?.id ?? null;
  const prevUserIdRef = useRef<string | null | undefined>(undefined);
  const { data: uploadedNodes, isLoading: isNodesLoading, error, refetch: refetchNodes } = useUploadedNodes();
  const reactFlowInstance = useRef<ReactFlowInstance | null>(null);
  const { isOpen: isCodeOpen, onOpen: onCodeOpen, onClose: onCodeClose } = useDisclosure();
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isGeneratingCode, setIsGeneratingCode] = useState<boolean>(false);

  // Autosave related status
  const [isConnected, setIsConnected] = useState<boolean>(true);
  const [autoSaveEnabled, setAutoSaveEnabled] = useState<boolean>(true);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pendingActionRef = useRef<(() => Promise<void>) | null>(null);
  const inFlightSaveRef = useRef<Promise<void> | null>(null);
  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const updateNodeAPIRef = useRef<(nodeId: string, nodeData: Partial<Node<CalculationNodeData>>) => Promise<void>>();

  // Node menu related status
  const [nodeMenuPosition, setNodeMenuPosition] = useState<{ x: number, y: number } | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Edge menu related status
  const [edgeMenuPosition, setEdgeMenuPosition] = useState<{ x: number, y: number } | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  const { isOpen: isViewOpen, onOpen: onViewOpen, onClose: onViewClose } = useDisclosure();
  const { isOpen: isEditOpen, onOpen: onEditOpen, onClose: onEditClose } = useDisclosure();
  const { isOpen: isDeleteOpen, onOpen: onDeleteOpen, onClose: onDeleteClose } = useDisclosure();
  const {
    isOpen: isNodeDeleteOpen,
    onOpen: onNodeDeleteOpen,
    onClose: onNodeDeleteClose,
  } = useDisclosure();
  const [selectedNode, setSelectedNode] = useState<Node<CalculationNodeData> | null>(null);
  const [nodesPendingDelete, setNodesPendingDelete] = useState<Node<CalculationNodeData>[]>([]);
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);
  const [isDeletingProject, setIsDeletingProject] = useState(false);
  const [isDeletingNodes, setIsDeletingNodes] = useState(false);
  const nodeDeleteCancelRef = useRef<HTMLButtonElement>(null);

  // Comon Selector
  const sharedNodes = useFlowStore(state => state.sharedNodes);
  const setSharedNodes = useFlowStore(state => state.setSharedNodes);
  const sharedEdges = useFlowStore(state => state.sharedEdges);
  const setSharedEdges = useFlowStore(state => state.setSharedEdges);
  const flowRefreshRequestedAt = useFlowStore(state => state.flowRefreshRequestedAt);
  const setFlowRefreshInProgress = useFlowStore(state => state.setFlowRefreshInProgress);
  const clearFlowRefreshRequest = useFlowStore(state => state.clearFlowRefreshRequest);

  // Use the tab system context
  const { addJupyterTab } = useTabContext();

  // Island menu opening/closing management
  const [isIslandCodeOpen, setIslandCodeOpen] = useState(true);

  // Viewing the source code of a workflow project
  const handleOpenJupyter = useCallback(async () => {
    if (!selectedProject) {
      toast({
        title: "No Project Selected",
        description: "Please select a project first",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    try {
      // Get project name
      const projectName = projects.find(p => p.id === selectedProject)?.name || selectedProject;
      // Initial capitalization
      const trimedProjectName = projectName.replace(/\s/g, '').toLowerCase();
      const capitalizedProjectName = trimedProjectName.charAt(0).toUpperCase() + trimedProjectName.slice(1);

      const jupyterUrl = `${JUPYTER_BASE_URL}/user/user1/lab/workspaces/auto-E/tree/codes/projects/${capitalizedProjectName}/${capitalizedProjectName}.py`;
      
      // Create new tab
      addJupyterTab(selectedProject, projectName, jupyterUrl);
      
      toast({
        title: "JupyterLab Tab Created",
        description: `Created tab for project "${projectName}"`,
        status: "success",
        duration: 2000,
        isClosable: true,
      });
      
    } catch (error) {
      console.error('Error creating JupyterLab tab:', error);
      toast({
        title: "Error",
        description: "Failed to create JupyterLab tab",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  }, [selectedProject, projects, addJupyterTab, toast]);

  // Node callback functions
  const handleNodeJupyter = useCallback((nodeId: string) => {
    const node = sharedNodes.find(n => n.id === nodeId);
    if (node) {
      setSelectedNode(node);
      onCodeOpen();
    }
  }, [sharedNodes, onCodeOpen]);


  const handleNodeInfo = useCallback((nodeId: string) => {
    // Try to find node in sharedNodes first
    let node = sharedNodes.find(n => n.id === nodeId);
    
    // If not found, try to get it from ReactFlow instance (for newly added nodes)
    if (!node && reactFlowInstance.current) {
      const flowNodes = reactFlowInstance.current.getNodes();
      node = flowNodes.find(n => n.id === nodeId) as Node<CalculationNodeData> | undefined;
    }
    
    if (node) {
      setSelectedNode(node);
      onViewOpen();
    } else {
      console.warn(`Node ${nodeId} not found for info display`);
      toast({
        title: "Node Not Found",
        description: "Could not find node information. Please try again.",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
    }
  }, [sharedNodes, onViewOpen, reactFlowInstance, toast]);

  // Clear the dirty flag only when nothing is queued, scheduled, or in flight.
  // Prevents an in-flight save's `finally` from clearing the flag while a newer
  // debounced save is already queued behind it.
  const maybeClearPendingState = useCallback(() => {
    if (
      saveTimeoutRef.current === null &&
      pendingActionRef.current === null &&
      inFlightSaveRef.current === null
    ) {
      setHasPendingSaves(false);
    }
  }, [setHasPendingSaves]);

  // Consume whatever is in pendingActionRef and run it, tracking the promise so
  // flushPendingSaves can await an already-running save. The promise stored on
  // inFlightSaveRef rejects when the action rejects, so callers (flush path)
  // can surface the failure; we also log here so debounce-driven errors are
  // not silently lost.
  const runPendingAction = useCallback(async () => {
    const action = pendingActionRef.current;
    pendingActionRef.current = null;
    if (!action) {
      maybeClearPendingState();
      return;
    }
    const promise = action();
    inFlightSaveRef.current = promise;
    try {
      await promise;
    } finally {
      if (inFlightSaveRef.current === promise) {
        inFlightSaveRef.current = null;
      }
      maybeClearPendingState();
    }
  }, [maybeClearPendingState]);

  // Debounced Storage Function
  const debouncedSave = useCallback((action: () => Promise<void>) => {
    if (useFlowStore.getState().flowRefreshInProgress ||
        useFlowStore.getState().flowRefreshRequestedAt > 0) {
      return;
    }

    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    pendingActionRef.current = action;
    setHasPendingSaves(true);

    saveTimeoutRef.current = setTimeout(() => {
      saveTimeoutRef.current = null;
      // Swallow errors on the debounce path so a transient save failure does
      // not turn into an unhandled rejection. The flush path re-throws.
      runPendingAction().catch((err) => {
        console.error("Debounced save failed:", err);
      });
    }, 500);
  }, [setHasPendingSaves, runPendingAction]);

  // Flush any debounced or in-flight save immediately. Called by the re-auth
  // modal before redirecting. Loops so a save queued during the wait is also
  // drained before we return. Rejects if any save fails so the modal can show
  // an error and keep the user from being redirected onto a lost write.
  const flushPendingSaves = useCallback(async () => {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    while (pendingActionRef.current || inFlightSaveRef.current) {
      if (pendingActionRef.current) {
        await runPendingAction();
      } else if (inFlightSaveRef.current) {
        await inFlightSaveRef.current;
      }
    }
    maybeClearPendingState();
  }, [runPendingAction, maybeClearPendingState]);

  // Register the flusher with AuthContext, and on unmount also reset any
  // pending-save bookkeeping. Otherwise a stale hasPendingSaves=true could
  // outlive HomeView and make ReAuthGate offer "Save & re-login" with no
  // flusher available to actually save.
  useEffect(() => {
    registerSaveFlusher(flushPendingSaves);
    return () => {
      registerSaveFlusher(null);
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
        saveTimeoutRef.current = null;
      }
      pendingActionRef.current = null;
      inFlightSaveRef.current = null;
      setHasPendingSaves(false);
    };
  }, [flushPendingSaves, registerSaveFlusher, setHasPendingSaves]);

  const handleNodeUpdate = useCallback((nodeId: string, updatedData: Partial<CalculationNodeData>) => {
    console.log('handleNodeUpdate called for node:', nodeId, 'with data:', updatedData);
    
    setSharedNodes((nds) => {
      const updatedNodes = nds.map((node) => {
        if (node.id === nodeId) {
          // Create an entirely new object and let React Flow recognize the change
          const updatedNode = { 
            ...node, 
            data: { ...node.data, ...updatedData },
            // Add a timestamp to force a re-render
            __timestamp: Date.now()
          };
          console.log('Node updated:', updatedNode);
          return updatedNode;
        }
        return node;
      });
      console.log('Updated nodes array length:', updatedNodes.length);
      return updatedNodes;
    });
    
    // selectedNode also updated
    setSelectedNode((prevNode) => {
      if (prevNode?.id === nodeId) {
        const updatedSelectedNode = {
          ...prevNode,
          data: { ...prevNode.data, ...updatedData }
        };
        console.log('Selected node updated:', updatedSelectedNode);
        return updatedSelectedNode;
      }
      return prevNode;
    });

    // Persist to backend via API (ref always points to the latest updateNodeAPI)
    debouncedSave(async () => {
      const updatedNode = useFlowStore.getState().sharedNodes.find(n => n.id === nodeId);
      if (updatedNode) {
        await updateNodeAPIRef.current?.(nodeId, updatedNode);
      }
    });
  }, [setSharedNodes, debouncedSave]);

  // The handleSyncWorkflowNodes function has been removed. - Sidebar and workflow nodes are treated independently

  // Viewing node information from the sidebar
  const handleSidebarNodeInfo = useCallback((nodeData: any) => {
    // Sidebar nodes are always created as independent temporary nodes
    console.log('Creating temporary node for sidebar view');
    const tempNode = {
      id: `sidebar_${nodeData.id}`,
      data: {
        label: nodeData.label,
        schema: nodeData.schema,
        file_name: nodeData.file_name
      }
    };
    setSelectedNode(tempNode as any);
    onViewOpen();
  }, [onViewOpen]);

  // View source code from the sidebar
  const handleSidebarViewCode = useCallback((nodeData: any) => {
    // Sidebar nodes are always created as independent temporary nodes
    console.log('Creating temporary node for sidebar code view');
    const tempNode = {
      id: `sidebar_${nodeData.id}`,
      data: {
        label: nodeData.label,
        schema: nodeData.schema,
        file_name: nodeData.file_name
      }
    };
    setSelectedNode(tempNode as any);
    onCodeOpen();
  }, [onCodeOpen]);

  // Change category color from the sidebar
  const handleChangeCategoryColor = useCallback((catname: string, color: string) => {
    console.log('Change category color for sidebar color change');
    //let a = uploadedNodes?.categories[catname]['color'] = color;
    handleRefreshNodeData();
  }, [uploadedNodes]);

  const requestNodeDelete = useCallback((nodeIds: string[]) => {
    if (nodeIds.length === 0) {
      return;
    }

    const pendingNodes = sharedNodes.filter((node) => nodeIds.includes(node.id));
    if (pendingNodes.length === 0) {
      return;
    }

    setNodesPendingDelete(pendingNodes);
    onNodeDeleteOpen();
  }, [sharedNodes, onNodeDeleteOpen]);

  const handleNodeDelete = useCallback((nodeId: string) => {
    requestNodeDelete([nodeId]);
  }, [requestNodeDelete]);

  // Define nodeTypes in useMemo - map all category types to calculationNode components
  const nodeTypes = useMemo(() => {
    const calculationNodeComponent = (props: NodeProps<CalculationNodeData>) => (
      <CalculationNode
        {...props}
        onJupyter={handleNodeJupyter}
        onInfo={handleNodeInfo}
        onDelete={handleNodeDelete}
        onNodeUpdate={handleNodeUpdate}
      />
    );

    // basic type
    const types: Record<string, any> = {
      calculationNode: calculationNodeComponent,
      default: calculationNodeComponent, // fallback
    };

    // Dynamically add category types from uploadedNodes.
    // Normalize to lowercase folder name (e.g. 'I/O' → 'io') so the registered
    // type matches what WorkflowCanvas stores in node.data.nodeType.
    if (uploadedNodes?.nodes) {
      const categories = new Set(uploadedNodes.nodes.map(node => node.category));
      categories.forEach(category => {
        if (category) {
          const normalizedCategory = category.toLowerCase().replace('/', '');
          if (!types[normalizedCategory]) {
            types[normalizedCategory] = calculationNodeComponent;
          }
        }
      });
    }

    // Predefined common categories
    const commonCategories = ['analysis', 'preprocessing', 'visualization', 'modeling', 'utils', 'Uploaded Nodes'];
    commonCategories.forEach(category => {
      if (!types[category]) {
        types[category] = calculationNodeComponent;
      }
    });

    return types;
  }, [handleNodeJupyter, handleNodeInfo, handleNodeDelete, handleNodeUpdate, uploadedNodes]);


  // Workflow API hooks
  const { createNodeAPI, updateNodeAPI, deleteNodeAPI, createEdgeAPI, deleteEdgeAPI } = useWorkflowApi({
    selectedProject,
    autoSaveEnabled,
    toast,
    setIsConnected,
  });
  updateNodeAPIRef.current = updateNodeAPI;

  // Monitor connection status
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const headers = await createAuthHeaders();
        const response = await fetch('/api/workflow/', {
          method: 'HEAD',
          credentials: 'include',
          headers: {
            ...headers
          }
        });
        setIsConnected(response.ok);
      } catch (error) {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);

    return () => clearInterval(interval);
  });

  // Execute onChange on first load
  useEffect(() => {
      const projectId = localStorage.getItem(PROJECT_ID_KEY);

    handleProjectChange( projectId );
  }, []);


  // Reset everything tied to the previous user when the signed-in user changes
  // (logout, then login as a different user). Skip the very first run so that
  // the initial mount doesn't wipe state before the user is resolved.
  useEffect(() => {
    if (prevUserIdRef.current === undefined) {
      prevUserIdRef.current = currentUserId;
      return;
    }
    if (prevUserIdRef.current === currentUserId) {
      return;
    }
    prevUserIdRef.current = currentUserId;
    setProjects([]);
    setSelectedProject(null);
    setSharedNodes([]);
    setSharedEdges([]);
    localStorage.removeItem(PROJECT_ID_KEY);
  }, [currentUserId, setSharedNodes, setSharedEdges]);


  // Get project list — refetch whenever the signed-in user changes
  useEffect(() => {
    if (!currentUserId) {
      setProjects([]);
      return;
    }
    const fetchProjects = async () => {
      try {
        console.log('Fetching projects...');
        const header = await createAuthHeaders();
        const response = await fetch('/api/workflow/', {
          credentials: 'include',
          headers: {
            ...header
          }
        });
        console.log('Projects response status:', response.status);
        
        if (response.ok) {
          const data: Project[] = await response.json();
          console.log('Projects data:', data);
          setProjects(data);
          setIsConnected(true);
        } else if (response.status === 401) {
          console.warn('Projects API returned 401 — user not authenticated');
          setProjects([]);
          setIsConnected(false);
          toast({
            title: "Sign in required",
            description: "Please sign in to load your projects.",
            status: "warning",
            duration: 4000,
            isClosable: true,
          });
        } else if (response.status === 403) {
          console.warn('Projects API returned 403 — access denied');
          setIsConnected(false);
          toast({
            title: "Access denied",
            description: "You don't have permission to view this project list.",
            status: "error",
            duration: 4000,
            isClosable: true,
          });
        } else {
          console.error('Projects API failed with status:', response.status);
          setIsConnected(false);
        }
      } catch (error) {
        console.error('Failed to fetch projects:', error);
        setIsConnected(false);
        toast({
          title: "Error",
          description: "Failed to fetch projects",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
      }
    };

    fetchProjects();
  }, [toast, currentUserId]);

  // Start project deletion
  const handleProjectDeleteStart = useCallback((project: Project) => {
    setProjectToDelete(project);
    onDeleteOpen();
  }, [onDeleteOpen]);

  const handleProjectUpdate = useCallback((projectId: string, updates: Partial<Project>) => {
    setProjects(prevProjects =>
      prevProjects.map(project =>
        project.id === projectId ? { ...project, ...updates } : project
      )
    );
  }, []);

  // Executing project deletion
  const handleProjectDelete = useCallback(async () => {
    if (!projectToDelete) return;

    setIsDeletingProject(true);
    try {
      const headers = await createAuthHeaders();
      const response = await fetch(`/api/workflow/${projectToDelete.id}/`, {
        method: 'DELETE',
        credentials: 'include',
        headers: {
          ...headers,
        },
      });

      if (response.ok) {
        // Remove from project list
        setProjects(prevProjects => prevProjects.filter(p => p.id !== projectToDelete.id));
        
        // If the deleted project is selected, clear
        if (selectedProject === projectToDelete.id) {
          setSelectedProject(null);
          setSharedNodes([]);
          setSharedEdges([]);
        }

        toast({
          title: "Project Deleted",
          description: `Project "${projectToDelete.name}" has been successfully deleted`,
          status: "success",
          duration: 3000,
          isClosable: true,
        });

        onDeleteClose();
        setProjectToDelete(null);
      } else {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to delete project');
      }
    } catch (error) {
      console.error('Error deleting project:', error);
      toast({
        title: "Deletion Error",
        description: `Failed to delete project: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsDeletingProject(false);
    }
  }, [projectToDelete, selectedProject, toast, onDeleteClose]);

  // Get flow data when selecting a project
  const handleProjectChange = async (projectId: string) => {
    if (!projectId) {
      setSelectedProject(null);
      setSharedNodes([]);
      setSharedEdges([]);
      localStorage.removeItem(PROJECT_ID_KEY);
      return;
    }

    localStorage.setItem(PROJECT_ID_KEY, projectId)

    setIsLoading(true);
    try {
      const header = await createAuthHeaders();
      const response = await fetch(`/api/workflow/${projectId}/flow/`, {
        credentials: 'include',
        headers:{
          ...header
        }
      });
      if (response.ok) {
        const flowData: FlowData = await response.json();

        // set changed color 
        for (let i = 0; i < flowData.nodes.length; i++) {          
          const cat_name = flowData.nodes[i].data.nodeType.toLowerCase();
          if (uploadedNodes?.categories != null) {
            const node_color = uploadedNodes?.categories[cat_name].color;
            if (flowData.nodes[i].data.color != node_color) {
              flowData.nodes[i].data.color = node_color;
            }
          }
        }

        setSharedNodes(flowData.nodes as Node<CalculationNodeData>[] || []);
        setSharedEdges(flowData.edges || []);
        setSelectedProject(projectId);
        setIsConnected(true);
        localStorage.setItem(PROJECT_ID_KEY, projectId);
        
        toast({
          title: "Loaded",
          description: "Flow data loaded successfully",
          status: "success",
          duration: 2000,
          isClosable: true,
        });
      } else {
        // Cannot load — clear any stale canvas state from a previous user/project
        setSharedNodes([]);
        setSharedEdges([]);
        setSelectedProject(null);
        localStorage.removeItem(PROJECT_ID_KEY);
        setIsConnected(false);
        if (response.status === 404 || response.status === 403) {
          toast({
            title: "Project unavailable",
            description: "The previously selected project is no longer accessible.",
            status: "info",
            duration: 3000,
            isClosable: true,
          });
        }
      }
    } catch (error) {
      console.error('Failed to fetch flow data:', error);
      setSharedNodes([]);
      setSharedEdges([]);
      setIsConnected(false);
      toast({
        title: "Error",
        description: "Failed to load flow data",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Keyboard shortcuts (Delete/Backspace for selected nodes/edges)
  useKeyboardShortcuts({
    sharedNodes,
    sharedEdges,
    setSharedNodes,
    setSharedEdges,
    toast,
    autoSaveEnabled,
    isViewOpen,
    isCodeOpen,
    onRequestDeleteNodes: requestNodeDelete,
    deleteEdgeAPI,
  });

  // handleRefreshNodeData
  const handleRefreshNodeData = useCallback(async (filename: string) => {
    try {
      console.log('Refreshing node data for filename:', filename);
      
      const headers = await createAuthHeaders();
      console.log('Auth headers created:', headers);
      
      const response = await fetch(`/api/box/uploaded-nodes/`, {
        method: 'GET',
        credentials: 'include',
        headers: {
          ...headers,
        },
      });

      console.log('Refresh response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Refresh API error:', errorText);
        throw new Error(`HTTP error! status: ${response.status}: ${errorText}`);
      }

      const result = await response.json();
      console.log('Refresh API result:', result);
      
      // filename Find a node with
      if (result.nodes && Array.isArray(result.nodes)) {
        const refreshedNode = result.nodes.find((node: any) => node.file_name === filename);
        console.log('Found refreshed node:', refreshedNode);
        return refreshedNode;
      }
      
      console.log('No nodes found in result or result.nodes is not an array');
      return null;
    } catch (error) {
      console.error('Error refreshing node data:', error);
      throw error;
    }
  }, []);

  const closeNodeDeleteDialog = useCallback(() => {
    setNodesPendingDelete([]);
    onNodeDeleteClose();
  }, [onNodeDeleteClose]);

  const handleConfirmNodeDelete = useCallback(async () => {
    if (nodesPendingDelete.length === 0) {
      return;
    }

    const nodeIds = nodesPendingDelete.map((node) => node.id);

    try {
      setIsDeletingNodes(true);

      if (autoSaveEnabled) {
        await Promise.all(nodeIds.map((nodeId) => deleteNodeAPI(nodeId)));
      }

      setSharedNodes((nds) => nds.filter((node) => !nodeIds.includes(node.id)));
      setSharedEdges((eds) => {
        const relatedEdges = eds.filter(
          (edge) => nodeIds.includes(edge.source) || nodeIds.includes(edge.target)
        );

        if (autoSaveEnabled) {
          relatedEdges.forEach((edge) => {
            deleteEdgeAPI(edge.id);
          });
        }

        const projectId = localStorage.getItem(PROJECT_ID_KEY);
        handleProjectChange(projectId);

        return eds.filter(
          (edge) => !nodeIds.includes(edge.source) && !nodeIds.includes(edge.target)
        );
      });

      toast({
        title: "Deleted",
        description: `${nodeIds.length} node(s) deleted`,
        status: "info",
        duration: 2000,
        isClosable: true,
      });
      closeNodeDeleteDialog();
    } catch (error) {
      console.error('Error deleting node:', error);
      toast({
        title: "Error",
        description: "Failed to delete node",
        status: "error",
        duration: 2000,
        isClosable: true,
      });
    } finally {
      setIsDeletingNodes(false);
    }
  }, [nodesPendingDelete, autoSaveEnabled, deleteNodeAPI, setSharedNodes, setSharedEdges, deleteEdgeAPI, toast, closeNodeDeleteDialog]);

  // Node deletion process (from menu)
  const handleDeleteNode = useCallback(() => {
    if (selectedNodeId) {
      requestNodeDelete([selectedNodeId]);
    }
  }, [selectedNodeId, requestNodeDelete]);

  // Edge removal process (from the menu)
  const handleDeleteEdge = useCallback(() => {
    if (selectedEdgeId) {
      if (autoSaveEnabled) {
        deleteEdgeAPI(selectedEdgeId);
      }
      
      setSharedEdges((eds) => eds.filter((edge) => edge.id !== selectedEdgeId));
      
      toast({
        title: "Deleted",
        description: `Connection deleted`,
        status: "info",
        duration: 2000,
        isClosable: true,
      });
    }
  }, [selectedEdgeId, setSharedEdges, toast, autoSaveEnabled]);

  // Export the entire flow as JSON
  const handleExportFlowJSON = useCallback(() => {
    if (!reactFlowInstance.current) {
      toast({
        title: "Error",
        description: "Flow instance not ready",
        status: "error",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    try {
      // Get the entire flow using React Flow's toObject() method
      const flowData = reactFlowInstance.current.toObject();
      
      // Include project information
      const exportData = {
        project: {
          id: selectedProject,
          name: projects.find(p => p.id === selectedProject)?.name || 'Unknown',
          exportedAt: new Date().toISOString()
        },
        flow: flowData
      };

      // Download as JSON file
      const jsonString = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const projectName = projects.find(p => p.id === selectedProject)?.name || 'flow';
      const filename = `${projectName}_flow_${new Date().toISOString().split('T')[0]}.json`;
      
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      toast({
        title: "Export Complete",
        description: `Flow exported as ${filename}`,
        status: "success",
        duration: 3000,
        isClosable: true,
      });
      
      console.log('Exported flow data:', exportData);
    } catch (error) {
      console.error('Failed to export flow:', error);
      toast({
        title: "Export Error",
        description: "Failed to export flow data",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    }
  }, [reactFlowInstance, selectedProject, projects, toast]);

  // Code generation (entire flow)
  const handleGenerateCode = useCallback(async () => {
    if (!selectedProject) {
      toast({
        title: "No Project Selected",
        description: "Please select a project first",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    if (!reactFlowInstance.current) {
      toast({
        title: "Flow Not Ready",
        description: "Flow instance is not ready, please wait",
        status: "warning",
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    if (sharedNodes.length === 0) {
      toast({
        title: "Empty Flow",
        description: "Please add nodes to the flow before generating code",
        status: "warning",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    // CRITICAL: Save all nodes to database before generating code
    // This ensures newly added nodes are persisted and won't disappear
    toast({
      title: "Saving nodes...",
      description: "Ensuring all nodes are saved before code generation",
      status: "info",
      duration: 2000,
      isClosable: true,
    });

    // Save all nodes that haven't been saved yet
    // Get current nodes from ReactFlow instance (includes newly added ones)
    const currentNodes = reactFlowInstance.current?.getNodes() || sharedNodes;
    
    for (const node of currentNodes) {
      try {
        await createNodeAPI(node as Node<CalculationNodeData>);
      } catch (error) {
        // Node might already exist, try updating instead
        try {
          await updateNodeAPI(node.id, node as Partial<Node<CalculationNodeData>>);
        } catch (updateError) {
          console.warn(`Failed to save/update node ${node.id}:`, updateError);
        }
      }
    }

    // Wait a bit for all saves to complete
    await new Promise(resolve => setTimeout(resolve, 500));

    setIsGeneratingCode(true);

    // Loading status toast
    const loadingToast = toast({
      title: "Generating Code...",
      description: "Please wait while we generate the code",
      status: "loading",
      duration: null,
      isClosable: false,
    });

    try {
      if (!reactFlowInstance.current) {
        toast.close(loadingToast);
        throw new Error('Flow instance not ready');
      }

      // Get flow data for React Flow
      const flowData = reactFlowInstance.current.toObject();
      console.log('Sending flow data to API:', flowData);

      const headers = await createAuthHeaders();
      const response = await fetch(`/api/workflow/${selectedProject}/generate-code/`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          nodes: flowData.nodes,
          edges: flowData.edges,
          project_id: selectedProject
        }),
      });

      // Close Loading Toast
      toast.close(loadingToast);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`HTTP ${response.status}: ${errorData.error || 'Failed to generate code'}`);
      }

      const result = await response.json();
      console.log('Code generation result:', result);

      toast({
        title: "Code Generated Successfully! ✅",
        description: result.message || "Code has been generated and is ready to use",
        status: "success",
        duration: 5000,
        isClosable: true,
      });

    } catch (error) {
      // Close loading toast (on error)
      toast.close(loadingToast);
      
      console.error('Code generation error:', error);
      toast({
        title: "Code Generation Failed ❌",
        description: `Failed to generate code: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsGeneratingCode(false);
    }
  }, [selectedProject, reactFlowInstance, sharedNodes, createNodeAPI, updateNodeAPI, toast]);

  // Clean up
  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, []);

  // Watch for flow refresh requests from MCP tool completions
  useEffect(() => {
    if (flowRefreshRequestedAt === 0) return;

    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
    }

    refreshTimeoutRef.current = setTimeout(async () => {
      const projectId = localStorage.getItem(PROJECT_ID_KEY);
      if (!projectId) {
        clearFlowRefreshRequest();
        return;
      }

      setFlowRefreshInProgress(true);
      try {
        const header = await createAuthHeaders();
        const response = await fetch(`/api/workflow/${projectId}/flow/`, {
          credentials: 'include',
          headers: { ...header },
        });
        if (response.ok) {
          const flowData: FlowData = await response.json();

          // Apply category colors (same logic as handleProjectChange)
          for (let i = 0; i < flowData.nodes.length; i++) {
            const cat_name = flowData.nodes[i].data.nodeType.toLowerCase();
            if (uploadedNodes?.categories != null) {
              const node_color = uploadedNodes?.categories[cat_name]?.color;
              if (node_color && flowData.nodes[i].data.color !== node_color) {
                flowData.nodes[i].data.color = node_color;
              }
            }
          }

          setSharedNodes(flowData.nodes as Node<CalculationNodeData>[] || []);
          setSharedEdges(flowData.edges || []);
        }
      } catch (error) {
        console.error('Failed to refresh flow data after MCP tool:', error);
      } finally {
        clearFlowRefreshRequest();
        setFlowRefreshInProgress(false);
      }
    }, 750);

    return () => {
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, [flowRefreshRequestedAt, uploadedNodes, setSharedNodes, setSharedEdges, setFlowRefreshInProgress, clearFlowRefreshRequest]);

  // ReactFlow closeMenu(onPanelClick)
  const closeMenu = useCallback(() => {
    setNodeMenuPosition(null);
    setSelectedNodeId(null);
    setEdgeMenuPosition(null);
    setSelectedEdgeId(null);
  }, []);

  // ReactFlow onPaneClick
  const onPaneClick = useCallback(() => {
    closeMenu();
  }, [closeMenu]);


  // OistWorkFlow extracted to WorkflowCanvas.tsx

  //////////////////////////////////////////////////////////////
  // homeView: Main Page
  //////////////////////////////////////////////////////////////
  return (
    <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, overflow: 'hidden' }}>
      <SideBoxArea
        nodes={uploadedNodes}
        isLoading={isNodesLoading}
        error={error}
        onRefresh={refetchNodes}
        onNodeInfo={handleSidebarNodeInfo}
        onViewCode={handleSidebarViewCode}
        onChangeColor={handleChangeCategoryColor}
      />
      <ChatbotArea />
        <style>
          {`
            .react-flow__controls {
              background: transparent;
            }
            
            .react-flow__controls-button {
              background: white;
              border: 1px solid #e2e8f0;
              color: #4a5568;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .react-flow__controls-button:hover {
              background: #f7fafc;
              border-color: #cbd5e0;
            }
            
            .react-flow__controls-button svg {
              fill: #4a5568;
            }
            
            .react-flow__minimap {
              background-color: #f8f9fa;
              border: 1px solid #e2e8f0;
            }
            
            .react-flow__minimap-mask {
              fill: rgba(50, 50, 50, 0.8);
            }
            
            .react-flow__minimap-node {
              fill: #8b5cf6;
              stroke: #7c3aed;
            }
        `}
        </style>
        
        {/* Project selection UI */}
        <ProjectSelector
          projects={projects}
          selectedProject={selectedProject}
          onProjectChange={handleProjectChange}
          onProjectDelete={handleProjectDeleteStart}
          onProjectUpdate={handleProjectUpdate}
          autoSaveEnabled={autoSaveEnabled}
          isConnected={isConnected}
        />
        <WorkflowToolbar
          isIslandCodeOpen={isIslandCodeOpen}
          setIslandCodeOpen={setIslandCodeOpen}
          isConnected={isConnected}
          autoSaveEnabled={autoSaveEnabled}
          selectedProject={selectedProject}
          projects={projects}
          sharedNodes={sharedNodes}
          isGeneratingCode={isGeneratingCode}
          handleOpenJupyter={handleOpenJupyter}
          handleGenerateCode={handleGenerateCode}
          handleExportFlowJSON={handleExportFlowJSON}
        />
          
        {/* ReactFlow: Needs to be wrapped in ReactFlowProvider */}
        <WorkflowCanvasProvider
          reactFlowInstance={reactFlowInstance}
          selectedProject={selectedProject}
          autoSaveEnabled={autoSaveEnabled}
          toast={toast}
          nodeTypes={nodeTypes}
          onPaneClick={onPaneClick}
          setEdgeMenuPosition={setEdgeMenuPosition}
          setSelectedEdgeId={setSelectedEdgeId}
          createNodeAPI={createNodeAPI}
          updateNodeAPI={updateNodeAPI}
          deleteNodeAPI={deleteNodeAPI}
          createEdgeAPI={createEdgeAPI}
          deleteEdgeAPI={deleteEdgeAPI}
          debouncedSave={debouncedSave}
          uploadedNodes={uploadedNodes}
          handleRefreshNodeData={handleRefreshNodeData}
          handleNodeUpdate={handleNodeUpdate}
        />
        {isLoading && (
          <Box
            position="absolute"
            top="50%"
            left="50%"
            transform="translate(-50%, -50%)"
            bg="white"
            p={4}
            borderRadius="md"
            boxShadow="lg"
            zIndex={1000}
          >
            <Text>Loading...</Text>
          </Box>
        )}
        
        {/* Async run status panel */}
        {selectedProject && (
          <RunStatusPanel workflowId={selectedProject} />
        )}

        {/* Node menu */}
        {nodeMenuPosition && (
          <NodeMenu
            position={nodeMenuPosition}
            onDelete={handleDeleteNode}
            onView={onViewOpen}
            onEdit={onEditOpen}
            onClose={closeMenu}
          />
        )}
        
        {/* Edge menu */}
        {edgeMenuPosition && (
          <EdgeMenu
            position={edgeMenuPosition}
            onDelete={handleDeleteEdge}
            onClose={closeMenu}
          />
        )}

        {/* View Modal */}
        <Modal isOpen={isViewOpen} onClose={onViewClose} size="2xl">
          <ModalOverlay />
          <ModalContent maxW="1200px" w="90vw">
            <ModalHeader>Node Details: {/* selectedNode?.data.label */}</ModalHeader>
            <ModalCloseButton />
            <ModalBody marginTop={5}>
              <NodeDetailsContent
                nodeData={selectedNode}
                onNodeUpdate={handleNodeUpdate}
                onRefreshNodeData={handleRefreshNodeData}
                onViewCode={() => {
                  onViewClose();
                  onCodeOpen();
                }}
                convertToStrIncFloat={convertToStrIncFloat}
                workflowId={selectedProject || undefined}
              />
            </ModalBody>
            <ModalFooter>
              <Button variant="ghost" onClick={onViewClose}>Close</Button>
            </ModalFooter>
          </ModalContent>
        </Modal>

        {/* Code Editor Modal */}
        <CodeEditorModal
          isOpen={isCodeOpen}
          onClose={onCodeClose}
          identifier={selectedNode?.data.file_name || ''}
          endpoints={{
            baseUrl: `${API_BASE_URL}/box`,
            getCode: '/files/{identifier}/code/',
            saveCode: '/files/{identifier}/code/',
          }}
          title={selectedNode ? `Code: ${selectedNode.data.label}` : 'Code Editor'}
          downloadFileName={selectedNode?.data.file_name || 'code.py'}
          showExecute={false}
          language="python"
        />

        {/* Project deletion confirmation dialog */}
        <DeleteConfirmDialog
          isOpen={isDeleteOpen}
          onClose={() => {
            onDeleteClose();
            setProjectToDelete(null);
          }}
          onConfirm={handleProjectDelete}
          project={projectToDelete}
          isDeleting={isDeletingProject}
        />

        <AlertDialog
          isOpen={isNodeDeleteOpen}
          leastDestructiveRef={nodeDeleteCancelRef}
          onClose={() => {
            if (!isDeletingNodes) {
              closeNodeDeleteDialog();
            }
          }}
        >
          <AlertDialogOverlay>
            <AlertDialogContent>
              <AlertDialogHeader fontSize="lg" fontWeight="bold" color="red.600">
                Delete Node{nodesPendingDelete.length === 1 ? '' : 's'}
              </AlertDialogHeader>

              <AlertDialogBody>
                <VStack spacing={3} align="stretch">
                  <Text>
                    You are about to permanently delete {nodesPendingDelete.length} node{nodesPendingDelete.length === 1 ? '' : 's'} from this workflow.
                  </Text>

                  {nodesPendingDelete.length === 1 ? (
                    <Text
                      p={3}
                      bg="red.50"
                      borderRadius="md"
                      borderLeft="4px solid"
                      borderColor="red.400"
                      fontWeight="semibold"
                      color="red.700"
                    >
                      Node Name: {nodesPendingDelete[0]?.data.instanceName || nodesPendingDelete[0]?.data.label || nodesPendingDelete[0]?.id}
                    </Text>
                  ) : (
                    <Box
                      p={3}
                      bg="red.50"
                      borderRadius="md"
                      borderLeft="4px solid"
                      borderColor="red.400"
                      color="red.700"
                    >
                      {nodesPendingDelete.map((node) => (
                        <Text key={node.id} fontWeight="semibold">
                          • {node.data.instanceName || node.data.label || node.id}
                        </Text>
                      ))}
                    </Box>
                  )}

                  <Text fontSize="sm" color="gray.600">
                    Connected edges will also be removed. This action cannot be undone.
                  </Text>
                </VStack>
              </AlertDialogBody>

              <AlertDialogFooter>
                <Button
                  ref={nodeDeleteCancelRef}
                  onClick={closeNodeDeleteDialog}
                  isDisabled={isDeletingNodes}
                >
                  Cancel
                </Button>
                <Button
                  colorScheme="red"
                  onClick={handleConfirmNodeDelete}
                  ml={3}
                  isLoading={isDeletingNodes}
                  loadingText="Deleting..."
                >
                  Delete Node{nodesPendingDelete.length === 1 ? '' : 's'}
                </Button>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogOverlay>
        </AlertDialog>
    </div>
  );
}

export default HomeView;
