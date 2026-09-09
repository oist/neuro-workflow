import React, { useCallback, useState, useEffect, MutableRefObject } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  Viewport,
  MiniMap,
  Controls,
  Background,
  addEdge,
  Node,
  Edge,
  BackgroundVariant,
  Connection,
  ReactFlowInstance,
  NodeMouseHandler,
  EdgeMouseHandler,
  NodeChange,
  EdgeChange,
  OnMoveEnd,
} from '@xyflow/react';
import { useToast } from '@chakra-ui/react';
import '@xyflow/react/dist/style.css';
import { SchemaFields, CalculationNodeData } from './type';
import { controlsStyle, minimapStyle } from './style';
import { parseHandleId, generateEdgeId } from '@/utils/handleId';
import { useFlowStore, FLOW_STATE_KEY, PROJECT_ID_KEY, ProjectViewport } from '../../stores/flowStore';
import { isPaletteNodeDroppable } from '../box/paletteFilter';

// UploadedNode type (matches backend node structure)
interface UploadedNode {
  id: string;
  type: string;
  label: string;
  name?: string;
  description: string;
  category: string;
  category_key?: string;
  file_name: string;
  schema: SchemaFields;
  nodeType?: string;
  color?: string;
  is_own?: boolean;
  parse_ok?: boolean;
  draggable?: boolean;
}

const readStoredViewports = (): ProjectViewport[] => {
  const viewportStr = localStorage.getItem(FLOW_STATE_KEY);
  if (!viewportStr) return [];
  try {
    const parsed = JSON.parse(viewportStr);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn('Ignoring invalid saved viewport state:', error);
    localStorage.removeItem(FLOW_STATE_KEY);
    return [];
  }
};

export interface WorkflowCanvasProps {
  reactFlowInstance: MutableRefObject<ReactFlowInstance<Node<CalculationNodeData>, Edge> | null>;
  selectedProject: string | null;
  autoSaveEnabled: boolean;
  toast: ReturnType<typeof useToast>;
  nodeTypes: Record<string, any>;
  onPaneClick: () => void;
  setEdgeMenuPosition: (pos: { x: number; y: number } | null) => void;
  setSelectedEdgeId: (id: string | null) => void;
  createNodeAPI: (nodeData: Node<CalculationNodeData>) => Promise<void>;
  updateNodeAPI: (nodeId: string, nodeData: Partial<Node<CalculationNodeData>>) => Promise<void>;
  deleteNodeAPI: (nodeId: string) => Promise<void>;
  createEdgeAPI: (edgeData: Edge) => Promise<void>;
  deleteEdgeAPI: (edgeId: string) => Promise<void>;
  debouncedSave: (action: () => Promise<void>) => void;
  uploadedNodes: any;
  handleRefreshNodeData: (filename: string) => Promise<any>;
  handleNodeUpdate: (nodeId: string, updatedData: Partial<CalculationNodeData>) => void;
}

const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  reactFlowInstance,
  selectedProject,
  autoSaveEnabled,
  toast,
  nodeTypes,
  onPaneClick,
  setEdgeMenuPosition,
  setSelectedEdgeId,
  createNodeAPI,
  updateNodeAPI,
  deleteNodeAPI,
  createEdgeAPI,
  deleteEdgeAPI,
  debouncedSave,
  uploadedNodes,
  handleRefreshNodeData,
  handleNodeUpdate,
}) => {
  // Read nodes/edges directly from Zustand (single source of truth)
  const sharedNodes = useFlowStore((s) => s.sharedNodes);
  const sharedEdges = useFlowStore((s) => s.sharedEdges);
  const setSharedEdges = useFlowStore((s) => s.setSharedEdges);
  const storeOnNodesChange = useFlowStore((s) => s.onNodesChange);
  const storeOnEdgesChange = useFlowStore((s) => s.onEdgesChange);
  const addNode = useFlowStore((s) => s.addNode);

  const projectId = localStorage.getItem(PROJECT_ID_KEY);

  const vps = readStoredViewports();
  let fvp: ProjectViewport = vps.find(view => view.projectId === projectId) ?? {
    projectId: projectId,
    x: 0,
    y: 0,
    zoom: 1,
  };
  /* Enable this to save zoom and pan ===> */
  const [initialViewport] = useState<Viewport>(fvp);
  const { setViewport } = useReactFlow();

  // onLoad: Read localStorage value
  useEffect(() => {
    const saved = localStorage.getItem(FLOW_STATE_KEY);
    if (saved) {
      const vps = readStoredViewports();
      const fvp = vps.find(view => view.projectId === projectId);
      if(fvp != undefined){
        setViewport(fvp);
        requestAnimationFrame(() => setViewport(fvp));
      }
    }
  }, [setViewport, projectId]);
  /* <=== Enable this to save zoom and pan */

  // ReactFlow onInit
  const onInit = useCallback((instance: ReactFlowInstance<Node<CalculationNodeData>, Edge>) => {
    reactFlowInstance.current = instance;
  }, [reactFlowInstance]);

  // ReactFlow onDrop
  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      if (!reactFlowInstance.current) {
        console.log('ReactFlow instance not ready');
        return;
      }

      if (!selectedProject) {
        console.log('No project selected');
        toast({
          title: "No Project",
          description: "Please select a project first",
          status: "warning",
          duration: 2000,
          isClosable: true,
        });
        return;
      }

      const reactFlowBounds = event.currentTarget.getBoundingClientRect();
      const position = reactFlowInstance.current.screenToFlowPosition({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const nodeDataString = event.dataTransfer.getData('application/nodedata');
      let nodeData;
      try {
        nodeData = JSON.parse(nodeDataString);
      } catch (error) {
        console.error('Invalid node data:', error);
        return;
      }

      if (!nodeData) {
        console.log('No node data received');
        return;
      }

      if (!isPaletteNodeDroppable(nodeData)) {
        return;
      }

      console.log('====================================');
      console.log('🔄 NEW DROP EVENT');
      console.log('Dropped nodeData:', nodeData);
      console.log('NodeData ID:', nodeData.id);
      console.log('NodeData Label:', nodeData.label);
      console.log('====================================');

      let schema: SchemaFields = {
        inputs: {},
        outputs: {},
        parameters: {},
        methods: {}
      };
      let nodeType = 'calculationNode';
      let label = nodeData.label || nodeData.name || 'New Calculator';
      let fileName: string = "";
      let color: string = nodeData.color;
      // Get the schema of the corresponding node from uploadedNodes
      if (uploadedNodes?.nodes && Array.isArray(uploadedNodes.nodes)) {
        console.log('Available nodes in uploadedNodes:', uploadedNodes.nodes.length);

        // Matching process
        let matchedNode: UploadedNode | null = null;
        const droppableNodes = uploadedNodes.nodes.filter((node: UploadedNode) =>
          isPaletteNodeDroppable(node)
        );

        // Attempt an exact match on the ID
        if (nodeData.id) {
          matchedNode = droppableNodes.find((node: UploadedNode) => node.id === nodeData.id);
          if (matchedNode) {
            console.log('✅ Matched by ID:', nodeData.id);
          }
        }

        // If no match by ID, try by label
        if (!matchedNode && nodeData.label) {
          matchedNode = droppableNodes.find((node: UploadedNode) => node.label === nodeData.label);
          if (matchedNode) {
            console.log('✅ Matched by label:', nodeData.label);
          }
        }

        // If that doesn't match, try by name
        if (!matchedNode && nodeData.name) {
          matchedNode = droppableNodes.find((node: UploadedNode) => node.name === nodeData.name);
          if (matchedNode) {
            console.log('✅ Matched by name:', nodeData.name);
          }
        }

        if (matchedNode && matchedNode.schema) {
          console.log('📋 Processing schema for:', matchedNode.label);
          console.log('Original schema structure:', matchedNode.schema);

          // Use the new structure schema as is
          schema = matchedNode.schema;

          // Check the contents of the schema
          const inputCount = schema.inputs ? Object.keys(schema.inputs).length : 0;
          const outputCount = schema.outputs ? Object.keys(schema.outputs).length : 0;
          const paramCount = schema.parameters ? Object.keys(schema.parameters).length : 0;
          const methodCount = schema.methods ? Object.keys(schema.methods).length : 0;

          console.log(`✅ Schema loaded: ${inputCount} inputs, ${outputCount} outputs, ${paramCount} parameters, ${methodCount} methods`);

          // If you need a default schema
          if (inputCount === 0 && outputCount === 0) {
            console.warn('⚠️ No ports found, using default schema');
            schema = {
              inputs: {
                "default_input": {
                  type: "any",
                  description: "Default input",
                }
              },
              outputs: {
                "default_output": {
                  type: "any",
                  description: "Default output",
                }
              },
              parameters: {},
              methods: {}
            };
          }

          // Get the correct label and type from matchedNode.
          // Normalize to lowercase folder name (e.g. 'I/O' → 'io') to match
          // backend validation which expects the directory name, not display name.
          const rawCategory = matchedNode.category || matchedNode.nodeType || matchedNode.type || 'calculationNode';
          nodeType = rawCategory.toLowerCase().replace('/', '');
          label = matchedNode.label || matchedNode.name || label;
          fileName = matchedNode.file_name || "" ;
          // add ".py"
          const chkPy = fileName.includes(".py");
          if (!chkPy) {
            fileName += ".py";
          }
        } else {
          console.log('❌ No matching node found, using fallback schema');
          // fallback schema
          schema = {
            inputs: {
              "input": {
                type: "any",
                description: "Input",
              }
            },
            outputs: {
              "output": {
                type: "any",
                description: "Output",
              }
            },
            parameters: {},
            methods: {}
          };
        }
      } else {
        console.warn('❌ uploadedNodes not available, using default schema');
      }

      // Generate new ID
      const newNodeId = `calc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Create a deep copy of the schema so that each node has independent parameter values
      const independentSchema = JSON.parse(JSON.stringify(schema));

      const newNode: Node<CalculationNodeData> = {
        id: newNodeId,
        type: nodeType,
        position,
        data: {
          file_name: fileName,
          label: label,
          instanceName: label,  // Default instance name
          schema: independentSchema,
          nodeType: nodeType,
          // Initialize with empty nodeParameters (for future parameter changes)
          nodeParameters: {},
          color: color,
        },
      };

      console.log('🎯 Creating NEW node:');
      console.log('Node data', newNode)
      console.log('  ID:', newNodeId);
      console.log('  Label:', label);
      console.log('  Schema:', schema);
      console.log(' file name:', fileName);
      console.log('====================================');

      // Add to Zustand store (single source of truth)
      addNode(newNode);

      // Send to API
      if (autoSaveEnabled) {
        createNodeAPI(newNode);
      }

      // For workflow nodes, auto-refresh is skipped to preserve individual parameter values
      console.log('Skipping auto-refresh for workflow node to maintain independent parameters:', newNodeId);

      // Count calculation (supports new structure)
      const inputCount = schema.inputs ? Object.keys(schema.inputs).length : 0;
      const outputCount = schema.outputs ? Object.keys(schema.outputs).length : 0;

      toast({
        title: "Node Added",
        description: `"${label}" (${inputCount} inputs, ${outputCount} outputs)`,
        status: "success",
        duration: 2000,
        isClosable: true,
      });
    },
    [addNode, toast, selectedProject, autoSaveEnabled, uploadedNodes, handleRefreshNodeData, handleNodeUpdate, createNodeAPI]
  );

  // ReactFlow onDragOver
  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Mouse Moved (Zoom Pan)
  const onMoveEnd: OnMoveEnd = useCallback((_, viewport) => {
    console.log("Move End:", viewport);

    const viewportList = readStoredViewports();

    // New Viewport
    const vp = {
      projectId: selectedProject,
      x: viewport.x,
      y: viewport.y,
      zoom: viewport.zoom,
    };

    // Update or Add
    const index = viewportList.findIndex(item => item.projectId === selectedProject);
    if (index !== -1) {
      viewportList[index] = vp;
    } else {
      viewportList.push(vp);
    }

    localStorage.setItem(FLOW_STATE_KEY, JSON.stringify(viewportList));
  }, [selectedProject]);

  // ReactFlow: Node change handler with API side-effects
  const handleNodesChange = useCallback((changes: NodeChange[]) => {
    // Update Zustand store (single source of truth)
    storeOnNodesChange(changes);

    if (!autoSaveEnabled) return;

    changes.forEach((change) => {
      switch (change.type) {
        case 'position':
          if (change.position) {
            debouncedSave(() => updateNodeAPI(change.id, {
              position: change.position
            }));
          }
          break;

        case 'remove':
          deleteNodeAPI(change.id);
          break;
      }
    });
  }, [storeOnNodesChange, debouncedSave, autoSaveEnabled, updateNodeAPI, deleteNodeAPI]);

  // ReactFlow: Edge change handler with API side-effects
  const handleEdgesChange = useCallback((changes: EdgeChange[]) => {
    // Update Zustand store (single source of truth)
    storeOnEdgesChange(changes);

    if (!autoSaveEnabled) return;

    changes.forEach((change) => {
      switch (change.type) {
        case 'remove':
          deleteEdgeAPI(change.id);
          break;
      }
    });
  }, [storeOnEdgesChange, autoSaveEnabled, deleteEdgeAPI]);

  // ReactFlow: On connect handler (edge creation) - with type checking
  const onConnect = useCallback(
    (params: Connection) => {
      // Parse handle IDs using the handleId utility (supports both :: and legacy - formats)
      const sourceParsed = params.sourceHandle ? parseHandleId(params.sourceHandle) : null;
      const targetParsed = params.targetHandle ? parseHandleId(params.targetHandle) : null;

      const sourceType = sourceParsed?.portType ?? null;
      const targetType = targetParsed?.portType ?? null;
      const sourcePortName = sourceParsed?.fieldName ?? null;
      const targetPortName = targetParsed?.fieldName ?? null;

      if (sourceParsed) {
        console.log('Source handle parsing:', {
          handle: params.sourceHandle,
          portName: sourcePortName,
          portDirection: sourceParsed.handleType,
          type: sourceType
        });
      }

      if (targetParsed) {
        console.log('Target handle parsing:', {
          handle: params.targetHandle,
          portName: targetPortName,
          portDirection: targetParsed.handleType,
          type: targetType
        });
      }

      // If the type cannot be obtained
      if (!sourceType || !targetType) {
        toast({
          title: "Connection Failed",
          description: "Could not determine port types",
          status: "error",
          duration: 3000,
          isClosable: true,
        });
        return;
      }

      // If the types don't match
      if (sourceType.toUpperCase() !== targetType.toUpperCase()) {
        toast({
          title: "Type Mismatch",
          description: `Cannot connect: ${sourcePortName || 'output'} (${sourceType}) and ${targetPortName || 'input'} (${targetType}) have different types`,
          status: "warning",
          duration: 4000,
          isClosable: true,
        });
        console.warn(
          `Type mismatch: ${sourcePortName} (${sourceType}) → ${targetPortName} (${targetType})`
        );
        return;
      }

      // Block duplicate connections unless the target port has fan_in=true
      if (targetPortName && params.target) {
        const targetNode = sharedNodes.find((n) => n.id === params.target);
        const targetPortSchema = targetNode?.data?.schema?.inputs?.[targetPortName];
        const isFanIn = targetPortSchema?.fan_in === true;

        if (!isFanIn) {
          const existingEdges = useFlowStore.getState().sharedEdges;
          const duplicate = existingEdges.some(
            (e) => e.target === params.target && e.targetHandle === params.targetHandle
          );
          if (duplicate) {
            toast({
              title: "Connection Not Allowed",
              description: `Port "${targetPortName}" only accepts one connection.`,
              status: "warning",
              duration: 4000,
              isClosable: true,
            });
            return;
          }
        }
      }

      // Create a connection if the types match
      const edgeId = generateEdgeId(
        params.source ?? '',
        params.sourceHandle || 'output',
        params.target ?? '',
        params.targetHandle || 'input'
      );

      const newEdge: Edge = {
        id: edgeId,
        ...params,
        style: { stroke: '#aaaaaa', strokeWidth: 2 }
      };

      console.log('Creating new edge:', {
        edge: newEdge,
        sourcePort: `${sourcePortName} (${sourceType})`,
        targetPort: `${targetPortName} (${targetType})`,
        typesMatch: true
      });

      // Update Zustand store directly
      setSharedEdges((eds) => {
        const updatedEdges = addEdge(newEdge, eds);
        console.log('Updated edges state:', updatedEdges.length);
        return updatedEdges;
      });

      // Send to API (executed asynchronously)
      if (autoSaveEnabled) {
        console.log('Calling createEdgeAPI...');
        createEdgeAPI(newEdge as Edge).then(() => {
          console.log('Edge creation API call completed');
        });
      } else {
        console.log('Auto-save disabled, skipping edge API call');
      }

      toast({
        title: "Connected",
        description: `Connected ${sourcePortName || 'output'} (${sourceType}) → ${targetPortName || 'input'} (${targetType})`,
        status: "success",
        duration: 2000,
        isClosable: true,
      });
    },
    [setSharedEdges, toast, autoSaveEnabled, createEdgeAPI],
  );

  // ReactFlow onNodeClick
  const onNodeClick: NodeMouseHandler<Node<CalculationNodeData>> = useCallback((event, node) => {
    // Only select the node (information display is available from the icon button)
    console.log("Node clicked:", node.id);
  }, []);

  // ReactFlow onNodeDragStop
  const onNodeDragStop = useCallback((event: any, node: any) => {
    console.log("Node Drag Stop:", selectedProject, node.id, node.position.x, node.position.y);

    debouncedSave(() => updateNodeAPI(node.id, node));
  }, [selectedProject, debouncedSave, updateNodeAPI]);

  // ReactFlow onEdgeClick
  const onEdgeClick: EdgeMouseHandler = useCallback((event, edge) => {
    event.preventDefault();

    setEdgeMenuPosition({
      x: event.clientX,
      y: event.clientY,
    });

    setSelectedEdgeId(edge.id);
  }, [setEdgeMenuPosition, setSelectedEdgeId]);

  // ReactFlow Page - fully controlled by Zustand store
  return (
    <ReactFlow
      style={{ position: 'absolute' }}
      nodes={sharedNodes}
      edges={sharedEdges}
      onNodesChange={handleNodesChange}
      onEdgesChange={handleEdgesChange}
      onConnect={onConnect}
      onInit={onInit}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onNodeClick={onNodeClick}
      onNodeDragStop={onNodeDragStop}
      onEdgeClick={onEdgeClick}
      onPaneClick={onPaneClick}
      onMoveEnd={onMoveEnd}
      defaultViewport={initialViewport}
      nodeTypes={nodeTypes}
      attributionPosition="bottom-right"
      connectionLineStyle={{ stroke: '#8b5cf6', strokeWidth: 2 }}
      defaultEdgeOptions={{
        style: { stroke: '#aaaaaa', strokeWidth: 2 },
        type: 'default',
      }}
      deleteKeyCode={null}
      selectionOnDrag
      multiSelectionKeyCode="Meta"
    >
      <Controls {...controlsStyle} />
      <MiniMap {...minimapStyle} />
      <Background variant={BackgroundVariant.Dots} gap={12} size={1} color="#cbd5e0" />
    </ReactFlow>
  );
};

//////////////////////////////////////////////////////////////
// ReactFlowProvider: Wrapping ReactFlow
//////////////////////////////////////////////////////////////
export const WorkflowCanvasProvider: React.FC<WorkflowCanvasProps> = (props) => {
  // ReactFlow Provider Page
  return (
    <ReactFlowProvider>
      {/* WorkflowCanvas: ReactFlow definition */}
      <WorkflowCanvas {...props} />
    </ReactFlowProvider>
  );
};

export default WorkflowCanvas;
