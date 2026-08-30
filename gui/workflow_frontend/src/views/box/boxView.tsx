import KeywordSearch from '@/shared/keyWordSearch/keyWordSearch';
import {
  VStack,
  Box,
  Text,
  SimpleGrid,
  Icon,
  Heading,
  Divider,
  Spinner,
  Alert,
  AlertIcon,
  IconButton,
  HStack,
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Input,
  Button,
  useDisclosure,
  AlertDialog,
  AlertDialogOverlay,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogBody,
  AlertDialogFooter,
  Tooltip,
  Collapse,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import { useEffect, useState, useRef } from 'react';
import { IconType } from 'react-icons';
import { FiBox, FiCopy, FiTrash2, FiEdit2, FiCode, FiRefreshCw, FiChevronDown, FiChevronRight, FiMenu } from 'react-icons/fi'; // Use as default icon
import { SchemaFields } from '../home/type';
import { createAuthHeaders } from '../../api/authHeaders';
import { openJupyterTree } from '../../api/jupyterTenant';
import { useTabContext } from '../../components/tabs/TabManager';

interface SidebarProps {
  nodes: UploadedNodesResponse | null;
  isLoading?: boolean;
  error?: string;
  onRefresh?: () => Promise<void>;
  onNodeInfo?: (node: BackendNodeType) => void;
  onViewCode?: (node: BackendNodeType) => void;
  onChangeColor?: (category: string, color: string) => void;
}

// Defined according to the backend response type
interface UploadedNodesResponse {
  categories: any;
  nodes: BackendNodeType[];
  total_files: number;
  total_nodes: number;
  is_node_reviewer?: boolean;
  tenant?: string;
}

interface BackendNodeType {
  id: string;
  type: string;
  label: string;
  description: string;
  category: string;
  file_id: string;
  class_name: string;
  file_name: string;
  schema: SchemaFields;
  color: string;
  status?: string;
  tenant?: string;
  can_submit?: boolean;
}

interface NodeTypeWithIcon extends Omit<BackendNodeType, 'icon'> {
  icon: IconType;
}

const SideBoxArea: React.FC<SidebarProps> = ({ nodes, isLoading = false, error, onRefresh, onNodeInfo, onViewCode, onChangeColor }) => {
  const [searchResult, setSearchResult] = useState<string>('');
  const [filteredNodes, setFilteredNodes] = useState<NodeTypeWithIcon[]>([]);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [isCopying, setIsCopying] = useState<string | null>(null);
  const [copyFileName, setCopyFileName] = useState<string>('');
  const [nodeToAction, setNodeToAction] = useState<NodeTypeWithIcon | null>(null);
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [isColorUpdating, setIsColorUpdating] = useState<boolean>(false);
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({});
  const toast = useToast();

  const [categoryColors, setCategoryColors] = useState<Record<string, string>>({});
  const [categoryColorKey, setCategoryColorKey] = useState('');
  const [categoryColorValue, setCategoryColorValue] = useState('');

  // Side menu opening/closing management
  const [isSideExpand, setIsSideExpand] = useState<boolean>(true);

  // Dialog management
  const { isOpen: isCopyModalOpen, onOpen: onCopyModalOpen, onClose: onCopyModalClose } = useDisclosure();
  const { isOpen: isDeleteAlertOpen, onOpen: onDeleteAlertOpen, onClose: onDeleteAlertClose } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement>(null);

  // Use the tab system context
  const { addJupyterTab } = useTabContext();

  const bg = useColorModeValue('white', 'gray.800');
  const panelBg = useColorModeValue('#f7f7f8', 'gray.900');
  const borderColor = useColorModeValue('#e5e5e5', 'gray.700');
  const textColor = useColorModeValue('#1a1a1a', 'white');
  const subtextColor = useColorModeValue('gray.500', 'gray.400');
  const hoverBg = useColorModeValue('#f5f5f5', 'gray.700');
  const inputBg = useColorModeValue('white', 'gray.700');
  const inputBorder = useColorModeValue('#e5e5e5', 'gray.600');
  // gray.750 does not exist — map to gray.700 (dark) / #ebebeb (light)
  const categoryHeaderBg = useColorModeValue('#ebebeb', 'gray.700');
  const nodeCardBg = useColorModeValue('white', 'gray.800');
  const nodeCardBorder = useColorModeValue('#e5e5e5', 'gray.700');
  const nodeHeaderBg = useColorModeValue('#f5f5f5', 'gray.700');
  const nodeHeaderBorder = useColorModeValue('#e5e5e5', 'gray.100');
  const tooltipBg = useColorModeValue('gray.700', 'gray.800');
  const menuButtonBg = useColorModeValue('gray.200', 'gray.200');

  //console.log("side box area", filteredNodes)

  useEffect(() => {
    if (nodes && nodes.nodes) {
      // Add icons to backend nodes
      const nodesWithIcons: NodeTypeWithIcon[] = nodes.nodes.map(node => ({
        ...node,
        icon: FiBox, // Default Icon (change as needed)
      }));
      setFilteredNodes(nodesWithIcons);
      initCategoryColors();
    } else {
      setFilteredNodes([]);
    }
  }, [nodes]);

  // Getting values from the color picker and updating the state
  const handleColorChange = async (selectedCategory: string, colorValue: string) => {
    if (!nodes?.categories?.[selectedCategory]) {
      return;
    }

    setCategoryColors(prevCategoryColors => ({
      ...prevCategoryColors,
      [selectedCategory]: colorValue,
    }));
    setCategoryColorKey(selectedCategory);
    setCategoryColorValue(colorValue);

    const updated = await updateCategoryColorAPI(selectedCategory, colorValue);
    if (updated) {
      onChangeColor?.(selectedCategory, colorValue);
    }
  };

  // Initialize category colors
  const initCategoryColors = () => {
    if(nodes?.categories){
      const initialColors: Record<string, string> = {};
      for (const key in nodes.categories) {
        initialColors[key] = nodes.categories[key].color;
      }
      setCategoryColors(initialColors);
    }
  };

  const handleSearch = (keyword: string) => {
    console.log('Searching for:', keyword);
    setSearchResult(keyword);

    if (!nodes || !nodes.nodes) {
      setFilteredNodes([]);
      return;
    }

    if (keyword.trim() === '') {
      const nodesWithIcons: NodeTypeWithIcon[] = nodes.nodes.map(node => ({
        ...node,
        icon: FiBox,
      }));
      setFilteredNodes(nodesWithIcons);
    } else {
      const filtered = nodes.nodes
        .filter(node =>
          node.label.toLowerCase().includes(keyword.toLowerCase()) ||
          node.description.toLowerCase().includes(keyword.toLowerCase()) ||
          //node.category.toLowerCase().includes(keyword.toLowerCase()) ||
          node.file_name.toLowerCase().includes(keyword.toLowerCase())
        )
        .map(node => ({
          ...node,
          icon: FiBox,
        }));
      setFilteredNodes(filtered);
    }
  };

  const onDragStart = (event: React.DragEvent, node: NodeTypeWithIcon, categoryColors: {}) => {
    // Include detailed backend information in drag data
    const dragData = {
      type: node.type,
      label: node.label,
      file_id: node.file_id,
      class_name: node.class_name,
      file_name: node.file_name,
      schema: node.schema,
      description: node.description,
      color: categoryColors[node.category.toLocaleLowerCase().replace("/","")],
    };

    event.dataTransfer.setData('application/reactflow', node.type);
    event.dataTransfer.setData('application/nodedata', JSON.stringify(dragData));
    event.dataTransfer.effectAllowed = 'move';
  };

  // Opens a delete confirmation dialog
  const openDeleteDialog = (node: NodeTypeWithIcon) => {
    if (!node.file_id) {
      toast({
        title: "Error",
        description: "No file ID available for deletion",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setNodeToAction(node);
    onDeleteAlertOpen();
  };

  // Delete execution on node list
  const handleDeleteNode = async () => {
    if (!nodeToAction) return;

    setIsDeleting(nodeToAction.file_id);

    try {
      const headers = await createAuthHeaders();
      const response = await fetch(`/api/box/files/${nodeToAction.file_id}/`, {
        method: 'DELETE',
        credentials: 'include',
        headers: {
          ...headers,
        },
      });

      if (response.ok) {
        toast({
          title: "Deleted",
          description: `Node "${nodeToAction.label}" deleted successfully`,
          status: "success",
          duration: 3000,
          isClosable: true,
        });

        onDeleteAlertClose();
        // Refresh the nodes list
        if (onRefresh) {
          await onRefresh();
        }
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to delete node');
      }
    } catch (error) {
      console.error('Error deleting node:', error);
      toast({
        title: "Error",
        description: `Failed to delete node: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsDeleting(null);
    }
  };

  // Node data synchronization
  const handleSyncNodes = async () => {
    setIsSyncing(true);

    try {
      const headers = await createAuthHeaders();
      const response = await fetch('/api/box/sync/', {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...headers,
        },
      });

      if (response.ok) {
        const result = await response.json();
        toast({
          title: "Sync Completed",
          description: result.message || "Node data synchronized successfully",
          status: "success",
          duration: 4000,
          isClosable: true,
        });

        // Refresh the nodes list after sync
        if (onRefresh) {
          await onRefresh();
        }
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to sync node data');
      }
    } catch (error) {
      console.error('Sync error:', error);
      toast({
        title: "Sync Failed",
        description: `Failed to sync node data: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSyncing(false);
    }
  };

  // Update category color
  const updateCategoryColorAPI = async (categoryName: string, categoryColor: string): Promise<boolean> => {
    if (!categoryName || !categoryColor) {
      console.log('Skipping category color update API call:', { categoryName, categoryColor });
      return false;
    }

    // Send category color updates to the server
    console.log('Updating category color via API:', { categoryName, categoryColor });

    try {
      const headers = await createAuthHeaders();
      const requestBody = {
        category: categoryName,
        color: categoryColor,
      };

      console.log('Update request body:', requestBody);

      const response = await fetch(`/api/box/categories/`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const responseData = await response.json();
      console.log('Update category color response:', responseData);

      if (!response.ok) {
        setIsColorUpdating(false);
        throw new Error(`HTTP ${response.status}: ${responseData.error || 'Failed to update category color'}`);
      }

      setIsColorUpdating(true);
      return true;
    } catch (error) {
      console.error('Error updating category color:', error);
      setIsColorUpdating(false);
      toast({
        title: "Save Error",
        description: `Failed to update category color: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 2000,
        isClosable: true,
      });
      return false;
    }
  };

  const postNodeGovernance = async (fileId: string, action: "submit" | "approve" | "reject") => {
    const headers = await createAuthHeaders();
    const body = action === "approve" ? { make_public: true } : {};
    const response = await fetch(`/api/box/files/${fileId}/${action}/`, {
      method: "POST",
      credentials: "include",
      headers,
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${response.status}`);
    }
    await onRefresh?.();
  };

  const statusColor = (status?: string) => {
    if (status === "public") return "green";
    if (status === "submitted") return "orange";
    if (status === "approved") return "blue";
    return "gray";
  };

  // Open copy dialog
  const openCopyDialog = (node: NodeTypeWithIcon) => {
    if (!node.file_name) {
      toast({
        title: "Error",
        description: "No file name available for copying",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setNodeToAction(node);
    // Set default filename (remove extension and add copy suffix)
    const baseName = node.file_name.replace(/\.py$/, '');
    setCopyFileName(`${baseName}_copy`);
    onCopyModalOpen();
  };

  // Copy execution
  const handleCopyNode = async () => {
    if (!nodeToAction || !copyFileName.trim()) {
      toast({
        title: "Error",
        description: "File name is required",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsCopying(nodeToAction.file_id);

    try {
      const headers = await createAuthHeaders();
      const requestBody = {
        source_filename: nodeToAction.file_name,
        target_filename: copyFileName.trim()
      };

      console.log('Copy request details:', {
        url: '/api/box/copy/',
        method: 'POST',
        body: requestBody,
      });

      const response = await fetch('/api/box/copy/', {
        method: 'POST',
        credentials: 'include',
        headers: {
          ...headers,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      console.log('Copy response status:', response.status);

      if (response.ok) {
        const responseData = await response.json();
        toast({
          title: "Copied",
          description: `Node "${nodeToAction.label}" copied as "${copyFileName}"`,
          status: "success",
          duration: 3000,
          isClosable: true,
        });

        console.log('Copy response data:', responseData);
        onCopyModalClose();
        // Refresh the nodes list to show the new copied node
        if (onRefresh) {
          await onRefresh();
        }
      } else {
        let errorData;
        const contentType = response.headers.get('content-type');

        if (contentType && contentType.includes('application/json')) {
          errorData = await response.json();
          console.error('Copy error (JSON):', errorData);
        } else {
          const textData = await response.text();
          console.error('Copy error (Text):', textData);
          errorData = { error: `HTTP ${response.status}: ${textData || 'Failed to copy node'}` };
        }

        throw new Error(errorData.error || `HTTP ${response.status}: Failed to copy node`);
      }
    } catch (error) {
      console.error('Error copying node:', error);
      toast({
        title: "Error",
        description: `Failed to copy node: ${error instanceof Error ? error.message : 'Unknown error'}`,
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsCopying(null);
    }
  };

  const nodesByCategory = filteredNodes.reduce((acc, node) => {
    if (!acc[node.category]) {
      acc[node.category] = [];
    }
    acc[node.category].push(node);
    return acc;
  }, {} as Record<string, NodeTypeWithIcon[]>);

  // A function to toggle the collapsed state of a category
  const toggleCategory = (category: string) => {
    setCollapsedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  // Function to expand/collapse all categories
  const toggleAllCategories = (collapsed: boolean) => {
    const newState = Object.keys(nodesByCategory).reduce((acc, category) => {
      acc[category] = collapsed;
      return acc;
    }, {} as Record<string, boolean>);
    setCollapsedCategories(newState);
  };

  // Open Jupyter in a new tab
  const OpenJupyter = async (filename : string, category : string) => {
    const chkPy = filename.includes(".py");
    if (!chkPy) {
      filename += ".py";
    }
    try {
      const jupyterUrl = await openJupyterTree(
        "codes/nodes/"+category.replace('/','').toLowerCase()+"/"+filename
      );

      let projectId = localStorage.getItem('projectId');
      projectId = projectId ? projectId : "";
      addJupyterTab(projectId, filename, jupyterUrl);
    } catch (err) {
      toast({
        title: "Could not open Jupyter",
        description: err instanceof Error ? err.message : "Failed to resolve the Jupyter URL",
        status: "error",
        duration: 4000,
        isClosable: true,
      });
    }
  };

  return (
    <Box
        position="absolute"
        top="268px"
        left="8px"
        bottom={0}
    >
      <IconButton
        position="absolute"
        top="12px"
        left="32px"
        zIndex={1000}
        aria-label="メニュー開閉"
        icon={<FiMenu />}
        onClick={() => setIsSideExpand(!isSideExpand)}
        colorScheme="gray"
        bg={menuButtonBg}
        _hover={{ bg: hoverBg }}
      />
      <Box
        position="absolute"
        left={0}
        top="8px"
        bottom="16px"
        width="320px"
        marginLeft="16px"
        // The width changes depending on isOpen. Smooth transition
        // width={isSideExpand ? '320px' : '8px'}
        display={isSideExpand ? 'block' : 'none'}
        transition="width 200ms ease"
        bg={panelBg}
        color={textColor}
        borderRight="1px solid"
        borderColor={borderColor}
        zIndex={10}
        flex="1"
        flexDirection="column"
      >
        <Box
          p={4}
          overflowY="auto"
          height="100%"
          css={{
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              width: '8px',
              background: '#2D3748',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb': {
              background: '#4A5568',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb:hover': {
              background: '#718096',
            },
          }}
        >
          <VStack spacing={6} align="stretch">
            <Box position="sticky" top={0} bg={panelBg} pb={2} zIndex={1}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={4} paddingBottom={2}>
                <HStack spacing={2}>
                  {nodes && (
                    <Text fontSize="xs" color={subtextColor} paddingLeft={16}>
                      {nodes.total_nodes} nodes from {nodes.total_files} files
                    </Text>
                  )}
                  <Tooltip
                    label="Node Refresh - Sync node data from server"
                    hasArrow
                    placement="bottom"
                    bg={tooltipBg}
                    color="white"
                    fontSize="sm"
                  >
                    <IconButton
                      position="absolute"
                      right="0px"
                      aria-label="Sync node data"
                      icon={<Icon as={FiRefreshCw} />}
                      size="sm"
                      colorScheme="blue"
                      variant="ghost"
                      isLoading={isSyncing}
                      onClick={handleSyncNodes}
                      _hover={{
                        bg: "blue.600",
                        color: "white"
                      }}
                      _active={{
                        bg: "blue.700"
                      }}
                      disabled={isSyncing}
                    />
                  </Tooltip>
                </HStack>
              </Box>
              <KeywordSearch
                onSearch={handleSearch}
                placeholder="Search nodes..."
                size="md"
                width="100%"
              />

              {/* Syncing indicator */}
              {isSyncing && (
                <Box
                  mt={2}
                  p={2}
                  bg="blue.900"
                  borderRadius="md"
                  border="1px solid"
                  borderColor="blue.700"
                >
                  <HStack spacing={2}>
                    <Spinner size="sm" color="blue.300" />
                    <Text fontSize="sm" color="blue.200">
                      Syncing node data... This may take a moment.
                    </Text>
                  </HStack>
                </Box>
              )}
            </Box>

            <Divider borderColor={borderColor} />

            <HStack
              justify="space-between"
              align="center"
              mt={-4}
              mb={-3}
            >
              <Text
                fontSize="sm"
                fontWeight="bold"
                color={subtextColor}
                textTransform="uppercase"
                letterSpacing="wider"
              >
                Nodes
              </Text>
              {Object.keys(nodesByCategory).length > 1 && (
                <HStack spacing={1}>
                  <Tooltip label="Expand all categories" hasArrow>
                    <IconButton
                      aria-label="Expand all"
                      icon={<FiChevronDown />}
                      size="xs"
                      variant="ghost"
                      color={subtextColor}
                      _hover={{ color: "blue.300" }}
                      onClick={() => toggleAllCategories(false)}
                    />
                  </Tooltip>
                  <Tooltip label="Collapse all categories" hasArrow>
                    <IconButton
                      aria-label="Collapse all"
                      icon={<FiChevronRight />}
                      size="xs"
                      variant="ghost"
                      color={subtextColor}
                      _hover={{ color: "blue.300" }}
                      onClick={() => toggleAllCategories(true)}
                    />
                  </Tooltip>
                </HStack>
              )}
            </HStack>

            <Box>
              {isLoading ? (
                <Box
                  textAlign="center"
                  py={8}
                >
                  <Spinner color="blue.400" size="lg" />
                  <Text mt={4} color={subtextColor}>Loading nodes...</Text>
                </Box>
              ) : error ? (
                <Alert status="error" bg="red.900" borderColor="red.700">
                  <AlertIcon />
                  <Text fontSize="sm">{error}</Text>
                </Alert>
              ) : nodes === null || !nodes.nodes || nodes.nodes.length === 0 ? (
                <Box
                  textAlign="center"
                  py={8}
                  color={subtextColor}
                >
                  <Text>No nodes available</Text>
                  <Text fontSize="sm" mt={2}>Upload Python files to add custom nodes</Text>
                </Box>
              ) : (
                <>
                  {Object.entries(nodesByCategory).length > 0 ? (
                    Object.entries(nodesByCategory).map(([category, categoryNodes]) => {
                      const isCollapsed = collapsedCategories[category];
                      const lower_category = category.toLowerCase().replace("/","");
                      return (
                        <Box key={category} mb={4}>
                          {/* category header */}
                          <Box
                            display="flex"
                            alignItems="center"
                            justifyContent="space-between"
                            p={2}
                            bg={categoryHeaderBg}
                            borderRadius="md"
                            cursor="pointer"
                            _hover={{ bg: hoverBg }}
                            onClick={() => toggleCategory(category)}
                            mb={2}
                          >
                            <HStack spacing={2}>
                              <Icon
                                as={isCollapsed ? FiChevronRight : FiChevronDown}
                                color={subtextColor}
                                transition="transform 0.2s"
                              />
                              <Text
                                fontSize="sm"
                                fontWeight="bold"
                                color={textColor}
                                textTransform="capitalize"
                              >
                                {category}
                              </Text>
                            </HStack>
                            <HStack>
                              <input
                                type="color"
                                value={categoryColors[lower_category]}
                                onChange={(event) => handleColorChange(lower_category, event.target.value)}
                                style={{ width: '24px', height: '24px', padding: '0', border: 'none', background: '#171923' }}
                              />
                              <Badge
                                size="sm"
                                colorScheme="blue"
                                variant="subtle"
                                borderRadius="full"
                              >
                                {categoryNodes.length}
                              </Badge>
                            </HStack>
                          </Box>
                          {/* Node in category */}
                          <Collapse in={!isCollapsed} animateOpacity>
                            <SimpleGrid columns={1} spacing={2}>
                              {categoryNodes.map((node) => (
                            <Box
                              key={node.id}
                              bg={nodeCardBg}
                              borderRadius="md"
                              border="1px solid"
                              borderColor={nodeCardBorder}
                              cursor="grab"
                              _hover={{
                                bg: hoverBg,
                                borderColor: "blue.500",
                                transform: "translateY(-2px)",
                                transition: "all 0.2s"
                              }}
                              onDragStart={(event) => onDragStart(event, node, categoryColors)}
                              draggable
                              overflow="hidden"
                            >
                              {/* header part */}
                              <Box
                                bg={nodeHeaderBg}
                                px={3}
                                borderBottom="1px solid"
                                borderColor={nodeHeaderBorder}
                                onDragStart={(e) => e.stopPropagation()}
                                onDrag={(e) => e.stopPropagation()}
                                draggable={false}
                              >
                                {/* action button area */}
                                <Box py={1} display="flex" justifyContent="flex-end">
                                  <HStack spacing={1}>
                                    <IconButton
                                      aria-label="View source code"
                                      icon={<FiCode />}
                                      size="xs"
                                      variant="ghost"
                                      color={subtextColor}
                                      _hover={{
                                        color: "purple.300",
                                        bg: "purple.700"
                                      }}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                        //onViewCode?.(node);
                                        OpenJupyter(node.file_name, node.category);
                                      }}
                                      onMouseDown={(e) => {
                                        e.stopPropagation();
                                      }}
                                      onDragStart={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                      }}
                                      draggable={false}
                                    />
                                    <IconButton
                                      aria-label="Edit node"
                                      icon={<FiEdit2 />}
                                      size="xs"
                                      variant="ghost"
                                      color={subtextColor}
                                      _hover={{
                                        color: "green.300",
                                        bg: "green.700"
                                      }}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                        onNodeInfo?.(node);
                                      }}
                                      onMouseDown={(e) => {
                                        e.stopPropagation();
                                      }}
                                      onDragStart={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                      }}
                                      draggable={false}
                                    />
                                    <IconButton
                                      aria-label="Copy node"
                                      icon={<FiCopy />}
                                      size="xs"
                                      variant="ghost"
                                      color={subtextColor}
                                      _hover={{
                                        color: "blue.300",
                                        bg: "blue.700"
                                      }}
                                      isLoading={isCopying === node.file_id}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                        openCopyDialog(node);
                                      }}
                                      onMouseDown={(e) => {
                                        e.stopPropagation();
                                      }}
                                      onDragStart={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                      }}
                                      draggable={false}
                                    />
                                    <IconButton
                                      aria-label="Delete node"
                                      icon={<FiTrash2 />}
                                      size="xs"
                                      variant="ghost"
                                      color={subtextColor}
                                      _hover={{
                                        color: "red.300",
                                        bg: "red.700"
                                      }}
                                      isLoading={isDeleting === node.file_id}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                        openDeleteDialog(node);
                                      }}
                                      onMouseDown={(e) => {
                                        e.stopPropagation();
                                      }}
                                      onDragStart={(e) => {
                                        e.stopPropagation();
                                        e.preventDefault();
                                      }}
                                      draggable={false}
                                    />
                                  </HStack>
                                </Box>

                                {/* title area */}
                                <Box pb={2}>
                                  <HStack alignItems="center" spacing={2}>
                                    <Icon
                                      as={node.icon}
                                      boxSize={4}
                                      color="blue.400"
                                    />
                                    <Text fontWeight="bold" fontSize="sm" color={textColor}>
                                      {node.label}
                                    </Text>
                                    {node.status && node.status !== "public" && (
                                      <Badge size="sm" colorScheme={statusColor(node.status)}>
                                        {node.status}
                                      </Badge>
                                    )}
                                  </HStack>
                                  {(node.can_submit || (nodes?.is_node_reviewer && node.status === "submitted")) && (
                                    <HStack spacing={1} mt={1}>
                                      {node.can_submit && (
                                        <Button
                                          size="xs"
                                          variant="outline"
                                          onClick={async (e) => {
                                            e.stopPropagation();
                                            try {
                                              await postNodeGovernance(node.file_id, "submit");
                                              toast({ title: "Submitted for review", status: "success", duration: 2000, isClosable: true });
                                            } catch (err) {
                                              toast({ title: "Submit failed", description: err instanceof Error ? err.message : "Unknown error", status: "error", duration: 3000, isClosable: true });
                                            }
                                          }}
                                        >
                                          Submit
                                        </Button>
                                      )}
                                      {nodes?.is_node_reviewer && node.status === "submitted" && (
                                        <>
                                          <Button
                                            size="xs"
                                            colorScheme="green"
                                            onClick={async (e) => {
                                              e.stopPropagation();
                                              try {
                                                await postNodeGovernance(node.file_id, "approve");
                                                toast({ title: "Approved and published", status: "success", duration: 2000, isClosable: true });
                                              } catch (err) {
                                                toast({ title: "Approve failed", description: err instanceof Error ? err.message : "Unknown error", status: "error", duration: 3000, isClosable: true });
                                              }
                                            }}
                                          >
                                            Approve
                                          </Button>
                                          <Button
                                            size="xs"
                                            colorScheme="red"
                                            variant="outline"
                                            onClick={async (e) => {
                                              e.stopPropagation();
                                              try {
                                                await postNodeGovernance(node.file_id, "reject");
                                                toast({ title: "Rejected", status: "info", duration: 2000, isClosable: true });
                                              } catch (err) {
                                                toast({ title: "Reject failed", description: err instanceof Error ? err.message : "Unknown error", status: "error", duration: 3000, isClosable: true });
                                              }
                                            }}
                                          >
                                            Reject
                                          </Button>
                                        </>
                                      )}
                                    </HStack>
                                  )}
                                </Box>
                              </Box>

                              {/* content part */}
                              <Box p={3}>
                                <Text fontSize="xs" color={subtextColor} mb={2}>
                                  from {node.file_name}
                                </Text>
                                <Text fontSize="xs" color={subtextColor} mb={2}>
                                  {node.description}
                                </Text>
                                {node.schema && (Object.keys(node.schema.outputs).length  > 0 || Object.keys(node.schema.inputs).length  > 0) && (
                                  <Box>
                                    <Text fontSize="xs" color={subtextColor} mb={1}>
                                      Ports: {Object.keys(node.schema.inputs).length}in / {Object.keys(node.schema.outputs).length}out
                                    </Text>
                                  </Box>
                                )}
                                </Box>
                              </Box>
                              ))}
                            </SimpleGrid>
                          </Collapse>
                        </Box>
                      );
                    })
                  ) : (
                    <Box
                      textAlign="center"
                      py={8}
                      color={subtextColor}
                    >
                      <Text>No nodes found matching "{searchResult}"</Text>
                    </Box>
                  )}
                </>
              )}
            </Box>
          </VStack>
        </Box>

        {/* Modal for inputting file name for copy */}
        <Modal isOpen={isCopyModalOpen} onClose={onCopyModalClose}>
          <ModalOverlay />
          <ModalContent bg={bg} color={textColor}>
            <ModalHeader>Copy Node</ModalHeader>
            <ModalCloseButton />
            <ModalBody>
              <Text mb={4}>
                Copy "{nodeToAction?.label}" as:
              </Text>
              <Input
                value={copyFileName}
                onChange={(e) => setCopyFileName(e.target.value)}
                placeholder="Enter new file name"
                bg={inputBg}
                border="1px solid"
                borderColor={inputBorder}
                _focus={{
                  borderColor: "blue.400",
                  boxShadow: "0 0 0 1px #63b3ed",
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleCopyNode();
                  }
                }}
              />
              <Text fontSize="xs" color={subtextColor} mt={2}>
                * .py extension will be added automatically
              </Text>
            </ModalBody>
            <ModalFooter>
              <Button
                variant="ghost"
                mr={3}
                onClick={onCopyModalClose}
                color={subtextColor}
              >
                Cancel
              </Button>
              <Button
                colorScheme="blue"
                onClick={handleCopyNode}
                isLoading={isCopying === nodeToAction?.file_id}
                isDisabled={!copyFileName.trim()}
              >
                Copy
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>

        {/* Delete confirmation alert dialog */}
        <AlertDialog
          isOpen={isDeleteAlertOpen}
          leastDestructiveRef={cancelRef}
          onClose={onDeleteAlertClose}
        >
          <AlertDialogOverlay>
            <AlertDialogContent bg={bg} color={textColor}>
              <AlertDialogHeader fontSize="lg" fontWeight="bold">
                Delete Node
              </AlertDialogHeader>

              <AlertDialogBody>
                Are you sure you want to delete "{nodeToAction?.label}"?
                <br />
                <Text fontSize="sm" color={subtextColor} mt={2}>
                  This action cannot be undone.
                </Text>
              </AlertDialogBody>

              <AlertDialogFooter>
                <Button
                  ref={cancelRef}
                  onClick={onDeleteAlertClose}
                  variant="ghost"
                  color={subtextColor}
                >
                  Cancel
                </Button>
                <Button
                  colorScheme="red"
                  onClick={handleDeleteNode}
                  ml={3}
                  isLoading={isDeleting === nodeToAction?.file_id}
                >
                  Delete
                </Button>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogOverlay>
        </AlertDialog>
      </Box>
    </Box>
  );
};

export default SideBoxArea;
