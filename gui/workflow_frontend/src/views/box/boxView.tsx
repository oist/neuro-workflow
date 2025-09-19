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
} from '@chakra-ui/react';
import { useEffect, useState, useRef } from 'react';
import { IconType } from 'react-icons';
import { FiBox, FiCopy, FiTrash2, FiInfo, FiCode, FiRefreshCw, FiChevronDown, FiChevronRight } from 'react-icons/fi'; // デフォルトアイコンとして使用
import { SchemaFields } from '../home/type';
import { createAuthHeaders } from '../../api/authHeaders';

interface SidebarProps {
  nodes: UploadedNodesResponse | null;
  isLoading?: boolean;
  error?: string;
  onRefresh?: () => Promise<void>;
  onNodeInfo?: (node: BackendNodeType) => void;
  onViewCode?: (node: BackendNodeType) => void;
}

// バックエンドのレスポンス型に合わせて定義
interface UploadedNodesResponse {
  nodes: BackendNodeType[];
  total_files: number;
  total_nodes: number;
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
}

interface NodeTypeWithIcon extends Omit<BackendNodeType, 'icon'> {
  icon: IconType;
}

// Jupyterを別タブで開く
const OpenJupyter = (filename : string, category : string) => {
    window.open("http://localhost:8000/user/user1/lab/workspaces/auto-E/tree/codes/nodes/"+category.toLowerCase()+"/"+filename+".py", "_blank");
};

const SideBoxArea: React.FC<SidebarProps> = ({ nodes, isLoading = false, error, onRefresh, onNodeInfo, onViewCode }) => {
  const [searchResult, setSearchResult] = useState<string>('');
  const [filteredNodes, setFilteredNodes] = useState<NodeTypeWithIcon[]>([]);
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [isCopying, setIsCopying] = useState<string | null>(null);
  const [copyFileName, setCopyFileName] = useState<string>('');
  const [nodeToAction, setNodeToAction] = useState<NodeTypeWithIcon | null>(null);
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({});
  const toast = useToast();

  // ダイアログ管理
  const { isOpen: isCopyModalOpen, onOpen: onCopyModalOpen, onClose: onCopyModalClose } = useDisclosure();
  const { isOpen: isDeleteAlertOpen, onOpen: onDeleteAlertOpen, onClose: onDeleteAlertClose } = useDisclosure();
  const cancelRef = useRef<HTMLButtonElement>(null);

  console.log("サイドボックスエリア", filteredNodes)
  
  useEffect(() => {
    if (nodes && nodes.nodes) {
      // バックエンドのノードにアイコンを追加
      const nodesWithIcons: NodeTypeWithIcon[] = nodes.nodes.map(node => ({
        ...node,
        icon: FiBox, // デフォルトアイコン（必要に応じて変更）
      }));
      setFilteredNodes(nodesWithIcons);
    } else {
      setFilteredNodes([]);
    }
  }, [nodes]);
  
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
          node.category.toLowerCase().includes(keyword.toLowerCase()) ||
          node.file_name.toLowerCase().includes(keyword.toLowerCase())
        )
        .map(node => ({
          ...node,
          icon: FiBox,
        }));
      setFilteredNodes(filtered);
    }
  };
  
  const onDragStart = (event: React.DragEvent, node: NodeTypeWithIcon) => {
    // バックエンドの詳細な情報もドラッグデータに含める
    const dragData = {
      type: node.type,
      label: node.label,
      file_id: node.file_id,
      class_name: node.class_name,
      file_name: node.file_name,
      schema: node.schema,
      description: node.description,
    };
    
    event.dataTransfer.setData('application/reactflow', node.type);
    event.dataTransfer.setData('application/nodedata', JSON.stringify(dragData));
    event.dataTransfer.effectAllowed = 'move';
  };

  // 削除確認ダイアログを開く
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

  // 削除実行
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

  // ノードデータ同期
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

  // コピーダイアログを開く
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
    // デフォルトファイル名を設定（拡張子を除去してコピー接尾辞を追加）
    const baseName = node.file_name.replace(/\.py$/, '');
    setCopyFileName(`${baseName}_copy`);
    onCopyModalOpen();
  };

  // コピー実行
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

  // カテゴリの折りたたみ状態をトグルする関数
  const toggleCategory = (category: string) => {
    setCollapsedCategories(prev => ({
      ...prev,
      [category]: !prev[category]
    }));
  };

  // 全カテゴリを展開/折りたたみする関数
  const toggleAllCategories = (collapsed: boolean) => {
    const newState = Object.keys(nodesByCategory).reduce((acc, category) => {
      acc[category] = collapsed;
      return acc;
    }, {} as Record<string, boolean>);
    setCollapsedCategories(newState);
  };
  
  return (
    <Box
      position="fixed"
      left={0}
      top="64px"
      height="calc(100vh - 64px)"
      width="320px"
      bg="gray.900"
      color="white"
      borderRight="1px solid"
      borderColor="gray.700"
      zIndex={10}
      display="flex"
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
          <Box position="sticky" top={0} bg="gray.900" pb={2} zIndex={1}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
              <Heading size="md">Node Library</Heading>
              <HStack spacing={2}>
                {nodes && (
                  <Text fontSize="xs" color="gray.400">
                    {nodes.total_nodes} nodes from {nodes.total_files} files
                  </Text>
                )}
                <Tooltip 
                  label="Node Refresh - Sync node data from server" 
                  hasArrow
                  placement="bottom"
                  bg="gray.800"
                  color="white"
                  fontSize="sm"
                >
                  <IconButton
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
            
            {/* 同期中のインディケーター */}
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
          
          <Divider borderColor="gray.700" />

          <HStack
            justify="space-between"
            align="center"
            mt={-4}
            mb={-3}
          >
            <Text
              fontSize="sm"
              fontWeight="bold"
              color="gray.400"
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
                    color="gray.400"
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
                    color="gray.400"
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
                <Text mt={4} color="gray.400">Loading nodes...</Text>
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
                color="gray.500"
              >
                <Text>No nodes available</Text>
                <Text fontSize="sm" mt={2}>Upload Python files to add custom nodes</Text>
              </Box>
            ) : (
              <>
                {Object.entries(nodesByCategory).length > 0 ? (
                  Object.entries(nodesByCategory).map(([category, categoryNodes]) => {
                    const isCollapsed = collapsedCategories[category];
                    return (
                      <Box key={category} mb={4}>
                        {/* カテゴリヘッダー */}
                        <Box
                          display="flex"
                          alignItems="center"
                          justifyContent="space-between"
                          p={2}
                          bg="gray.750"
                          borderRadius="md"
                          cursor="pointer"
                          _hover={{ bg: "gray.700" }}
                          onClick={() => toggleCategory(category)}
                          mb={2}
                        >
                          <HStack spacing={2}>
                            <Icon
                              as={isCollapsed ? FiChevronRight : FiChevronDown}
                              color="gray.400"
                              transition="transform 0.2s"
                            />
                            <Text
                              fontSize="sm"
                              fontWeight="bold"
                              color="gray.300"
                              textTransform="capitalize"
                            >
                              {category}
                            </Text>
                          </HStack>
                          <Badge
                            size="sm"
                            colorScheme="blue"
                            variant="subtle"
                            borderRadius="full"
                          >
                            {categoryNodes.length}
                          </Badge>
                        </Box>

                        {/* カテゴリ内のノード */}
                        <Collapse in={!isCollapsed} animateOpacity>
                          <SimpleGrid columns={1} spacing={2}>
                            {categoryNodes.map((node) => (
                          <Box
                            key={node.id}
                            bg="gray.800"
                            borderRadius="md"
                            border="1px solid"
                            borderColor="gray.700"
                            cursor="grab"
                            _hover={{
                              bg: "gray.700",
                              borderColor: "blue.500",
                              transform: "translateY(-2px)",
                              transition: "all 0.2s"
                            }}
                            onDragStart={(event) => onDragStart(event, node)}
                            draggable
                            overflow="hidden"
                          >
                            {/* ヘッダー部分 */}
                            <Box
                              bg="gray.750"
                              px={3}
                              borderBottom="1px solid"
                              borderColor="gray.600"
                              onDragStart={(e) => e.stopPropagation()}
                              onDrag={(e) => e.stopPropagation()}
                              draggable={false}
                            >
                              {/* アクションボタンエリア */}
                              <Box py={1} display="flex" justifyContent="flex-end">
                                <HStack spacing={1}>
                                  <IconButton
                                    aria-label="View source code"
                                    icon={<FiCode />}
                                    size="xs"
                                    variant="ghost"
                                    color="gray.400"
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
                                    aria-label="Node information"
                                    icon={<FiInfo />}
                                    size="xs"
                                    variant="ghost"
                                    color="gray.400"
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
                                    color="gray.400"
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
                                    color="gray.400"
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

                              {/* タイトルエリア */}
                              <Box pb={2}>
                                <HStack alignItems="center" spacing={2}>
                                  <Icon 
                                    as={node.icon} 
                                    boxSize={4} 
                                    color="blue.400"
                                  />
                                  <Text fontWeight="bold" fontSize="sm" color="white">
                                    {node.label}
                                  </Text>
                                </HStack>
                              </Box>
                            </Box>

                            {/* コンテンツ部分 */}
                            <Box p={3}>
                              <Text fontSize="xs" color="gray.500" mb={2}>
                                from {node.file_name}
                              </Text>
                              <Text fontSize="xs" color="gray.400" mb={2}>
                                {node.description}
                              </Text>
                              {node.schema && (Object.keys(node.schema.outputs).length  > 0 || Object.keys(node.schema.inputs).length  > 0) && (
                                <Box>
                                  <Text fontSize="xs" color="gray.500" mb={1}>
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
                    color="gray.500"
                  >
                    <Text>No nodes found matching "{searchResult}"</Text>
                  </Box>
                )}
              </>
            )}
          </Box>
        </VStack>
      </Box>

      {/* コピー用ファイル名入力モーダル */}
      <Modal isOpen={isCopyModalOpen} onClose={onCopyModalClose}>
        <ModalOverlay />
        <ModalContent bg="gray.800" color="white">
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
              bg="gray.700"
              border="1px solid"
              borderColor="gray.600"
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
            <Text fontSize="xs" color="gray.400" mt={2}>
              * .py extension will be added automatically
            </Text>
          </ModalBody>
          <ModalFooter>
            <Button 
              variant="ghost" 
              mr={3} 
              onClick={onCopyModalClose}
              color="gray.300"
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

      {/* 削除確認アラートダイアログ */}
      <AlertDialog
        isOpen={isDeleteAlertOpen}
        leastDestructiveRef={cancelRef}
        onClose={onDeleteAlertClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent bg="gray.800" color="white">
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Node
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure you want to delete "{nodeToAction?.label}"?
              <br />
              <Text fontSize="sm" color="gray.400" mt={2}>
                This action cannot be undone.
              </Text>
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button 
                ref={cancelRef} 
                onClick={onDeleteAlertClose}
                variant="ghost"
                color="gray.300"
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
  );
};

export default SideBoxArea;
