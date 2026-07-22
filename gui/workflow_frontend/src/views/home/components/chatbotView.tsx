import {
  Box,
  Flex,
  IconButton,
  Text,
  HStack,
  useDisclosure,
  useToast,
  useColorModeValue,
} from '@chakra-ui/react';
import { useEffect, useCallback } from 'react';
import { FiMenu, FiX, FiPlus, FiFileText } from 'react-icons/fi';
import { IoChatboxEllipses } from 'react-icons/io5';

import useChatStore from '@/stores/chatStore';
import { useFlowStore, PROJECT_ID_KEY } from '@/stores/flowStore';
import { useViewerStore, buildViewerContext } from '@/stores/viewerStore';
import { useTabContext } from '@/components/tabs/TabManager';
import {
  listConversations,
  getConversation,
  deleteConversation,
  sendMessageStream,
  type SSEEvent,
} from '@/api/chatApi';
import ChatMessageList from './ChatMessageList';
import ChatInput from './ChatInput';
import ConversationSelector from './ConversationSelector';

const SIDEBAR_WIDTH = '600px';
const TOGGLE_WIDTH = '16px';

const FLOW_MODIFYING_TOOLS = new Set([
  'add_node', 'update_node', 'delete_node',
  'update_node_parameter', 'update_node_instance_name',
  'add_edge', 'delete_edge', 'update_flow',
]);

// Group 5 viewer-control tools: their result is an {action, ...} dict we forward
// to the live 3D viewer iframe (mirrors the FLOW_MODIFYING_TOOLS pattern).
const VIEWER_ACTION_TOOLS = new Set([
  'viewer_highlight_region', 'viewer_focus_region', 'viewer_set_time_window',
  'viewer_show_trace', 'viewer_clear_selection',
]);

const ChatbotArea: React.FC = () => {
  const toast = useToast();
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: false });

  const bg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('#e5e5e5', 'gray.600');
  const textColor = useColorModeValue('#1a1a1a', 'white');
  const subtextColor = useColorModeValue('gray.500', 'gray.400');
  const hoverBg = useColorModeValue('#f5f5f5', 'gray.700');
  const toggleBg = useColorModeValue('#f5f5f5', 'gray.700');

  const conversations = useChatStore((s) => s.conversations);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const setConversations = useChatStore((s) => s.setConversations);
  const setActiveConversationId = useChatStore((s) => s.setActiveConversationId);
  const setMessages = useChatStore((s) => s.setMessages);
  const addMessage = useChatStore((s) => s.addMessage);
  const updateLastAssistantMessage = useChatStore((s) => s.updateLastAssistantMessage);
  const setIsStreaming = useChatStore((s) => s.setIsStreaming);
  const setAbortController = useChatStore((s) => s.setAbortController);
  const addToolCall = useChatStore((s) => s.addToolCall);
  const updateToolCallArgs = useChatStore((s) => s.updateToolCallArgs);
  const updateToolCallResult = useChatStore((s) => s.updateToolCallResult);
  const clearToolCalls = useChatStore((s) => s.clearToolCalls);
  const setError = useChatStore((s) => s.setError);
  const resetChat = useChatStore((s) => s.resetChat);

  const requestFlowRefresh = useFlowStore((s) => s.requestFlowRefresh);

  const { postToActiveViewer } = useTabContext();
  const getActiveSnapshot = useViewerStore((s) => s.getActiveSnapshot);

  const loadConversations = useCallback(async () => {
    try {
      const data = await listConversations();
      setConversations(data);
    } catch (err) {
      console.error('Failed to load conversations:', err);
    }
  }, [setConversations]);

  // Load conversations on mount & cleanup AbortController on unmount
  useEffect(() => {
    loadConversations();
    return () => {
      useChatStore.getState().abortController?.abort();
    };
  }, [loadConversations]);

  const handleSelectConversation = async (id: string) => {
    try {
      const data = await getConversation(id);
      setActiveConversationId(id);
      setMessages(
        (data.messages?.map((m: Record<string, unknown>) => ({
          id: m.id as string,
          role: m.role as string,
          content: m.content as string,
          tool_calls: m.tool_calls,
          tool_call_id: m.tool_call_id as string,
          tool_name: m.tool_name as string,
          created_at: m.created_at as string,
        })) || [])
      );
      clearToolCalls();
    } catch (err) {
      toast({
        title: 'Failed to load conversation',
        status: 'error',
        duration: 3000,
      });
    }
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      await deleteConversation(id);
      if (activeConversationId === id) {
        resetChat();
      }
      loadConversations();
    } catch (err) {
      toast({
        title: 'Failed to delete conversation',
        status: 'error',
        duration: 3000,
      });
    }
  };

  const handleNewConversation = () => {
    resetChat();
  };

  const handleSend = useCallback(
    async (message: string) => {
      if (isStreaming) return;

      // Add user message to UI immediately
      addMessage({
        id: `user-${Date.now()}`,
        role: 'user',
        content: message,
      });

      setIsStreaming(true);
      setError(null);
      clearToolCalls();

      const controller = new AbortController();
      setAbortController(controller);

      try {
        const currentProjectId = localStorage.getItem(PROJECT_ID_KEY);
        const viewerContext = buildViewerContext(getActiveSnapshot());
        await sendMessageStream(
          {
            message,
            conversation_id: activeConversationId,
            project_id: currentProjectId,
            viewer_context: viewerContext,
          },
          // onEvent
          (event: SSEEvent) => {
            switch (event.type) {
              case 'text_delta':
                updateLastAssistantMessage(
                  (event.data.content as string) || ''
                );
                break;

              case 'tool_call_start':
                addToolCall({
                  tool_call_id: (event.data.tool_call_id as string) || '',
                  tool_name: (event.data.tool_name as string) || '',
                  arguments: '',
                  status: 'running',
                });
                break;

              case 'tool_call_args_delta':
                updateToolCallArgs((event.data.content as string) || '');
                break;

              case 'tool_result': {
                const toolName = (event.data.tool_name as string) || '';
                const resultStr = (event.data.result as string) || '';
                updateToolCallResult(
                  (event.data.tool_call_id as string) || '',
                  resultStr,
                  'done'
                );
                if (FLOW_MODIFYING_TOOLS.has(toolName)) {
                  requestFlowRefresh();
                }
                // Forward a Group-5 viewer action to the live 3D viewer iframe.
                if (VIEWER_ACTION_TOOLS.has(toolName) && resultStr) {
                  try {
                    const action = JSON.parse(resultStr);
                    if (action && action.action) postToActiveViewer(action);
                  } catch {
                    // non-JSON result (e.g. error string) — nothing to forward
                  }
                }
                break;
              }

              case 'error':
                setError((event.data.message as string) || 'Unknown error');
                break;

              case 'done':
                break;
            }
          },
          // onConversationId
          (convId: string) => {
            setActiveConversationId(convId);
          },
          controller.signal
        );
      } catch (err: unknown) {
        if (err instanceof Error && err.name !== 'AbortError') {
          setError(err.message);
          toast({
            title: 'Chat error',
            description: err.message,
            status: 'error',
            duration: 5000,
          });
        }
      } finally {
        setIsStreaming(false);
        setAbortController(null);
        // Refresh conversation list
        loadConversations();
      }
    },
    [
      activeConversationId,
      isStreaming,
      addMessage,
      setIsStreaming,
      setError,
      clearToolCalls,
      setAbortController,
      updateLastAssistantMessage,
      addToolCall,
      updateToolCallArgs,
      updateToolCallResult,
      setActiveConversationId,
      loadConversations,
      requestFlowRefresh,
      postToActiveViewer,
      getActiveSnapshot,
      toast,
    ]
  );

  const handleStop = useCallback(() => {
    const controller = useChatStore.getState().abortController;
    controller?.abort();
    setIsStreaming(false);
    setAbortController(null);
  }, [setIsStreaming, setAbortController]);

  return (
    <Flex
      align="center"
      overflow="visible"
      position="absolute"
      top="72px"
      bottom="16px"
      left="8px"
      zIndex="1010"
      pointerEvents="none"
    >
      {/* Collapsed handle — the only thing left when folded, so the viewer
          underneath stays fully visible and clickable. */}
      <IconButton
        icon={<FiMenu />}
        onClick={onToggle}
        aria-label="Open Sidebar"
        display={isOpen ? 'none' : 'flex'}
        pointerEvents="auto"
        bg={toggleBg}
        color={textColor}
        width={TOGGLE_WIDTH}
        minW={TOGGLE_WIDTH}
        height="64px"
        borderLeftRadius={0}
        borderRightRadius="md"
        _hover={{ bg: 'blue.600', color: 'white' }}
      />

      <Box
        bg={bg}
        color={textColor}
        height="100%"
        width={SIDEBAR_WIDTH}
        display={isOpen ? 'block' : 'none'}
        pointerEvents="auto"
        overflow="hidden"
        position="relative"
        flexShrink={0}
      >
        <Flex
          direction="column"
          align="stretch"
          height="100%"
          minH={0}
          width={SIDEBAR_WIDTH}
        >
          {/* Close button */}
          <IconButton
            icon={<FiX />}
            onClick={onToggle}
            aria-label="Close Sidebar"
            position="absolute"
            top="50%"
            transform="translateY(-50%)"
            right="0"
            zIndex="1020"
            bg={toggleBg}
            color={textColor}
            width="12px"
            height="64px"
            _hover={{ bg: 'blue.600', color: 'white' }}
          />

          {/* Header */}
          <HStack
            px={3}
            py={2}
            borderBottom="1px solid"
            borderColor={borderColor}
            spacing={2}
            flexShrink={0}
          >
            <IoChatboxEllipses size={16} />
            <Text fontSize="sm" fontWeight="bold" flexShrink={0}>
              AI Assistant
            </Text>
            <ConversationSelector
              conversations={conversations}
              activeConversationId={activeConversationId}
              onSelect={handleSelectConversation}
              onDelete={handleDeleteConversation}
              onNew={handleNewConversation}
            />
            <IconButton
              icon={<FiFileText />}
              aria-label="Generate report"
              size="xs"
              variant="ghost"
              color={subtextColor}
              _hover={{ color: textColor, bg: hoverBg }}
              title="Generate Methods section report"
              onClick={() => handleSend(
                "Generate a scientific Methods section report for the active project. " +
                "Step 1: call get_workflow_facts to collect all parameters and results. " +
                "Step 2: write a Methods section as a neuroscientist would — flowing prose, no bullet points, " +
                "no mention of node names or workflow software internals; embed all parameter values in the text; " +
                "use sections: Abstract, Computational Model, Structural Connectivity, Simulation Setup, " +
                "Stimulation Protocol (if any), Data Recording and Output. " +
                "Step 3: call save_report to save it. " +
                "Step 4: reply with only a short confirmation — do not print the full report in chat."
              )}
            />
            <IconButton
              icon={<FiPlus />}
              aria-label="New conversation"
              size="xs"
              variant="ghost"
              color={subtextColor}
              _hover={{ color: textColor, bg: hoverBg }}
              onClick={handleNewConversation}
              ml="auto"
            />
          </HStack>

          {/* Message List */}
          <ChatMessageList />

          {/* Input */}
          <ChatInput
            onSend={handleSend}
            onStop={handleStop}
            isStreaming={isStreaming}
          />
        </Flex>
      </Box>
    </Flex>
  );
};

export default ChatbotArea;
