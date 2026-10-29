import { useCallback, useEffect, useRef, useState } from "react";
import {
  Box,
  Button,
  Flex,
  HStack,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Spinner,
  Text,
  VStack,
  useColorModeValue,
  useToast,
} from "@chakra-ui/react";
import {
  deleteNotebookConversation,
  getNotebookConversation,
  listNotebookConversations,
  type NotebookConversation,
  type NotebookConversationSummary,
} from "@/api/chatApi";
import ChatMessageBubble from "./ChatMessageBubble";
import ToolCallDisplay from "./ToolCallDisplay";
import {
  buildTranscriptItems,
  downloadText,
  transcriptFilename,
  transcriptToJSON,
  transcriptToMarkdown,
} from "./notebookTranscript";

interface NotebookChatsModalProps {
  isOpen: boolean;
  onClose: () => void;
  projectId: string | null;
}

// Read-only browser of the conversations the in-notebook agent saved for a
// project (Issue #122).
export default function NotebookChatsModal({
  isOpen,
  onClose,
  projectId,
}: NotebookChatsModalProps) {
  const toast = useToast();
  const [conversations, setConversations] = useState<
    NotebookConversationSummary[]
  >([]);
  const [selected, setSelected] = useState<NotebookConversation | null>(null);
  const [isLoadingList, setIsLoadingList] = useState(false);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  // Bumped to drop responses the user has moved on from: another conversation
  // picked, the modal closed or the project changed.
  const listRequest = useRef(0);
  const detailRequest = useRef(0);

  const listBg = useColorModeValue("gray.50", "gray.900");
  const activeBg = useColorModeValue("purple.50", "gray.700");
  const hoverBg = useColorModeValue("gray.100", "gray.800");
  const borderColor = useColorModeValue("gray.200", "gray.600");
  const subtextColor = useColorModeValue("gray.500", "gray.400");

  const showError = useCallback(
    (title: string, error: unknown) =>
      toast({
        title,
        description: error instanceof Error ? error.message : String(error),
        status: "error",
        duration: 5000,
        isClosable: true,
      }),
    [toast]
  );

  const loadList = useCallback(async () => {
    if (!projectId) return;
    const request = ++listRequest.current;
    setIsLoadingList(true);
    try {
      const rows = await listNotebookConversations(projectId);
      if (request === listRequest.current) setConversations(rows);
    } catch (error) {
      if (request === listRequest.current)
        showError("Could not load notebook chats", error);
    } finally {
      if (request === listRequest.current) setIsLoadingList(false);
    }
  }, [projectId, showError]);

  useEffect(() => {
    setSelected(null);
    setConversations([]);
    if (isOpen) loadList();
    return () => {
      listRequest.current += 1;
      detailRequest.current += 1;
      setIsLoadingList(false);
      setIsLoadingConversation(false);
    };
  }, [isOpen, loadList]);

  const handleSelect = async (id: string) => {
    const request = ++detailRequest.current;
    setIsLoadingConversation(true);
    try {
      const conversation = await getNotebookConversation(id);
      if (request === detailRequest.current) setSelected(conversation);
    } catch (error) {
      if (request === detailRequest.current)
        showError("Could not open the conversation", error);
    } finally {
      if (request === detailRequest.current) setIsLoadingConversation(false);
    }
  };

  const handleDelete = async () => {
    if (!selected) return;
    if (!window.confirm(`Delete "${selected.title}"?`)) return;
    const request = detailRequest.current;
    try {
      await deleteNotebookConversation(selected.id);
      if (request !== detailRequest.current) return;
      setSelected(null);
      await loadList();
    } catch (error) {
      if (request === detailRequest.current)
        showError("Could not delete the conversation", error);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="6xl" scrollBehavior="inside">
      <ModalOverlay />
      <ModalContent h="80vh">
        <ModalHeader>Notebook chats</ModalHeader>
        <ModalCloseButton />
        <ModalBody display="flex" gap={4} overflow="hidden" pb={6}>
          {/* Conversation list */}
          <Box
            w="300px"
            flexShrink={0}
            overflowY="auto"
            bg={listBg}
            borderWidth={1}
            borderColor={borderColor}
            borderRadius="md"
          >
            {isLoadingList ? (
              <Flex justify="center" p={4}>
                <Spinner size="sm" />
              </Flex>
            ) : conversations.length === 0 ? (
              <Text fontSize="sm" color={subtextColor} p={4}>
                No saved conversations yet. Conversations with the notebook
                chat agent (%chat or ChatPanel) in this project's notebooks
                are saved here automatically.
              </Text>
            ) : (
              conversations.map((c) => (
                <Box
                  as="button"
                  type="button"
                  key={c.id}
                  display="block"
                  w="100%"
                  textAlign="left"
                  px={3}
                  py={2}
                  cursor="pointer"
                  aria-current={selected?.id === c.id ? "true" : undefined}
                  _focusVisible={{
                    outline: "2px solid",
                    outlineColor: "purple.400",
                    outlineOffset: "-2px",
                  }}
                  borderBottomWidth={1}
                  borderColor={borderColor}
                  bg={selected?.id === c.id ? activeBg : undefined}
                  _hover={{ bg: selected?.id === c.id ? activeBg : hoverBg }}
                  onClick={() => handleSelect(c.id)}
                >
                  <Text fontSize="sm" fontWeight="medium" noOfLines={2}>
                    {c.title}
                  </Text>
                  <Text fontSize="xs" color={subtextColor}>
                    {new Date(c.updated_at).toLocaleString()} ·{" "}
                    {c.message_count} messages
                  </Text>
                  <Text fontSize="xs" color={subtextColor} noOfLines={1}>
                    {[c.user_label, c.model].filter(Boolean).join(" · ")}
                  </Text>
                </Box>
              ))
            )}
          </Box>

          {/* Transcript */}
          <Flex flex={1} direction="column" minW={0}>
            {isLoadingConversation ? (
              <Flex justify="center" p={4}>
                <Spinner />
              </Flex>
            ) : !selected ? (
              <Text fontSize="sm" color={subtextColor} p={4}>
                Select a conversation to read it.
              </Text>
            ) : (
              <>
                <HStack justify="space-between" mb={3} spacing={2}>
                  <Box minW={0}>
                    <Text fontWeight="bold" noOfLines={1}>
                      {selected.title}
                    </Text>
                    <Text fontSize="xs" color={subtextColor}>
                      {[
                        selected.user_label,
                        selected.model,
                        new Date(selected.created_at).toLocaleString(),
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </Text>
                  </Box>
                  <HStack spacing={2} flexShrink={0}>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        downloadText(
                          transcriptFilename(selected, "md"),
                          transcriptToMarkdown(selected),
                          "text/markdown"
                        )
                      }
                    >
                      Download Markdown
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        downloadText(
                          transcriptFilename(selected, "json"),
                          transcriptToJSON(selected),
                          "application/json"
                        )
                      }
                    >
                      Download JSON
                    </Button>
                    {selected.can_delete && (
                      <Button
                        size="sm"
                        colorScheme="red"
                        variant="outline"
                        onClick={handleDelete}
                      >
                        Delete
                      </Button>
                    )}
                  </HStack>
                </HStack>
                <VStack
                  flex={1}
                  overflowY="auto"
                  align="stretch"
                  spacing={2}
                  pr={1}
                >
                  {buildTranscriptItems(selected.messages).map((item) =>
                    item.type === "message" ? (
                      <ChatMessageBubble key={item.data.id} message={item.data} />
                    ) : (
                      <ToolCallDisplay
                        key={item.data.tool_call_id}
                        toolCall={item.data}
                      />
                    )
                  )}
                </VStack>
              </>
            )}
          </Flex>
        </ModalBody>
      </ModalContent>
    </Modal>
  );
}
