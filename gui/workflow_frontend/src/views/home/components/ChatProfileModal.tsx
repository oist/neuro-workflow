import React, { useEffect, useMemo, useState } from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  Button,
  VStack,
  HStack,
  Text,
  Input,
  Textarea,
  FormControl,
  FormLabel,
  FormHelperText,
  useToast,
  Spinner,
  Alert,
  AlertIcon,
  Box,
  Checkbox,
  Code,
  Divider,
} from '@chakra-ui/react';
import {
  createChatProfile,
  updateChatProfile,
  listChatTools,
  type ChatProfile,
  type ChatTool,
} from '@/api/chatProfileApi';
import { groupToolsByCategory } from './chatToolCategories';

interface ChatProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
  profile?: ChatProfile | null; // If provided, edit mode; otherwise, create mode
  onSaved: () => void;
}

const firstLine = (text: string) => text.split('\n')[0].trim();

const ChatProfileModal: React.FC<ChatProfileModalProps> = ({
  isOpen,
  onClose,
  profile,
  onSaved,
}) => {
  const toast = useToast();
  const [name, setName] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [tools, setTools] = useState<ChatTool[] | null>(null);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [loadingTools, setLoadingTools] = useState(false);
  const [saving, setSaving] = useState(false);

  // Reset the form and (re)load the tool catalog every time the modal opens.
  useEffect(() => {
    if (!isOpen) return;
    setName(profile?.name ?? '');
    setSystemPrompt(profile?.system_prompt ?? '');
    setSelected(new Set(profile?.allowed_tools ?? []));
    setToolsError(null);
    setLoadingTools(true);
    let cancelled = false;
    listChatTools()
      .then((list) => {
        if (!cancelled) setTools(list);
      })
      .catch((err) => {
        if (cancelled) return;
        // Keep the picker hidden so the existing selection is preserved as is.
        setTools(null);
        setToolsError(err instanceof Error ? err.message : 'Failed to load tools');
      })
      .finally(() => {
        if (!cancelled) setLoadingTools(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, profile]);

  const groups = useMemo(() => groupToolsByCategory(tools ?? []), [tools]);
  const allToolNames = useMemo(() => (tools ?? []).map((t) => t.name), [tools]);

  const toggleMany = (names: string[], on: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev);
      names.forEach((n) => (on ? next.add(n) : next.delete(n)));
      return next;
    });
  };

  const handleSave = async () => {
    const trimmed = name.trim();
    if (!trimmed) {
      toast({ title: 'Name is required', status: 'warning', duration: 3000 });
      return;
    }
    setSaving(true);
    try {
      const payload = {
        name: trimmed,
        system_prompt: systemPrompt,
        allowed_tools: Array.from(selected),
      };
      if (profile) {
        await updateChatProfile(profile.id, payload);
      } else {
        await createChatProfile(payload);
      }
      toast({
        title: profile ? 'Profile updated' : 'Profile created',
        status: 'success',
        duration: 3000,
      });
      onSaved();
      onClose();
    } catch (err) {
      toast({
        title: 'Failed to save profile',
        description: err instanceof Error ? err.message : String(err),
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="xl" scrollBehavior="inside">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>{profile ? 'Edit Chat Profile' : 'New Chat Profile'}</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack spacing={4} align="stretch">
            <FormControl isRequired>
              <FormLabel>Name</FormLabel>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Viewer only"
              />
            </FormControl>

            <FormControl>
              <FormLabel>System prompt override</FormLabel>
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                rows={5}
                placeholder="Leave empty to use the default assistant prompt"
              />
              <FormHelperText>
                Leave empty to use the default assistant prompt.
              </FormHelperText>
            </FormControl>

            <Divider />

            <Box>
              <HStack justify="space-between" mb={2}>
                <Text fontWeight="semibold">
                  Allowed MCP tools{' '}
                  <Text as="span" fontWeight="normal" color="gray.500">
                    ({selected.size} selected)
                  </Text>
                </Text>
                {tools && (
                  <HStack spacing={2}>
                    <Button size="xs" variant="outline" onClick={() => toggleMany(allToolNames, true)}>
                      Select all
                    </Button>
                    <Button size="xs" variant="outline" onClick={() => toggleMany(allToolNames, false)}>
                      Select none
                    </Button>
                  </HStack>
                )}
              </HStack>

              {selected.size === 0 && (
                <Alert status="warning" mb={2} fontSize="sm" borderRadius="md">
                  <AlertIcon />
                  Tools will be disabled for this profile.
                </Alert>
              )}

              {loadingTools && (
                <HStack>
                  <Spinner size="sm" />
                  <Text fontSize="sm">Loading tools…</Text>
                </HStack>
              )}

              {toolsError && (
                <Alert status="error" fontSize="sm" borderRadius="md">
                  <AlertIcon />
                  Could not load the tool list ({toolsError}). The current
                  selection is kept as is.
                </Alert>
              )}

              {tools &&
                groups.map((group) => {
                  const names = group.tools.map((t) => t.name);
                  const checkedCount = names.filter((n) => selected.has(n)).length;
                  const all = checkedCount === names.length;
                  return (
                    <Box key={group.id} mb={3}>
                      <Checkbox
                        isChecked={all}
                        isIndeterminate={checkedCount > 0 && !all}
                        onChange={(e) => toggleMany(names, e.target.checked)}
                      >
                        <Text as="span" fontWeight="semibold">
                          {group.label}
                        </Text>{' '}
                        <Text as="span" fontSize="sm" color="gray.500">
                          ({checkedCount}/{names.length})
                        </Text>
                      </Checkbox>
                      <VStack align="stretch" spacing={1} pl={6} mt={1}>
                        {group.tools.map((tool) => (
                          <Checkbox
                            key={tool.name}
                            size="sm"
                            isChecked={selected.has(tool.name)}
                            onChange={(e) => toggleMany([tool.name], e.target.checked)}
                          >
                            <HStack spacing={2} align="baseline">
                              <Code fontSize="xs" flexShrink={0}>
                                {tool.name}
                              </Code>
                              {tool.description && (
                                <Text fontSize="xs" color="gray.500" noOfLines={1}>
                                  {firstLine(tool.description)}
                                </Text>
                              )}
                            </HStack>
                          </Checkbox>
                        ))}
                      </VStack>
                    </Box>
                  );
                })}
            </Box>
          </VStack>
        </ModalBody>

        <ModalFooter>
          <Button onClick={onClose} variant="ghost" mr={3}>
            Cancel
          </Button>
          <Button colorScheme="blue" onClick={handleSave} isLoading={saving}>
            {profile ? 'Save' : 'Create'}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ChatProfileModal;
