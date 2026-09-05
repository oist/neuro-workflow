import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  HStack,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Badge,
  IconButton,
  useToast,
  Spinner,
  Alert,
  AlertIcon,
  useDisclosure,
  TableContainer,
} from '@chakra-ui/react';
import { AddIcon, ArrowBackIcon, EditIcon, DeleteIcon } from '@chakra-ui/icons';
import { useNavigate } from 'react-router-dom';
import ChatProfileModal from './ChatProfileModal';
import { deleteChatProfile, type ChatProfile } from '@/api/chatProfileApi';
import { useChatProfileStore } from '@/stores/chatProfileStore';

// Settings page (/settings/chat-profiles) to manage the user's chat profiles.
const ChatProfileManager: React.FC = () => {
  const toast = useToast();
  const navigate = useNavigate();
  const profiles = useChatProfileStore((s) => s.profiles);
  const loadProfiles = useChatProfileStore((s) => s.loadProfiles);
  const selectedProfileId = useChatProfileStore((s) => s.selectedProfileId);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<ChatProfile | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();

  useEffect(() => {
    let cancelled = false;
    loadProfiles()
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load chat profiles');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [loadProfiles]);

  const refresh = () => {
    loadProfiles().catch((err) => {
      toast({
        title: 'Failed to reload profiles',
        description: err instanceof Error ? err.message : String(err),
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    });
  };

  const handleAdd = () => {
    setEditing(null);
    onOpen();
  };

  const handleEdit = (profile: ChatProfile) => {
    setEditing(profile);
    onOpen();
  };

  const handleDelete = async (profile: ChatProfile) => {
    if (!window.confirm(`Delete chat profile "${profile.name}"?`)) {
      return;
    }
    try {
      await deleteChatProfile(profile.id);
      toast({ title: 'Profile deleted', status: 'success', duration: 3000 });
      refresh();
    } catch (err) {
      toast({
        title: 'Failed to delete profile',
        description: err instanceof Error ? err.message : String(err),
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Box p={6} maxW="1000px" mx="auto">
      <Button
        leftIcon={<ArrowBackIcon />}
        variant="ghost"
        size="sm"
        mb={3}
        onClick={() => navigate('/')}
      >
        Back to Workflow
      </Button>
      <HStack justify="space-between" align="flex-start" mb={4}>
        <Box>
          <Text fontSize="xl" fontWeight="bold">
            Chat Profiles
          </Text>
          <Text fontSize="sm" color="gray.500">
            Choose which MCP tools the AI Assistant may use and optionally
            override its system prompt. Pick a profile from the chat header;
            &quot;Default&quot; keeps all tools and the default prompt.
          </Text>
        </Box>
        <Button leftIcon={<AddIcon />} colorScheme="blue" size="sm" onClick={handleAdd} flexShrink={0}>
          New Profile
        </Button>
      </HStack>

      {error && (
        <Alert status="error" mb={4} borderRadius="md">
          <AlertIcon />
          {error}
        </Alert>
      )}

      {loading ? (
        <Spinner />
      ) : profiles.length === 0 ? (
        <Box p={8} textAlign="center" borderWidth="1px" borderStyle="dashed" borderRadius="md">
          <Text color="gray.500">
            No chat profiles yet. Create one to restrict the assistant&apos;s tools
            or change its prompt.
          </Text>
        </Box>
      ) : (
        <TableContainer borderWidth="1px" borderRadius="md">
          <Table size="sm">
            <Thead>
              <Tr>
                <Th>Name</Th>
                <Th>Tools</Th>
                <Th>Prompt</Th>
                <Th isNumeric>Actions</Th>
              </Tr>
            </Thead>
            <Tbody>
              {profiles.map((profile) => (
                <Tr key={profile.id}>
                  <Td>
                    <HStack>
                      <Text fontWeight="medium">{profile.name}</Text>
                      {profile.id === selectedProfileId && (
                        <Badge colorScheme="blue">Selected</Badge>
                      )}
                    </HStack>
                  </Td>
                  <Td>
                    {profile.allowed_tools.length === 0 ? (
                      <Badge colorScheme="orange">None (tools disabled)</Badge>
                    ) : (
                      `${profile.allowed_tools.length} tools`
                    )}
                  </Td>
                  <Td>
                    <Badge colorScheme={profile.system_prompt ? 'purple' : 'gray'}>
                      {profile.system_prompt ? 'Custom' : 'Default'}
                    </Badge>
                  </Td>
                  <Td isNumeric>
                    <IconButton
                      aria-label="Edit profile"
                      icon={<EditIcon />}
                      size="sm"
                      variant="ghost"
                      onClick={() => handleEdit(profile)}
                    />
                    <IconButton
                      aria-label="Delete profile"
                      icon={<DeleteIcon />}
                      size="sm"
                      variant="ghost"
                      colorScheme="red"
                      onClick={() => handleDelete(profile)}
                    />
                  </Td>
                </Tr>
              ))}
            </Tbody>
          </Table>
        </TableContainer>
      )}

      <ChatProfileModal
        isOpen={isOpen}
        onClose={onClose}
        profile={editing}
        onSaved={refresh}
      />
    </Box>
  );
};

export default ChatProfileManager;
