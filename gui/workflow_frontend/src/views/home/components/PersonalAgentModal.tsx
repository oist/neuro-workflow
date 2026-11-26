import { useEffect, useRef, useState } from 'react';
import {
  Button,
  FormControl,
  FormLabel,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Text,
  useToast,
  VStack,
} from '@chakra-ui/react';
import { createAuthHeaders } from '../../../api/authHeaders';
import { personalAgentFailureMessage } from '../personalAgentFailureMessage';

type SavedFlags = { claude: boolean; codex: boolean };

function savedText(flags: SavedFlags): string {
  return `Claude key saved: ${flags.claude ? 'yes' : 'no'}. Codex key saved: ${flags.codex ? 'yes' : 'no'}.`;
}

async function failureText(response: Response): Promise<string> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  return personalAgentFailureMessage(body, response.status);
}

export function PersonalAgentModal({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const toast = useToast();
  const [claudeKey, setClaudeKey] = useState('');
  const [codexKey, setCodexKey] = useState('');
  const [status, setStatus] = useState('');
  const [saving, setSaving] = useState(false);
  const inFlight = useRef(false);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    let cancelled = false;
    setClaudeKey('');
    setCodexKey('');
    setStatus("");
    (async () => {
      try {
        const headers = await createAuthHeaders();
        const response = await fetch('/api/chat/personal-agent/', {
          credentials: 'include',
          headers,
        });
        if (!response.ok) {
          throw new Error(await failureText(response));
        }
        const data = (await response.json()) as SavedFlags;
        if (!cancelled) {
          setStatus(savedText(data));
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : 'Could not read the saved sign-in.');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  const save = async () => {
    const body: Record<string, string> = {};
    if (claudeKey.trim()) {
      body.claude_key = claudeKey.trim();
    }
    if (codexKey.trim()) {
      body.codex_key = codexKey.trim();
    }
    if (Object.keys(body).length === 0) {
      toast({
        title: 'Enter a key to save',
        status: 'info',
        duration: 2000,
        isClosable: true,
      });
      return;
    }
    if (inFlight.current) {
      return;
    }
    inFlight.current = true;
    setSaving(true);
    try {
      const headers = await createAuthHeaders();
      const response = await fetch('/api/chat/personal-agent/', {
        method: 'PUT',
        credentials: 'include',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(await failureText(response));
      }
      const data = (await response.json()) as SavedFlags;
      setClaudeKey('');
      setCodexKey('');
      setStatus(savedText(data));
      toast({
        title: 'Personal agent sign-in saved',
        status: 'success',
        duration: 2000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Could not save the sign-in',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      inFlight.current = false;
      setSaving(false);
    }
  };

  const clear = async () => {
    if (inFlight.current) {
      return;
    }
    inFlight.current = true;
    setSaving(true);
    try {
      const headers = await createAuthHeaders();
      const response = await fetch('/api/chat/personal-agent/', {
        method: 'DELETE',
        credentials: 'include',
        headers,
      });
      if (!response.ok) {
        throw new Error(await failureText(response));
      }
      const data = (await response.json()) as SavedFlags;
      setClaudeKey('');
      setCodexKey('');
      setStatus(savedText(data));
    } catch (error) {
      toast({
        title: 'Could not clear the sign-in',
        description: error instanceof Error ? error.message : 'Unknown error',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    } finally {
      inFlight.current = false;
      setSaving(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Personal Claude or Codex</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <VStack align="stretch" spacing={3}>
            <Text fontSize="sm">
              These keys are yours. While this project's notebook is open, the lab uses them instead of the shared account. Leave a field blank to keep the key already saved.
            </Text>
            {status && <Text fontSize="sm">{status}</Text>}
            <FormControl>
              <FormLabel fontSize="sm">Claude API key</FormLabel>
              <Input
                type="password"
                autoComplete="new-password"
                value={claudeKey}
                onChange={(event) => setClaudeKey(event.target.value)}
              />
            </FormControl>
            <FormControl>
              <FormLabel fontSize="sm">Codex / OpenAI API key</FormLabel>
              <Input
                type="password"
                autoComplete="new-password"
                value={codexKey}
                onChange={(event) => setCodexKey(event.target.value)}
              />
            </FormControl>
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={() => void clear()}>
            Clear
          </Button>
          <Button colorScheme="blue" isLoading={saving} onClick={() => void save()}>
            Save
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
}
