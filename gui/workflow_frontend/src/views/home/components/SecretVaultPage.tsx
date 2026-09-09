import React, { useEffect, useState } from 'react';
import {
  Alert,
  AlertIcon,
  Box,
  Button,
  Code,
  FormControl,
  FormHelperText,
  FormLabel,
  HStack,
  IconButton,
  Input,
  Spinner,
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  useToast,
  VStack,
} from '@chakra-ui/react';
import { DeleteIcon } from '@chakra-ui/icons';
import { createAuthHeaders } from '../../api/authHeaders';

type VaultSecret = {
  id: string;
  name: string;
  description?: string;
  is_set: boolean;
  created_at?: string;
  updated_at?: string;
};

const SecretVaultPage: React.FC = () => {
  const [secrets, setSecrets] = useState<VaultSecret[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState('ASPERA_PASSWORD');
  const [value, setValue] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [rotateId, setRotateId] = useState<string | null>(null);
  const [rotateValue, setRotateValue] = useState('');
  const toast = useToast();

  const fetchSecrets = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/secrets/', { headers: await createAuthHeaders() });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setSecrets(Array.isArray(data) ? data : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to list secrets');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSecrets();
  }, []);

  const createSecret = async () => {
    setSaving(true);
    try {
      const response = await fetch('/api/secrets/', {
        method: 'POST',
        headers: { ...(await createAuthHeaders()), 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, value, description }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || `HTTP ${response.status}`);
      }
      setValue('');
      toast({ title: 'Secret saved', status: 'success', duration: 2000, isClosable: true });
      await fetchSecrets();
    } catch (err) {
      toast({
        title: 'Could not save secret',
        description: err instanceof Error ? err.message : 'Unknown error',
        status: 'error',
        duration: 4000,
        isClosable: true,
      });
    } finally {
      setSaving(false);
    }
  };

  const rotateSecret = async (secret: VaultSecret, nextValue: string) => {
    const response = await fetch(`/api/secrets/${secret.id}/`, {
      method: 'PATCH',
      headers: { ...(await createAuthHeaders()), 'Content-Type': 'application/json' },
      body: JSON.stringify({ value: nextValue }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  };

  const deleteSecret = async (secret: VaultSecret) => {
    const response = await fetch(`/api/secrets/${secret.id}/`, {
      method: 'DELETE',
      headers: await createAuthHeaders(),
    });
    if (!response.ok && response.status !== 204) throw new Error(`HTTP ${response.status}`);
    await fetchSecrets();
  };

  const copyName = async (secretName: string) => {
    await navigator.clipboard.writeText(secretName);
    toast({ title: 'Copied name', status: 'info', duration: 1500, isClosable: true });
  };

  return (
    <Box p={6} overflowY="auto" h="100%">
      <VStack align="stretch" spacing={6} maxW="900px">
        <Box>
          <Text fontSize="2xl" fontWeight="bold">
            Secrets
          </Text>
          <Text mt={2} color="gray.500">
            Store credentials here (or set Jupyter env <Code>NW_SECRET_NAME</Code> for local
            notebooks). Never put passwords in node parameters — the workflow JSON only keeps a
            named reference.
          </Text>
        </Box>

        {error && (
          <Alert status="error">
            <AlertIcon />
            {error}
          </Alert>
        )}

        <Box borderWidth="1px" borderRadius="md" p={4}>
          <Text fontWeight="semibold" mb={3}>
            Create secret
          </Text>
          <VStack align="stretch" spacing={3}>
            <FormControl>
              <FormLabel>Name</FormLabel>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value.toUpperCase())}
                placeholder="ASPERA_PASSWORD"
              />
              <FormHelperText>Must match ^[A-Z][A-Z0-9_]{'{1,63}'}$</FormHelperText>
            </FormControl>
            <FormControl>
              <FormLabel>Value</FormLabel>
              <Input
                type="password"
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder="••••"
                autoComplete="new-password"
              />
            </FormControl>
            <FormControl>
              <FormLabel>Description</FormLabel>
              <Input value={description} onChange={(e) => setDescription(e.target.value)} />
            </FormControl>
            <Button
              colorScheme="blue"
              onClick={createSecret}
              isLoading={saving}
              isDisabled={!name || !value}
              alignSelf="flex-start"
            >
              Save
            </Button>
          </VStack>
        </Box>

        {loading ? (
          <Spinner />
        ) : (
          <TableContainer>
            <Table size="sm">
              <Thead>
                <Tr>
                  <Th>Name</Th>
                  <Th>Description</Th>
                  <Th>Value</Th>
                  <Th></Th>
                </Tr>
              </Thead>
              <Tbody>
                {secrets.map((secret) => (
                  <Tr key={secret.id}>
                    <Td>
                      <Code>{secret.name}</Code>
                    </Td>
                    <Td>{secret.description || '—'}</Td>
                    <Td>{secret.is_set ? '••••' : '—'}</Td>
                    <Td>
                      <HStack>
                        <Button size="xs" onClick={() => copyName(secret.name)}>
                          Copy name
                        </Button>
                        {rotateId === secret.id ? (
                          <HStack>
                            <Input
                              type="password"
                              size="xs"
                              value={rotateValue}
                              onChange={(e) => setRotateValue(e.target.value)}
                              placeholder="New value"
                              autoComplete="new-password"
                              w="140px"
                            />
                            <Button
                              size="xs"
                              colorScheme="blue"
                              isDisabled={!rotateValue}
                              onClick={async () => {
                                try {
                                  await rotateSecret(secret, rotateValue);
                                  setRotateId(null);
                                  setRotateValue('');
                                  toast({ title: 'Rotated', status: 'success', duration: 2000 });
                                  await fetchSecrets();
                                } catch (err) {
                                  toast({
                                    title: 'Rotate failed',
                                    description: err instanceof Error ? err.message : 'Unknown error',
                                    status: 'error',
                                  });
                                }
                              }}
                            >
                              Save
                            </Button>
                            <Button
                              size="xs"
                              variant="ghost"
                              onClick={() => {
                                setRotateId(null);
                                setRotateValue('');
                              }}
                            >
                              Cancel
                            </Button>
                          </HStack>
                        ) : (
                          <Button
                            size="xs"
                            onClick={() => {
                              setRotateId(secret.id);
                              setRotateValue('');
                            }}
                          >
                            Rotate
                          </Button>
                        )}
                        <IconButton
                          aria-label="Delete secret"
                          icon={<DeleteIcon />}
                          size="xs"
                          onClick={async () => {
                            try {
                              await deleteSecret(secret);
                            } catch (err) {
                              toast({
                                title: 'Delete failed',
                                description: err instanceof Error ? err.message : 'Unknown error',
                                status: 'error',
                              });
                            }
                          }}
                        />
                      </HStack>
                    </Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          </TableContainer>
        )}
      </VStack>
    </Box>
  );
};

export default SecretVaultPage;
