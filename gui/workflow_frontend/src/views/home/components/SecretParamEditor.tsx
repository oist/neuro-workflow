import React, { useEffect, useState } from 'react';
import {
  Button,
  Code,
  HStack,
  Input,
  Select,
  Text,
  VStack,
  useToast,
} from '@chakra-ui/react';
import { createAuthHeaders } from '../../../api/authHeaders';
import { isSecretRef, makeSecretRef, secretRefName } from '../../../utils/secretRefs';

type VaultSecret = { id: string; name: string; is_set?: boolean };

interface SecretParamEditorProps {
  value: unknown;
  isWorkflowNode: boolean;
  onBind: (ref: ReturnType<typeof makeSecretRef>) => Promise<void> | void;
}

const SecretParamEditor: React.FC<SecretParamEditorProps> = ({
  value,
  isWorkflowNode,
  onBind,
}) => {
  const toast = useToast();
  const [secrets, setSecrets] = useState<VaultSecret[]>([]);
  const [selected, setSelected] = useState(secretRefName(value));
  const [newValue, setNewValue] = useState('');
  const [newName, setNewName] = useState('NODE_SECRET');

  useEffect(() => {
    setSelected(secretRefName(value));
  }, [value]);

  useEffect(() => {
    (async () => {
      const response = await fetch('/api/secrets/', { headers: await createAuthHeaders() });
      if (!response.ok) return;
      const data = await response.json();
      setSecrets(Array.isArray(data) ? data : []);
    })();
  }, []);

  const boundName = secretRefName(value);
  const showRef = isSecretRef(value) && boundName;

  const rejectSidebar = () => {
    toast({
      title: 'Use the vault',
      description: 'Secret parameters cannot be written into node source. Open Settings → Secrets.',
      status: 'warning',
      duration: 4000,
      isClosable: true,
    });
  };

  const bindExisting = async () => {
    if (!isWorkflowNode) {
      rejectSidebar();
      return;
    }
    const match = secrets.find((s) => s.name === selected);
    if (!match) return;
    await onBind(makeSecretRef(match.id, match.name));
  };

  const createAndBind = async () => {
    if (!isWorkflowNode) {
      rejectSidebar();
      return;
    }
    const response = await fetch('/api/secrets/', {
      method: 'POST',
      headers: { ...(await createAuthHeaders()), 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: newName, value: newValue }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      toast({
        title: 'Could not create secret',
        description: typeof body.error === 'string' ? body.error : `HTTP ${response.status}`,
        status: 'error',
      });
      return;
    }
    const created = await response.json();
    setNewValue('');
    await onBind(makeSecretRef(created.id, created.name));
  };

  return (
    <VStack align="stretch" spacing={2} flex="1">
      <HStack>
        <Text fontSize="xs">••••</Text>
        {showRef ? <Code fontSize="xs">{boundName}</Code> : <Text fontSize="xs">not bound</Text>}
      </HStack>
      <Select
        size="xs"
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
        placeholder="Select from my secrets"
      >
        {secrets.map((secret) => (
          <option key={secret.id} value={secret.name}>
            {secret.name}
          </option>
        ))}
      </Select>
      <Button size="xs" onClick={bindExisting} isDisabled={!selected}>
        Bind selected
      </Button>
      <Input
        size="xs"
        value={newName}
        onChange={(e) => setNewName(e.target.value.toUpperCase())}
        placeholder="NEW_SECRET_NAME"
      />
      <Input
        size="xs"
        type="password"
        value={newValue}
        onChange={(e) => setNewValue(e.target.value)}
        placeholder="New value"
        autoComplete="new-password"
      />
      <Button size="xs" onClick={createAndBind} isDisabled={!newName || !newValue}>
        Save to vault and bind
      </Button>
    </VStack>
  );
};

export default SecretParamEditor;
