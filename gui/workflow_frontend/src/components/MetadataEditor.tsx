import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  Button,
  HStack,
  IconButton,
  Input,
  Tag,
  TagLabel,
  Text,
  VStack,
  Wrap,
  WrapItem,
  useColorModeValue,
} from '@chakra-ui/react';
import { AddIcon, DeleteIcon } from '@chakra-ui/icons';

interface MetadataRow {
  id: string;
  key: string;
  value: string;
}

interface MetadataEditorProps {
  initialMetadata?: Record<string, string>;
  label?: string;
  disabled?: boolean;
  presetKeys?: string[];
  onChange?: (metadata: Record<string, string>) => void;
}

const DEFAULT_PRESET_KEYS = [
  'Affiliation',
  'Affiliation URL',
  'Collaborators',
  'paper DOI',
  'Funding',
];

const EMPTY_METADATA: Record<string, string> = {};

const rowsToDict = (rows: MetadataRow[]): Record<string, string> => {
  const dict: Record<string, string> = {};
  for (const row of rows) {
    const k = row.key.trim();
    if (!k) continue;
    dict[k] = row.value;
  }
  return dict;
};

const dictToRows = (dict: Record<string, string>, nextId: () => string): MetadataRow[] =>
  Object.entries(dict).map(([key, value]) => ({
    id: nextId(),
    key,
    value: typeof value === 'string' ? value : String(value),
  }));

export const MetadataEditor = ({
  initialMetadata = EMPTY_METADATA,
  label = 'Metadata (Optional)',
  disabled = false,
  presetKeys = DEFAULT_PRESET_KEYS,
  onChange,
}: MetadataEditorProps) => {
  const counterRef = useRef(0);
  const nextId = () => {
    counterRef.current += 1;
    return `meta-${counterRef.current}`;
  };

  const [rows, setRows] = useState<MetadataRow[]>(() => dictToRows(initialMetadata, nextId));

  useEffect(() => {
    setRows(dictToRows(initialMetadata, nextId));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialMetadata]);

  useEffect(() => {
    onChange?.(rowsToDict(rows));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows]);

  const duplicateKeyWarning = useMemo(() => {
    const seen = new Set<string>();
    const dups = new Set<string>();
    for (const row of rows) {
      const k = row.key.trim();
      if (!k) continue;
      if (seen.has(k)) dups.add(k);
      seen.add(k);
    }
    return dups.size > 0 ? Array.from(dups) : null;
  }, [rows]);

  const labelColor = useColorModeValue('gray.700', 'gray.200');
  const helperColor = useColorModeValue('gray.500', 'gray.400');

  const addRow = (key = '') => {
    setRows(prev => [...prev, { id: nextId(), key, value: '' }]);
  };

  const removeRow = (id: string) => {
    setRows(prev => prev.filter(r => r.id !== id));
  };

  const updateRow = (id: string, patch: Partial<Omit<MetadataRow, 'id'>>) => {
    setRows(prev => prev.map(r => (r.id === id ? { ...r, ...patch } : r)));
  };

  return (
    <Box>
      <Text fontSize="sm" fontWeight="semibold" mb={2} color={labelColor}>
        {label}
      </Text>

      {presetKeys.length > 0 && (
        <Box mb={3}>
          <Text fontSize="xs" color={helperColor} mb={1}>
            Quick add:
          </Text>
          <Wrap spacing={2}>
            {presetKeys.map(preset => (
              <WrapItem key={preset}>
                <Tag
                  as="button"
                  type="button"
                  size="sm"
                  colorScheme="blue"
                  variant="subtle"
                  disabled={disabled}
                  aria-label={`Add metadata row for ${preset}`}
                  onClick={() => addRow(preset)}
                  _disabled={{ cursor: 'not-allowed', opacity: 0.6 }}
                >
                  <TagLabel>+ {preset}</TagLabel>
                </Tag>
              </WrapItem>
            ))}
          </Wrap>
        </Box>
      )}

      <VStack spacing={2} align="stretch">
        {rows.map((row, index) => {
          const trimmedKey = row.key.trim();
          const rowLabel = trimmedKey || `row ${index + 1}`;
          return (
            <HStack key={row.id} spacing={2}>
              <Input
                size="sm"
                placeholder="Key"
                aria-label={`Metadata key (row ${index + 1})`}
                value={row.key}
                onChange={e => updateRow(row.id, { key: e.target.value })}
                isDisabled={disabled}
                flex="1"
              />
              <Input
                size="sm"
                placeholder="Value"
                aria-label={
                  trimmedKey
                    ? `Metadata value for ${trimmedKey}`
                    : `Metadata value (row ${index + 1})`
                }
                value={row.value}
                onChange={e => updateRow(row.id, { value: e.target.value })}
                isDisabled={disabled}
                flex="2"
              />
              <IconButton
                aria-label={`Remove metadata ${rowLabel}`}
                icon={<DeleteIcon />}
                size="sm"
                variant="ghost"
                colorScheme="red"
                onClick={() => removeRow(row.id)}
                isDisabled={disabled}
              />
            </HStack>
          );
        })}
      </VStack>

      <Button
        leftIcon={<AddIcon />}
        size="sm"
        variant="outline"
        mt={2}
        onClick={() => addRow()}
        isDisabled={disabled}
      >
        Add row
      </Button>

      {duplicateKeyWarning && (
        <Text fontSize="xs" color="orange.500" mt={2}>
          Duplicate key{duplicateKeyWarning.length > 1 ? 's' : ''}:{' '}
          {duplicateKeyWarning.join(', ')}. The last row's value will be saved.
        </Text>
      )}
    </Box>
  );
};
