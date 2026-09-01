import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Checkbox,
  Code,
  HStack,
  Spinner,
  Tag,
  TagCloseButton,
  TagLabel,
  Text,
  VStack,
  Wrap,
  WrapItem,
  useColorModeValue,
} from '@chakra-ui/react';
import KeywordSearch from '../../../shared/keyWordSearch/keyWordSearch';
import { createAuthHeaders } from '../../../api/authHeaders';

interface SearchResult {
  identifier: string;
  name: string;
  description: string;
  laboratory_name: string;
  datestamp: string;
  set_specs: string[];
  file_count: number;
}

interface OAIPMHRecordSearchProps {
  // Current value of the node's `identifiers` parameter (comma/newline separated)
  currentValue: string;
  // True when the node's schema copy predates the `identifiers` parameter
  disabled: boolean;
  onApply: (identifiers: string) => Promise<boolean>;
}

const splitIdentifiers = (raw: string): string[] =>
  (raw || '')
    .split(/[,\n]/)
    .map((part) => part.trim())
    .filter(Boolean);

const OAIPMHRecordSearch: React.FC<OAIPMHRecordSearchProps> = ({
  currentValue,
  disabled,
  onApply,
}) => {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selected, setSelected] = useState<string[]>(splitIdentifiers(currentValue));
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searched, setSearched] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const [lastQuery, setLastQuery] = useState('');
  // ISO timestamp of the latest harvest; null = never harvested; undefined = unknown
  const [harvestedAt, setHarvestedAt] = useState<string | null | undefined>(undefined);

  const subtextColor = useColorModeValue('gray.500', 'gray.400');
  const listBorderColor = useColorModeValue('gray.200', 'gray.600');

  // Keep the selection in sync when the parameter is edited elsewhere
  useEffect(() => {
    setSelected(splitIdentifiers(currentValue));
  }, [currentValue]);

  const handleSearch = async (query: string) => {
    setLastQuery(query);
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ q: query, limit: '50' });
      const headers = await createAuthHeaders();
      const response = await fetch(`/api/harvest/oai/search/?${params.toString()}`, {
        headers,
        credentials: 'include',
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.error || `Search failed (HTTP ${response.status})`);
      }
      const data = await response.json();
      setResults(Array.isArray(data.results) ? data.results : []);
      setTruncated(Boolean(data.truncated));
      setHarvestedAt(data.harvested_at ?? null);
      setSearched(true);
    } catch (e) {
      console.error('OAI-PMH search error:', e);
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const toggle = (identifier: string) => {
    setSelected((prev) =>
      prev.includes(identifier)
        ? prev.filter((item) => item !== identifier)
        : [...prev, identifier]
    );
  };

  const apply = async (value: string) => {
    setApplying(true);
    try {
      await onApply(value);
    } finally {
      setApplying(false);
    }
  };

  if (disabled) {
    return (
      <Text fontSize="sm" color={subtextColor}>
        This node was added before dataset search existed. Remove it and drag a
        fresh OAIPMHRecordsNode from the sidebar to enable selection.
      </Text>
    );
  }

  return (
    <VStack align="stretch" spacing={3}>
      <Text fontSize="sm" color={subtextColor}>
        Search the harvested repository records and select datasets to fetch. The
        selection is stored in the <Code fontSize="xs">identifiers</Code> parameter.
      </Text>
      <KeywordSearch
        onSearch={handleSearch}
        placeholder="Search repository datasets..."
        size="sm"
      />
      {harvestedAt !== undefined && (
        <Text fontSize="xs" color={subtextColor}>
          {harvestedAt
            ? `Repository index updated: ${new Date(harvestedAt).toLocaleString()}`
            : 'The repository index has not been harvested yet.'}
        </Text>
      )}
      {selected.length > 0 && (
        <Box>
          <Text fontSize="xs" fontWeight="bold" mb={1}>
            Selected ({selected.length})
          </Text>
          <Wrap>
            {selected.map((identifier) => (
              <WrapItem key={identifier}>
                <Tag size="sm" colorScheme="teal">
                  <TagLabel maxW="280px">{identifier}</TagLabel>
                  <TagCloseButton onClick={() => toggle(identifier)} />
                </Tag>
              </WrapItem>
            ))}
          </Wrap>
        </Box>
      )}
      {loading && (
        <HStack>
          <Spinner size="sm" />
          <Text fontSize="sm">Searching...</Text>
        </HStack>
      )}
      {error && (
        <HStack>
          <Text fontSize="sm" color="red.400">
            {error}
          </Text>
          <Button size="xs" onClick={() => handleSearch(lastQuery)}>
            Retry
          </Button>
        </HStack>
      )}
      {!loading && !error && searched && results.length === 0 && (
        <Text fontSize="sm" color={subtextColor}>
          No matching records
        </Text>
      )}
      {!loading && results.length > 0 && (
        <VStack
          align="stretch"
          spacing={2}
          maxH="240px"
          overflowY="auto"
          border="1px"
          borderColor={listBorderColor}
          borderRadius="md"
          p={2}
        >
          {results.map((result) => (
            <Checkbox
              key={result.identifier}
              size="sm"
              isChecked={selected.includes(result.identifier)}
              onChange={() => toggle(result.identifier)}
            >
              <HStack spacing={2} align="baseline" flexWrap="wrap">
                <Text fontSize="sm" fontWeight="semibold">
                  {result.name || result.identifier}
                </Text>
                <Code fontSize="xs">{result.identifier}</Code>
                {result.laboratory_name && (
                  <Text fontSize="xs" color={subtextColor}>
                    {result.laboratory_name}
                  </Text>
                )}
                <Text fontSize="xs" color={subtextColor}>
                  {result.file_count} file{result.file_count === 1 ? '' : 's'}
                </Text>
              </HStack>
            </Checkbox>
          ))}
          {truncated && (
            <Text fontSize="xs" color={subtextColor}>
              More matches exist — refine the search to narrow them down.
            </Text>
          )}
        </VStack>
      )}
      <HStack>
        <Button
          size="sm"
          colorScheme="teal"
          onClick={() => apply(selected.join(', '))}
          isLoading={applying}
          isDisabled={loading}
        >
          Apply selection
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            setSelected([]);
            apply('');
          }}
          isDisabled={applying || (selected.length === 0 && !currentValue)}
        >
          Clear
        </Button>
      </HStack>
    </VStack>
  );
};

export default OAIPMHRecordSearch;
