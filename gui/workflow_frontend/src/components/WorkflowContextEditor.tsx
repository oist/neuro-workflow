import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Flex,
  HStack,
  Input,
  Tag,
  TagCloseButton,
  TagLabel,
  Text,
  Textarea,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react';

interface WorkflowContextEditorProps {
  initialContext?: Record<string, any>;
  label?: string;
  disabled?: boolean;
  onChange?: (context: Record<string, any> | null, rawText: string, isValid: boolean) => void;
}

const EMPTY_CONTEXT: Record<string, any> = {};

const safeParseJson = (text: string) => {
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
};

export const WorkflowContextEditor = ({
  initialContext = EMPTY_CONTEXT,
  label = 'Workflow Context (Optional, JSON)',
  disabled = false,
  onChange,
}: WorkflowContextEditorProps) => {
  const [useAdvancedContext, setUseAdvancedContext] = useState<boolean>(false);
  const [contextText, setContextText] = useState<string>('');
  const [contextError, setContextError] = useState<string | null>(null);

  const [contextSpecies, setContextSpecies] = useState<string>('');
  const [contextMetadataSources, setContextMetadataSources] = useState<string[]>([]);
  const [contextModelScale, setContextModelScale] = useState<string>('');
  const [contextBrainRegion, setContextBrainRegion] = useState<string>('');
  const [metadataInput, setMetadataInput] = useState<string>('');
  const [resourceCpus, setResourceCpus] = useState<string>('');
  const [resourceMemory, setResourceMemory] = useState<string>('');
  const [resourceGpus, setResourceGpus] = useState<string>('');
  const [resourceWalltime, setResourceWalltime] = useState<string>('');
  const [resourceQueue, setResourceQueue] = useState<string>('');
  const [resourceAccount, setResourceAccount] = useState<string>('');

  const labelColor = useColorModeValue('gray.700', 'gray.200');

  const buildStructuredContext = () => {
    const context: Record<string, any> = {};
    if (contextSpecies.trim()) {
      context.species = contextSpecies.trim();
    }
    if (contextMetadataSources.length > 0) {
      context.metadata_sources = contextMetadataSources;
    }
    if (contextModelScale.trim()) {
      context.model_scale = contextModelScale.trim();
    }
    if (contextBrainRegion.trim()) {
      context.brain_region = contextBrainRegion.trim();
    }
    const resources: Record<string, any> = {};
    if (resourceCpus.trim()) {
      resources.cpus = Number(resourceCpus);
    }
    if (resourceMemory.trim()) {
      resources.memory_gb = Number(resourceMemory);
    }
    if (resourceGpus.trim()) {
      resources.gpus = Number(resourceGpus);
    }
    if (resourceWalltime.trim()) {
      resources.walltime_hours = Number(resourceWalltime);
    }
    if (resourceQueue.trim()) {
      resources.queue = resourceQueue.trim();
    }
    if (resourceAccount.trim()) {
      resources.account = resourceAccount.trim();
    }
    if (Object.keys(resources).length > 0) {
      context.resource_requirements = resources;
    }
    return context;
  };

  const syncStructuredFromJson = (context: Record<string, any>) => {
    setContextSpecies(typeof context.species === 'string' ? context.species : '');
    setContextMetadataSources(
      Array.isArray(context.metadata_sources) ? context.metadata_sources.map(String) : []
    );
    setContextModelScale(typeof context.model_scale === 'string' ? context.model_scale : '');
    setContextBrainRegion(typeof context.brain_region === 'string' ? context.brain_region : '');
    const resources =
      context.resource_requirements && typeof context.resource_requirements === 'object'
        ? context.resource_requirements
        : {};
    setResourceCpus(resources.cpus !== undefined ? String(resources.cpus) : '');
    setResourceMemory(resources.memory_gb !== undefined ? String(resources.memory_gb) : '');
    setResourceGpus(resources.gpus !== undefined ? String(resources.gpus) : '');
    setResourceWalltime(resources.walltime_hours !== undefined ? String(resources.walltime_hours) : '');
    setResourceQueue(resources.queue !== undefined ? String(resources.queue) : '');
    setResourceAccount(resources.account !== undefined ? String(resources.account) : '');
  };

  const syncJsonFromStructured = () => {
    const context = buildStructuredContext();
    const raw = JSON.stringify(context, null, 2);
    setContextText(raw);
    setContextError(null);
    onChange?.(context, raw, true);
    return context;
  };

  const addMetadataSource = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    if (!contextMetadataSources.includes(trimmed)) {
      setContextMetadataSources(prev => [...prev, trimmed]);
    }
    setMetadataInput('');
  };

  const syncAllFromContext = (context: Record<string, any>) => {
    const raw = JSON.stringify(context, null, 2);
    setContextText(raw);
    setContextError(null);
    syncStructuredFromJson(context);
    onChange?.(context, raw, true);
  };

  const structuredContext = useMemo(() => buildStructuredContext(), [
    contextSpecies,
    contextMetadataSources,
    contextModelScale,
    contextBrainRegion,
    resourceCpus,
    resourceMemory,
    resourceGpus,
    resourceWalltime,
    resourceQueue,
    resourceAccount,
  ]);

  useEffect(() => {
    syncAllFromContext(initialContext || {});
    setUseAdvancedContext(false);
  }, [initialContext]);

  useEffect(() => {
    if (!useAdvancedContext) {
      const raw = JSON.stringify(structuredContext, null, 2);
      setContextText(raw);
      setContextError(null);
      onChange?.(structuredContext, raw, true);
    }
  }, [structuredContext, useAdvancedContext, onChange]);

  const handleAdvancedChange = (value: string) => {
    setContextText(value);
    const parsed = safeParseJson(value);
    if (parsed === null) {
      setContextError('Workflow context must be valid JSON.');
      onChange?.(null, value, false);
      return;
    }
    setContextError(null);
    syncStructuredFromJson(parsed);
    onChange?.(parsed, value, true);
  };

  return (
    <Box>
      <HStack justify="space-between" mb={2}>
        <Text fontSize="sm" fontWeight="semibold">
          {label}
        </Text>
        <Button
          size="xs"
          variant="outline"
          isDisabled={disabled}
          onClick={() => {
            const next = !useAdvancedContext;
            setUseAdvancedContext(next);
            if (next) {
              const raw = JSON.stringify(structuredContext, null, 2);
              setContextText(raw);
              setContextError(null);
              onChange?.(structuredContext, raw, true);
            } else {
              const parsed = safeParseJson(contextText) || {};
              syncStructuredFromJson(parsed);
              syncJsonFromStructured();
            }
          }}
        >
          {useAdvancedContext ? 'Use Structured' : 'Use Advanced JSON'}
        </Button>
      </HStack>

      {!useAdvancedContext && (
        <VStack spacing={3} align="stretch">
          <Box>
            <Text fontSize="md" color={labelColor}>Species</Text>
            <Input
              value={contextSpecies}
              onChange={(e) => setContextSpecies(e.target.value)}
              placeholder="e.g., human"
              size="sm"
              isDisabled={disabled}
            />
          </Box>

          <Box>
            <Text fontSize="md" color={labelColor}>Metadata sources</Text>
            <HStack>
              <Input
                value={metadataInput}
                onChange={(e) => setMetadataInput(e.target.value)}
                placeholder="e.g., literature"
                size="sm"
                isDisabled={disabled}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addMetadataSource(metadataInput);
                  }
                }}
              />
              <Button size="sm" onClick={() => addMetadataSource(metadataInput)} isDisabled={disabled}>
                Add
              </Button>
            </HStack>
            <Flex gap={2} mt={2} flexWrap="wrap">
              {contextMetadataSources.map((src) => (
                <Tag key={src} size="sm" colorScheme="blue">
                  <TagLabel>{src}</TagLabel>
                  <TagCloseButton
                    onClick={() => setContextMetadataSources(prev => prev.filter(s => s !== src))}
                  />
                </Tag>
              ))}
            </Flex>
          </Box>

          <Box>
            <Text fontSize="md" color={labelColor}>Model scale</Text>
            <Input
              value={contextModelScale}
              onChange={(e) => setContextModelScale(e.target.value)}
              placeholder="e.g., 1:10"
              size="sm"
              isDisabled={disabled}
            />
          </Box>

          <Box>
            <Text fontSize="md" color={labelColor}>Brain region</Text>
            <Input
              value={contextBrainRegion}
              onChange={(e) => setContextBrainRegion(e.target.value)}
              placeholder="e.g., temporal_lobe"
              size="sm"
              isDisabled={disabled}
            />
          </Box>

          <Box>
            <Text fontSize="md" color={labelColor}>Resource requirements</Text>
            <HStack spacing={2}>
              <Input value={resourceCpus} onChange={(e) => setResourceCpus(e.target.value)} placeholder="cpus" size="sm" isDisabled={disabled} />
              <Input value={resourceMemory} onChange={(e) => setResourceMemory(e.target.value)} placeholder="memory_gb" size="sm" isDisabled={disabled} />
              <Input value={resourceGpus} onChange={(e) => setResourceGpus(e.target.value)} placeholder="gpus" size="sm" isDisabled={disabled} />
            </HStack>
            <HStack spacing={2} mt={2}>
              <Input value={resourceWalltime} onChange={(e) => setResourceWalltime(e.target.value)} placeholder="walltime_hours" size="sm" isDisabled={disabled} />
              <Input value={resourceQueue} onChange={(e) => setResourceQueue(e.target.value)} placeholder="queue" size="sm" isDisabled={disabled} />
              <Input value={resourceAccount} onChange={(e) => setResourceAccount(e.target.value)} placeholder="account" size="sm" isDisabled={disabled} />
            </HStack>
          </Box>
        </VStack>
      )}

      {useAdvancedContext && (
        <Box>
          <Textarea
            value={contextText}
            onChange={(e) => handleAdvancedChange(e.target.value)}
            minH="180px"
            isDisabled={disabled}
            placeholder='{"species":"human","metadata_sources":["literature"],"resource_requirements":{"cpus":4,"memory_gb":16}}'
          />
          {contextError && (
            <Text fontSize="xs" color="red.500" mt={1}>
              {contextError}
            </Text>
          )}
        </Box>
      )}
    </Box>
  );
};
