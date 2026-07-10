import {
  Box,
  Button,
  Divider,
  FormLabel,
  HStack,
  IconButton,
  Input,
  SimpleGrid,
  Text,
  Textarea,
  VStack,
} from '@chakra-ui/react';
import { AddIcon, DeleteIcon } from '@chakra-ui/icons';
import { AttributionDraft, Contributor, ProjectLink } from '../type';

const emptyContributor = (): Contributor => ({
  name: '',
  affiliation: '',
  orcid: '',
  researchmap: '',
  role: '',
});

const emptyLink = (): ProjectLink => ({ label: '', url: '' });

interface Props {
  value: AttributionDraft;
  onChange: (next: AttributionDraft) => void;
  isDisabled?: boolean;
}

export const AttributionEditor = ({ value, onChange, isDisabled = false }: Props) => {
  const patch = (partial: Partial<AttributionDraft>) => onChange({ ...value, ...partial });

  const updateContributor = (idx: number, field: keyof Contributor, v: string) => {
    const next = value.contributors.map((c, i) => (i === idx ? { ...c, [field]: v } : c));
    patch({ contributors: next });
  };
  const addContributor = () => patch({ contributors: [...value.contributors, emptyContributor()] });
  const removeContributor = (idx: number) =>
    patch({ contributors: value.contributors.filter((_, i) => i !== idx) });

  const updateLink = (idx: number, field: keyof ProjectLink, v: string) => {
    const next = value.links.map((l, i) => (i === idx ? { ...l, [field]: v } : l));
    patch({ links: next });
  };
  const addLink = () => patch({ links: [...value.links, emptyLink()] });
  const removeLink = (idx: number) => patch({ links: value.links.filter((_, i) => i !== idx) });

  return (
    <Box>
      <Text fontWeight="semibold" fontSize="sm" mb={2}>
        Attribution &amp; Acknowledgment
      </Text>
      <Text fontSize="xs" color="gray.500" mb={3}>
        Credit the people, data, and publications behind this workflow so the work can be properly
        acknowledged and cited.
      </Text>

      {/* Contributors */}
      <FormLabel fontSize="sm" mb={1}>
        Contributors
      </FormLabel>
      <VStack spacing={3} align="stretch">
        {value.contributors.map((c, idx) => (
          <Box key={idx} borderWidth="1px" borderRadius="md" p={3}>
            <HStack justify="space-between" mb={2}>
              <Text fontSize="xs" color="gray.500">
                Contributor {idx + 1}
              </Text>
              <IconButton
                aria-label="Remove contributor"
                icon={<DeleteIcon />}
                size="xs"
                variant="ghost"
                colorScheme="red"
                isDisabled={isDisabled}
                onClick={() => removeContributor(idx)}
              />
            </HStack>
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={2}>
              <Input
                size="sm"
                placeholder="Name"
                value={c.name}
                isDisabled={isDisabled}
                onChange={(e) => updateContributor(idx, 'name', e.target.value)}
              />
              <Input
                size="sm"
                placeholder="Affiliation"
                value={c.affiliation ?? ''}
                isDisabled={isDisabled}
                onChange={(e) => updateContributor(idx, 'affiliation', e.target.value)}
              />
              <Input
                size="sm"
                placeholder="Role (e.g. PI, data, code)"
                value={c.role ?? ''}
                isDisabled={isDisabled}
                onChange={(e) => updateContributor(idx, 'role', e.target.value)}
              />
              <Input
                size="sm"
                placeholder="ORCID (e.g. 0000-0002-1825-0097)"
                value={c.orcid ?? ''}
                isDisabled={isDisabled}
                onChange={(e) => updateContributor(idx, 'orcid', e.target.value)}
              />
              <Input
                size="sm"
                placeholder="researchmap URL"
                value={c.researchmap ?? ''}
                isDisabled={isDisabled}
                onChange={(e) => updateContributor(idx, 'researchmap', e.target.value)}
              />
            </SimpleGrid>
          </Box>
        ))}
      </VStack>
      <Button
        leftIcon={<AddIcon />}
        size="xs"
        variant="outline"
        mt={2}
        isDisabled={isDisabled}
        onClick={addContributor}
      >
        Add contributor
      </Button>

      <Divider my={4} />

      {/* Fixed project-level fields */}
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={3}>
        <Box>
          <FormLabel fontSize="sm">DOI</FormLabel>
          <Input
            size="sm"
            placeholder="10.1234/zenodo.123456"
            value={value.doi}
            isDisabled={isDisabled}
            onChange={(e) => patch({ doi: e.target.value })}
          />
        </Box>
        <Box>
          <FormLabel fontSize="sm">License</FormLabel>
          <Input
            size="sm"
            placeholder="e.g. CC-BY-4.0"
            value={value.license}
            isDisabled={isDisabled}
            onChange={(e) => patch({ license: e.target.value })}
          />
        </Box>
        <Box>
          <FormLabel fontSize="sm">Contact email</FormLabel>
          <Input
            size="sm"
            type="email"
            placeholder="contact@example.org"
            value={value.contact_email}
            isDisabled={isDisabled}
            onChange={(e) => patch({ contact_email: e.target.value })}
          />
        </Box>
      </SimpleGrid>
      <Box mt={3}>
        <FormLabel fontSize="sm">Data source</FormLabel>
        <Textarea
          size="sm"
          placeholder="Origin of the data used (dataset name, repository, URL, accession, etc.)"
          value={value.data_source}
          isDisabled={isDisabled}
          onChange={(e) => patch({ data_source: e.target.value })}
        />
      </Box>
      <Box mt={3}>
        <FormLabel fontSize="sm">Funding / grant acknowledgment</FormLabel>
        <Textarea
          size="sm"
          placeholder="e.g. Supported by Brain/MINDS 2.0 (grant no. ...)"
          value={value.funding}
          isDisabled={isDisabled}
          onChange={(e) => patch({ funding: e.target.value })}
        />
      </Box>

      <Divider my={4} />

      {/* Links */}
      <FormLabel fontSize="sm" mb={1}>
        Links
      </FormLabel>
      <VStack spacing={2} align="stretch">
        {value.links.map((l, idx) => (
          <HStack key={idx} align="center">
            <Input
              size="sm"
              placeholder="Label (e.g. Paper, Dataset, Homepage)"
              value={l.label}
              isDisabled={isDisabled}
              onChange={(e) => updateLink(idx, 'label', e.target.value)}
              flex="1"
            />
            <Input
              size="sm"
              placeholder="https://..."
              value={l.url}
              isDisabled={isDisabled}
              onChange={(e) => updateLink(idx, 'url', e.target.value)}
              flex="2"
            />
            <IconButton
              aria-label="Remove link"
              icon={<DeleteIcon />}
              size="xs"
              variant="ghost"
              colorScheme="red"
              isDisabled={isDisabled}
              onClick={() => removeLink(idx)}
            />
          </HStack>
        ))}
      </VStack>
      <Button
        leftIcon={<AddIcon />}
        size="xs"
        variant="outline"
        mt={2}
        isDisabled={isDisabled}
        onClick={addLink}
      >
        Add link
      </Button>
    </Box>
  );
};
