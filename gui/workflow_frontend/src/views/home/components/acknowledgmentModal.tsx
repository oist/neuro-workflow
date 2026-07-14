import {
  Box,
  Button,
  HStack,
  Link,
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
import { CopyIcon, ExternalLinkIcon } from '@chakra-ui/icons';
import { Project } from '../type';

const doiUrl = (doi: string) =>
  doi.startsWith('http') ? doi : `https://doi.org/${doi.replace(/^doi:/i, '').trim()}`;

const orcidUrl = (orcid: string) =>
  orcid.startsWith('http') ? orcid : `https://orcid.org/${orcid.trim()}`;

const hasAttribution = (p: Project): boolean =>
  Boolean(
    (p.contributors && p.contributors.length) ||
      (p.links && p.links.length) ||
      p.doi ||
      p.data_source ||
      p.license ||
      p.funding ||
      p.contact_email
  );

const buildPlainText = (p: Project): string => {
  const lines: string[] = [];
  lines.push(`Project: ${p.name}`);
  if (p.contributors && p.contributors.length) {
    const people = p.contributors
      .map((c) => {
        const parts = [c.name];
        if (c.affiliation) parts.push(`(${c.affiliation})`);
        if (c.role) parts.push(`[${c.role}]`);
        return parts.filter(Boolean).join(' ');
      })
      .filter(Boolean);
    if (people.length) lines.push(`Contributors: ${people.join('; ')}`);
  }
  if (p.doi) lines.push(`DOI: ${doiUrl(p.doi)}`);
  if (p.data_source) lines.push(`Data source: ${p.data_source}`);
  if (p.license) lines.push(`License: ${p.license}`);
  if (p.funding) lines.push(`Funding: ${p.funding}`);
  if (p.contact_email) lines.push(`Contact: ${p.contact_email}`);
  if (p.links && p.links.length) {
    lines.push('Links:');
    p.links.forEach((l) => lines.push(`  - ${l.label ? `${l.label}: ` : ''}${l.url}`));
  }
  return lines.join('\n');
};

const Field = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <Box>
    <Text fontSize="xs" color="gray.500" fontWeight="semibold" textTransform="uppercase">
      {label}
    </Text>
    <Box fontSize="sm">{children}</Box>
  </Box>
);

interface Props {
  project: Project | null;
  isOpen: boolean;
  onClose: () => void;
}

export const AcknowledgmentModal = ({ project, isOpen, onClose }: Props) => {
  const toast = useToast();
  if (!project) return null;

  const empty = !hasAttribution(project);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(buildPlainText(project));
      toast({ title: 'Copied to clipboard', status: 'success', duration: 2000, isClosable: true });
    } catch {
      toast({ title: 'Copy failed', status: 'error', duration: 2000, isClosable: true });
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="lg" scrollBehavior="inside">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>How to cite / Acknowledgment</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Text fontWeight="semibold" mb={3}>
            {project.name}
          </Text>

          {empty ? (
            <Text fontSize="sm" color="gray.500">
              No attribution details have been added for this project yet.
            </Text>
          ) : (
            <VStack align="stretch" spacing={3}>
              {project.contributors && project.contributors.length > 0 && (
                <Field label="Contributors">
                  <VStack align="stretch" spacing={1} mt={1}>
                    {project.contributors.map((c, i) => (
                      <Box key={i}>
                        <Text as="span" fontWeight="medium">
                          {c.name}
                        </Text>
                        {c.affiliation && (
                          <Text as="span" color="gray.600">
                            {' '}
                            — {c.affiliation}
                          </Text>
                        )}
                        {c.role && (
                          <Text as="span" color="gray.500">
                            {' '}
                            [{c.role}]
                          </Text>
                        )}
                        <HStack spacing={3} mt={0.5}>
                          {c.orcid && (
                            <Link href={orcidUrl(c.orcid)} isExternal color="teal.600" fontSize="xs">
                              ORCID <ExternalLinkIcon mx="1px" />
                            </Link>
                          )}
                          {c.researchmap && (
                            <Link href={c.researchmap} isExternal color="teal.600" fontSize="xs">
                              researchmap <ExternalLinkIcon mx="1px" />
                            </Link>
                          )}
                        </HStack>
                      </Box>
                    ))}
                  </VStack>
                </Field>
              )}

              {project.doi && (
                <Field label="DOI">
                  <Link href={doiUrl(project.doi)} isExternal color="teal.600">
                    {project.doi} <ExternalLinkIcon mx="1px" />
                  </Link>
                </Field>
              )}

              {project.data_source && (
                <Field label="Data source">
                  <Text whiteSpace="pre-wrap">{project.data_source}</Text>
                </Field>
              )}

              {project.license && <Field label="License">{project.license}</Field>}

              {project.funding && (
                <Field label="Funding">
                  <Text whiteSpace="pre-wrap">{project.funding}</Text>
                </Field>
              )}

              {project.contact_email && (
                <Field label="Contact">
                  <Link href={`mailto:${project.contact_email}`} color="teal.600">
                    {project.contact_email}
                  </Link>
                </Field>
              )}

              {project.links && project.links.length > 0 && (
                <Field label="Links">
                  <VStack align="stretch" spacing={1} mt={1}>
                    {project.links.map((l, i) => (
                      <Link key={i} href={l.url} isExternal color="teal.600">
                        {l.label || l.url} <ExternalLinkIcon mx="1px" />
                      </Link>
                    ))}
                  </VStack>
                </Field>
              )}
            </VStack>
          )}
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" mr={3} onClick={onClose}>
            Close
          </Button>
          <Button leftIcon={<CopyIcon />} colorScheme="blue" onClick={handleCopy} isDisabled={empty}>
            Copy acknowledgment
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default AcknowledgmentModal;
