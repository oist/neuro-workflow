import { useEffect, useState } from "react";
import {
  Button,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalFooter,
  ModalHeader,
  ModalOverlay,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
  useToast,
} from "@chakra-ui/react";
import { createAuthHeaders } from "../../../api/authHeaders";
import { SimulationFileNote, simulationFileRows } from "../simulationFileRows";

export function SimulationIoModal({
  isOpen,
  onClose,
  projectId,
  canEdit,
}: {
  isOpen: boolean;
  onClose: () => void;
  projectId: string | null;
  canEdit: boolean;
}) {
  const toast = useToast();
  const [rows, setRows] = useState<SimulationFileNote[]>([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!isOpen || !projectId) {
      return;
    }
    let cancelled = false;
    (async () => {
      const headers = await createAuthHeaders();
      const [projectRes, filesRes, resultsRes] = await Promise.all([
        fetch(`/api/workflow/${projectId}/`, { credentials: "include", headers }),
        fetch(`/api/workflow/${projectId}/files/`, { credentials: "include", headers }),
        fetch(`/api/workflow/${projectId}/results/`, { credentials: "include", headers }),
      ]);
      if (!projectRes.ok || !filesRes.ok || !resultsRes.ok) {
        return;
      }
      const project = await projectRes.json();
      const files = await filesRes.json();
      const results = await resultsRes.json();
      const saved = (project.workflow_context?.simulation_io?.records || []) as SimulationFileNote[];
      if (!cancelled) {
        setRows(simulationFileRows(files.files || [], results.results || [], saved));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isOpen, projectId]);

  const update = (index: number, field: keyof SimulationFileNote, value: string) => {
    setRows((current) => current.map((row, i) => (i === index ? { ...row, [field]: value } : row)));
  };

  const save = async () => {
    if (!projectId || !canEdit) {
      return;
    }
    setSaving(true);
    try {
      const headers = await createAuthHeaders();
      const projectRes = await fetch(`/api/workflow/${projectId}/`, { credentials: "include", headers });
      if (!projectRes.ok) {
        throw new Error(`HTTP ${projectRes.status}`);
      }
      const project = await projectRes.json();
      const response = await fetch(`/api/workflow/${projectId}/`, {
        method: "PATCH",
        credentials: "include",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          workflow_context: {
            ...(project.workflow_context || {}),
            simulation_io: { records: rows },
          },
        }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      toast({ title: "Simulation files saved", status: "success", duration: 2000, isClosable: true });
      onClose();
    } catch (error) {
      toast({
        title: "Could not save simulation files",
        description: error instanceof Error ? error.message : "Unknown error",
        status: "error",
        duration: 3000,
        isClosable: true,
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="6xl">
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Simulation inputs and outputs</ModalHeader>
        <ModalCloseButton />
        <ModalBody overflowX="auto">
          {rows.length === 0 ? (
            <Text fontSize="sm">This project folder has no data files or result files yet.</Text>
          ) : (
            <Table size="sm">
              <Thead>
                <Tr>
                  <Th>Kind</Th>
                  <Th>File</Th>
                  <Th>Source</Th>
                  <Th>Owner</Th>
                  <Th>Paper</Th>
                  <Th>Acknowledgment</Th>
                </Tr>
              </Thead>
              <Tbody>
                {rows.map((row, index) => (
                  <Tr key={`${row.kind}:${row.path}`}>
                    <Td>{row.kind}</Td>
                    <Td>{row.path}</Td>
                    <Td><Input size="sm" value={row.source} isReadOnly={!canEdit} onChange={(event) => update(index, "source", event.target.value)} /></Td>
                    <Td><Input size="sm" value={row.owner} isReadOnly={!canEdit} onChange={(event) => update(index, "owner", event.target.value)} /></Td>
                    <Td><Input size="sm" value={row.paper} isReadOnly={!canEdit} onChange={(event) => update(index, "paper", event.target.value)} /></Td>
                    <Td><Input size="sm" value={row.acknowledgment} isReadOnly={!canEdit} onChange={(event) => update(index, "acknowledgment", event.target.value)} /></Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          )}
        </ModalBody>
        {canEdit && (
          <ModalFooter>
            <Button colorScheme="blue" isLoading={saving} onClick={() => void save()} isDisabled={!projectId}>
              Save
            </Button>
          </ModalFooter>
        )}
      </ModalContent>
    </Modal>
  );
}
