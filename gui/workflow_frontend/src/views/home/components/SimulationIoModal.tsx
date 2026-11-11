import { useEffect, useRef, useState } from "react";
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

async function responseError(response: Response): Promise<string> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  const record =
    body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  if (typeof record.error === "string" && record.error) {
    return record.error;
  }
  if (typeof record.detail === "string" && record.detail) {
    return record.detail;
  }
  return `HTTP ${response.status}`;
}

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
  const [loadError, setLoadError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const savingRef = useRef(false);

  useEffect(() => {
    setRows([]);
    setLoadError(null);
    setReady(false);
    if (!isOpen || !projectId) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const headers = await createAuthHeaders();
        const [projectRes, filesRes, resultsRes] = await Promise.all([
          fetch(`/api/workflow/${projectId}/`, { credentials: "include", headers }),
          fetch(`/api/workflow/${projectId}/files/`, { credentials: "include", headers }),
          fetch(`/api/workflow/${projectId}/results/`, { credentials: "include", headers }),
        ]);
        const failed = [projectRes, filesRes, resultsRes].find((response) => !response.ok);
        if (failed) {
          const message = await responseError(failed);
          if (!cancelled) {
            setRows([]);
            setLoadError(message);
            setReady(false);
          }
          return;
        }
        const project = await projectRes.json();
        const files = await filesRes.json();
        const results = await resultsRes.json();
        const saved = (project.workflow_context?.simulation_io?.records || []) as SimulationFileNote[];
        if (!cancelled) {
          setLoadError(null);
          setRows(simulationFileRows(files.files || [], results.results || [], saved));
          setReady(true);
        }
      } catch (error) {
        if (!cancelled) {
          setRows([]);
          setReady(false);
          setLoadError(error instanceof Error ? error.message : "Could not load simulation files");
        }
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
    if (!projectId || !canEdit || !ready || savingRef.current) {
      return;
    }
    savingRef.current = true;
    setSaving(true);
    try {
      const headers = await createAuthHeaders();
      const projectRes = await fetch(`/api/workflow/${projectId}/`, { credentials: "include", headers });
      if (!projectRes.ok) {
        throw new Error(await responseError(projectRes));
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
        throw new Error(await responseError(response));
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
      savingRef.current = false;
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
          {loadError ? (
            <Text fontSize="sm">{loadError}</Text>
          ) : ready && rows.length === 0 ? (
            <Text fontSize="sm">This project folder has no data files or result files yet.</Text>
          ) : ready ? (
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
          ) : null}
        </ModalBody>
        {canEdit && (
          <ModalFooter>
            <Button colorScheme="blue" isLoading={saving} onClick={() => void save()} isDisabled={!projectId || !ready}>
              Save
            </Button>
          </ModalFooter>
        )}
      </ModalContent>
    </Modal>
  );
}
