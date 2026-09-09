import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  Box,
  Badge,
  Button,
  HStack,
  VStack,
  Text,
  Spinner,
  Link,
  useColorModeValue,
  Collapse,
  IconButton,
  Tooltip,
} from "@chakra-ui/react";
import {
  FiChevronDown,
  FiChevronUp,
  FiXCircle,
  FiRefreshCw,
  FiDownload,
  FiTrash2,
} from "react-icons/fi";
import {
  WorkflowRunRecord,
  ArtifactFile,
  getWorkflowRunStatus,
  cancelWorkflowRun,
  listWorkflowRuns,
  downloadArtifact,
  deleteWorkflowRun,
} from "../../../api/workflowRunApi";
import { JUPYTER_BASE_URL } from "../../../config/urls";

interface RunStatusPanelProps {
  workflowId: string;
  latestRunId?: string;
  onResubmit?: (runId: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
  draft: "purple",
  pending: "yellow",
  running: "blue",
  completed: "green",
  failed: "red",
  cancelled: "gray",
};

const TERMINAL_STATUSES = new Set(["draft", "completed", "failed", "cancelled"]);

const jupyterRunFolderUrl = (workflowId: string, runId: string) =>
  `${JUPYTER_BASE_URL}/user/user1/lab/workspaces/auto-E/tree/codes/projects/${workflowId}/batch/${runId}`;

const RunStatusPanel: React.FC<RunStatusPanelProps> = ({
  workflowId,
  latestRunId,
  onResubmit,
}) => {
  const [runs, setRuns] = useState<WorkflowRunRecord[]>([]);
  const [expanded, setExpanded] = useState(false);
  const [selectedRun, setSelectedRun] = useState<WorkflowRunRecord | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bg = useColorModeValue("white", "gray.800");
  const borderColor = useColorModeValue("gray.200", "gray.600");

  const loadRuns = useCallback(async () => {
    try {
      const data = await listWorkflowRuns(workflowId);
      setRuns(data);
      if (latestRunId) {
        const latest = data.find((r) => r.id === latestRunId);
        if (latest) setSelectedRun(latest);
      }
    } catch {
      // ignore
    }
  }, [workflowId, latestRunId]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (!selectedRun || TERMINAL_STATUSES.has(selectedRun.status)) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const updated = await getWorkflowRunStatus(
          workflowId,
          selectedRun.id
        );
        setSelectedRun(updated);
        if (TERMINAL_STATUSES.has(updated.status)) {
          if (pollRef.current) clearInterval(pollRef.current);
          loadRuns();
        }
      } catch {
        // ignore
      }
    }, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [selectedRun, workflowId, loadRuns]);

  const handleCancel = async () => {
    if (!selectedRun) return;
    try {
      const updated = await cancelWorkflowRun(workflowId, selectedRun.id);
      setSelectedRun(updated);
      loadRuns();
    } catch {
      // ignore
    }
  };

  const handleDelete = async (run: WorkflowRunRecord) => {
    if (
      !window.confirm(
        `Delete run ${run.id.slice(0, 8)}? This permanently removes its logs ` +
          `and results from the server and cannot be undone.`
      )
    ) {
      return;
    }
    try {
      await deleteWorkflowRun(workflowId, run.id);
      if (selectedRun?.id === run.id) setSelectedRun(null);
      loadRuns();
    } catch {
      // ignore
    }
  };

  const artifactFiles: ArtifactFile[] =
    selectedRun?.artifacts?.files ?? [];
  const logFiles: ArtifactFile[] = selectedRun?.artifacts?.logs ?? [];

  if (runs.length === 0 && !latestRunId) return null;

  return (
    <Box
      position="absolute"
      bottom="16px"
      right="16px"
      zIndex={900}
      bg={bg}
      border="1px solid"
      borderColor={borderColor}
      borderRadius="md"
      shadow="lg"
      minW="320px"
      maxW="480px"
      p={3}
    >
      <HStack justify="space-between" mb={expanded ? 2 : 0}>
        <HStack>
          <Text fontWeight="bold" fontSize="sm">
            Runs
          </Text>
          {selectedRun && (
            <Badge colorScheme={STATUS_COLORS[selectedRun.status] || "gray"}>
              {selectedRun.status}
            </Badge>
          )}
          {selectedRun &&
            !TERMINAL_STATUSES.has(selectedRun.status) && (
              <Spinner size="xs" />
            )}
        </HStack>
        <HStack>
          <Tooltip label="Refresh">
            <IconButton
              aria-label="Refresh"
              icon={<FiRefreshCw />}
              size="xs"
              variant="ghost"
              onClick={loadRuns}
            />
          </Tooltip>
          <IconButton
            aria-label="Toggle"
            icon={expanded ? <FiChevronDown /> : <FiChevronUp />}
            size="xs"
            variant="ghost"
            onClick={() => setExpanded(!expanded)}
          />
        </HStack>
      </HStack>

      <Collapse in={expanded}>
        <VStack align="stretch" spacing={2} maxH="300px" overflowY="auto">
          {runs.map((run) => (
            <Box
              key={run.id}
              p={2}
              bg={
                run.id === selectedRun?.id
                  ? useColorModeValue("blue.50", "blue.900")
                  : "transparent"
              }
              borderRadius="sm"
              cursor="pointer"
              onClick={() => setSelectedRun(run)}
              _hover={{ bg: useColorModeValue("gray.50", "gray.700") }}
            >
              <HStack justify="space-between">
                <Text fontSize="xs" fontFamily="mono">
                  {run.id.slice(0, 8)}
                </Text>
                <HStack spacing={1}>
                  <Badge
                    size="sm"
                    colorScheme={STATUS_COLORS[run.status] || "gray"}
                  >
                    {run.status}
                  </Badge>
                  <Tooltip label="Delete run (logs + results)">
                    <IconButton
                      aria-label={`Delete run ${run.id.slice(0, 8)}`}
                      icon={<FiTrash2 />}
                      size="xs"
                      variant="ghost"
                      colorScheme="red"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(run);
                      }}
                    />
                  </Tooltip>
                </HStack>
              </HStack>
              <Text fontSize="xs" color="gray.500">
                {run.backend} &middot;{" "}
                {new Date(run.submitted_at).toLocaleString()}
              </Text>
            </Box>
          ))}
        </VStack>

        {selectedRun && (
          <Box mt={2}>
            {selectedRun.stdout && (
              <Box
                bg="gray.900"
                color="green.300"
                p={2}
                borderRadius="sm"
                maxH="150px"
                overflowY="auto"
                fontSize="xs"
                fontFamily="mono"
                whiteSpace="pre-wrap"
              >
                {selectedRun.stdout}
              </Box>
            )}
            {selectedRun.stderr && (
              <Box
                bg="gray.900"
                color="red.300"
                p={2}
                mt={1}
                borderRadius="sm"
                maxH="100px"
                overflowY="auto"
                fontSize="xs"
                fontFamily="mono"
                whiteSpace="pre-wrap"
              >
                {selectedRun.stderr}
              </Box>
            )}
            {selectedRun.error_message && (
              <Text fontSize="xs" color="red.500" mt={1}>
                {selectedRun.error_message}
              </Text>
            )}
            {logFiles.length > 0 && (
              <Box mt={2}>
                <Text fontSize="xs" fontWeight="bold" mb={1}>
                  Logs ({logFiles.length})
                </Text>
                <VStack
                  align="stretch"
                  spacing={0}
                  maxH="160px"
                  overflowY="auto"
                  border="1px solid"
                  borderColor={borderColor}
                  borderRadius="sm"
                >
                  {logFiles.map((f) => (
                    <HStack
                      key={f.path}
                      justify="space-between"
                      px={2}
                      py={1}
                      _hover={{ bg: useColorModeValue("gray.50", "gray.700") }}
                    >
                      <Text
                        fontSize="xs"
                        fontFamily="mono"
                        isTruncated
                        title={f.path}
                      >
                        {f.path}
                      </Text>
                      <Tooltip label={`Download (${f.size} bytes)`}>
                        <IconButton
                          aria-label={`Download ${f.path}`}
                          icon={<FiDownload />}
                          size="xs"
                          variant="ghost"
                          onClick={() =>
                            downloadArtifact(workflowId, selectedRun.id, f.path)
                          }
                        />
                      </Tooltip>
                    </HStack>
                  ))}
                </VStack>
              </Box>
            )}
            {artifactFiles.length > 0 && (
              <Box mt={2}>
                <Text fontSize="xs" fontWeight="bold" mb={1}>
                  Results ({artifactFiles.length})
                </Text>
                <VStack
                  align="stretch"
                  spacing={0}
                  maxH="160px"
                  overflowY="auto"
                  border="1px solid"
                  borderColor={borderColor}
                  borderRadius="sm"
                >
                  {artifactFiles.map((f) => (
                    <HStack
                      key={f.path}
                      justify="space-between"
                      px={2}
                      py={1}
                      _hover={{ bg: useColorModeValue("gray.50", "gray.700") }}
                    >
                      <Text
                        fontSize="xs"
                        fontFamily="mono"
                        isTruncated
                        title={f.path}
                      >
                        {f.path}
                      </Text>
                      <Tooltip label={`Download (${f.size} bytes)`}>
                        <IconButton
                          aria-label={`Download ${f.path}`}
                          icon={<FiDownload />}
                          size="xs"
                          variant="ghost"
                          onClick={() =>
                            downloadArtifact(workflowId, selectedRun.id, f.path)
                          }
                        />
                      </Tooltip>
                    </HStack>
                  ))}
                </VStack>
              </Box>
            )}
            {selectedRun.backend === "slurm" && (
              <HStack mt={2} spacing={2} flexWrap="wrap">
                <Link
                  href={jupyterRunFolderUrl(workflowId, selectedRun.id)}
                  isExternal
                  fontSize="xs"
                  color="teal.600"
                >
                  Open run folder in Jupyter
                </Link>
                {onResubmit && (
                  <Button
                    size="xs"
                    variant="outline"
                    onClick={() => onResubmit(selectedRun.id)}
                  >
                    Edit script & resubmit
                  </Button>
                )}
              </HStack>
            )}
            {selectedRun.status !== "draft" &&
              !TERMINAL_STATUSES.has(selectedRun.status) && (
              <Button
                size="xs"
                colorScheme="red"
                variant="outline"
                leftIcon={<FiXCircle />}
                mt={2}
                onClick={handleCancel}
              >
                Cancel
              </Button>
            )}
          </Box>
        )}
      </Collapse>
    </Box>
  );
};

export default RunStatusPanel;
