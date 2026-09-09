import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalBody,
  ModalFooter,
  Button,
  FormControl,
  FormLabel,
  Select,
  Input,
  VStack,
  HStack,
  Text,
  Textarea,
  Link,
  Spinner,
} from "@chakra-ui/react";
import {
  prepareClusterRun,
  putClusterSbatch,
  getClusterSbatch,
} from "../../../api/workflowRunApi";
import { JUPYTER_BASE_URL } from "../../../config/urls";

export interface ClusterSubmitPayload {
  resourceRequests: Record<string, unknown>;
  runId: string;
  sbatch: string;
}

interface ClusterRunModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (payload: ClusterSubmitPayload) => void;
  isSubmitting: boolean;
  workflowId?: string | null;
  fromRunId?: string | null;
  contextResources?: Record<string, unknown>;
}

const GPU_PARTITIONS: Record<string, string> = {
  gcalc1: "L40",
  gcalc2: "H100",
};

const KNOWN_PARTITIONS = new Set(["ccalc", "gcalc1", "gcalc2"]);

const hoursToHHMMSS = (h: unknown): string | undefined => {
  const n = typeof h === "number" ? h : Number(h);
  if (!n || n <= 0 || Number.isNaN(n)) return undefined;
  const total = Math.round(n * 3600);
  const hh = Math.floor(total / 3600);
  const mm = Math.floor((total % 3600) / 60);
  const ss = total % 60;
  return [hh, mm, ss].map((x) => String(x).padStart(2, "0")).join(":");
};

const jupyterFileUrl = (jupyterPath: string) =>
  `${JUPYTER_BASE_URL}/user/user1/lab/workspaces/auto-E/tree/${jupyterPath}`;

const ClusterRunModal: React.FC<ClusterRunModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  isSubmitting,
  workflowId,
  fromRunId,
  contextResources,
}) => {
  const [partition, setPartition] = useState("ccalc");
  const [walltime, setWalltime] = useState("00:30:00");
  const [cpus, setCpus] = useState("2");
  const [memGb, setMemGb] = useState("4");
  const [gpus, setGpus] = useState("1");
  const [sbatch, setSbatch] = useState("");
  const [draftId, setDraftId] = useState<string | null>(null);
  const [jupyterPath, setJupyterPath] = useState("");
  const [dirty, setDirty] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [prepareError, setPrepareError] = useState("");
  const rrSignature = useRef("");

  const isGpu = partition in GPU_PARTITIONS;

  const buildResourceRequests = useCallback(() => {
    const rr: Record<string, unknown> = { partition, time: walltime };
    if (cpus.trim()) rr.cpus_per_task = Number(cpus);
    if (memGb.trim()) rr.mem = `${Number(memGb)}G`;
    if (isGpu && gpus.trim()) {
      rr.gres = `gpu:${GPU_PARTITIONS[partition]}:${Number(gpus)}`;
    }
    return rr;
  }, [partition, walltime, cpus, memGb, gpus, isGpu]);

  useEffect(() => {
    if (!isOpen) {
      setSbatch("");
      setDraftId(null);
      setJupyterPath("");
      setDirty(false);
      setPrepareError("");
      setPreparing(false);
      rrSignature.current = "";
      return;
    }
    const r = (contextResources || {}) as Record<string, any>;
    const q = typeof r.queue === "string" ? r.queue : "";
    const nextPartition = KNOWN_PARTITIONS.has(q) ? q : "ccalc";
    const nextCpus = r.cpus != null ? String(r.cpus) : "2";
    const nextMem = r.memory_gb != null ? String(r.memory_gb) : "4";
    const nextWall = hoursToHHMMSS(r.walltime_hours) ?? "00:30:00";
    const nextGpus = r.gpus != null ? String(r.gpus) : "1";
    setPartition(nextPartition);
    setCpus(nextCpus);
    setMemGb(nextMem);
    setWalltime(nextWall);
    setGpus(nextGpus);
    setDirty(false);

    if (!workflowId) return;
    const rr: Record<string, unknown> = {
      partition: nextPartition,
      time: nextWall,
    };
    if (nextCpus.trim()) rr.cpus_per_task = Number(nextCpus);
    if (nextMem.trim()) rr.mem = `${Number(nextMem)}G`;
    if (nextPartition in GPU_PARTITIONS && nextGpus.trim()) {
      rr.gres = `gpu:${GPU_PARTITIONS[nextPartition]}:${Number(nextGpus)}`;
    }
    let cancelled = false;
    setPreparing(true);
    setPrepareError("");
    prepareClusterRun(workflowId, rr, fromRunId || undefined)
      .then((run) => {
        if (cancelled) return;
        setDraftId(run.id);
        setSbatch(run.sbatch || "");
        setJupyterPath(run.jupyter_path || "");
        rrSignature.current = JSON.stringify(rr);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setPrepareError(
          err instanceof Error ? err.message : "Failed to prepare run.sbatch"
        );
      })
      .finally(() => {
        if (!cancelled) setPreparing(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, workflowId, fromRunId, contextResources]);

  useEffect(() => {
    if (!isOpen || !workflowId || !draftId || preparing) return;
    const rr = buildResourceRequests();
    const sig = JSON.stringify(rr);
    if (sig === rrSignature.current) return;
    if (dirty) {
      const replace = window.confirm(
        "Resource fields changed. Replace the edited run.sbatch with a newly generated script?"
      );
      if (!replace) {
        rrSignature.current = sig;
        return;
      }
      setDirty(false);
    }
    let cancelled = false;
    putClusterSbatch(workflowId, draftId, { resource_requests: rr })
      .then((run) => {
        if (cancelled) return;
        setSbatch(run.sbatch || "");
        rrSignature.current = sig;
      })
      .catch(() => {
        /* keep the last script; user can still submit or reload */
      });
    return () => {
      cancelled = true;
    };
  }, [
    isOpen,
    workflowId,
    draftId,
    dirty,
    preparing,
    buildResourceRequests,
  ]);

  const handleReload = async () => {
    if (!workflowId || !draftId) return;
    try {
      const run = await getClusterSbatch(workflowId, draftId);
      setSbatch(run.sbatch || "");
      setDirty(false);
    } catch (err: unknown) {
      setPrepareError(
        err instanceof Error ? err.message : "Failed to reload run.sbatch"
      );
    }
  };

  const handleSubmit = () => {
    if (!draftId) return;
    onSubmit({
      resourceRequests: buildResourceRequests(),
      runId: draftId,
      sbatch,
    });
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered size="xl">
      <ModalOverlay />
      <ModalContent maxW="760px">
        <ModalHeader>Run on compute cluster</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Text fontSize="sm" color="gray.600" mb={4}>
            Resource fields generate a Slurm script. Edit it here or in
            Jupyter, then submit. Closing without submit keeps a draft in the
            Runs panel. Progress and copied logs appear there after the job
            finishes.
          </Text>
          <VStack spacing={4} align="stretch">
            <FormControl>
              <FormLabel fontSize="sm">Partition</FormLabel>
              <Select
                size="sm"
                value={partition}
                onChange={(e) => setPartition(e.target.value)}
              >
                <option value="ccalc">ccalc — CPU nodes</option>
                <option value="gcalc1">gcalc1 — GPU (NVIDIA L40)</option>
                <option value="gcalc2">gcalc2 — GPU (NVIDIA H100)</option>
              </Select>
            </FormControl>

            <HStack spacing={3}>
              <FormControl>
                <FormLabel fontSize="sm">CPUs</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  min={1}
                  value={cpus}
                  onChange={(e) => setCpus(e.target.value)}
                />
              </FormControl>
              <FormControl>
                <FormLabel fontSize="sm">Memory (GB)</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  min={1}
                  value={memGb}
                  onChange={(e) => setMemGb(e.target.value)}
                />
              </FormControl>
            </HStack>

            <HStack spacing={3}>
              <FormControl>
                <FormLabel fontSize="sm">Wall time (HH:MM:SS)</FormLabel>
                <Input
                  size="sm"
                  value={walltime}
                  onChange={(e) => setWalltime(e.target.value)}
                  placeholder="00:30:00"
                />
              </FormControl>
              {isGpu && (
                <FormControl>
                  <FormLabel fontSize="sm">GPUs</FormLabel>
                  <Input
                    size="sm"
                    type="number"
                    min={1}
                    value={gpus}
                    onChange={(e) => setGpus(e.target.value)}
                  />
                </FormControl>
              )}
            </HStack>

            <FormControl>
              <HStack justify="space-between" mb={1}>
                <FormLabel fontSize="sm" mb={0}>
                  run.sbatch
                </FormLabel>
                <HStack spacing={3}>
                  <Button
                    size="xs"
                    variant="ghost"
                    onClick={handleReload}
                    isDisabled={!draftId || preparing}
                  >
                    Reload from project
                  </Button>
                  {jupyterPath && (
                    <Link
                      href={jupyterFileUrl(jupyterPath)}
                      isExternal
                      fontSize="xs"
                      color="teal.600"
                    >
                      Edit in Jupyter
                    </Link>
                  )}
                </HStack>
              </HStack>
              {preparing ? (
                <HStack py={6} justify="center">
                  <Spinner size="sm" />
                  <Text fontSize="sm" color="gray.500">
                    Preparing run.sbatch…
                  </Text>
                </HStack>
              ) : (
                <Textarea
                  value={sbatch}
                  onChange={(e) => {
                    setSbatch(e.target.value);
                    setDirty(true);
                  }}
                  fontFamily="mono"
                  fontSize="xs"
                  minH="240px"
                  spellCheck={false}
                />
              )}
              {prepareError && (
                <Text fontSize="xs" color="red.500" mt={1}>
                  {prepareError}
                </Text>
              )}
            </FormControl>
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" size="sm" mr={3} onClick={onClose}>
            Close
          </Button>
          <Button
            colorScheme="teal"
            size="sm"
            onClick={handleSubmit}
            isLoading={isSubmitting}
            loadingText="Submitting..."
            isDisabled={!draftId || preparing || !sbatch.trim()}
          >
            Submit job
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ClusterRunModal;
