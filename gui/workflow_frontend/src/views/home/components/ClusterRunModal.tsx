import React, { useEffect, useState } from "react";
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
} from "@chakra-ui/react";

interface ClusterRunModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (resourceRequests: Record<string, unknown>) => void;
  isSubmitting: boolean;
  // Project-level defaults from FlowProject.workflow_context.resource_requirements
  // (cpus / memory_gb / gpus / walltime_hours / queue). Used to prefill the
  // form; the user can still override for this particular run.
  contextResources?: Record<string, unknown>;
}

// Partition -> GPU model, per the RIKEN compute server (gcalc1: L40, gcalc2: H100).
const GPU_PARTITIONS: Record<string, string> = {
  gcalc1: "L40",
  gcalc2: "H100",
};

const KNOWN_PARTITIONS = new Set(["ccalc", "gcalc1", "gcalc2"]);

// WorkflowContextEditor stores wall time as a number of hours; the sbatch
// --time directive wants HH:MM:SS.
const hoursToHHMMSS = (h: unknown): string | undefined => {
  const n = typeof h === "number" ? h : Number(h);
  if (!n || n <= 0 || Number.isNaN(n)) return undefined;
  const total = Math.round(n * 3600);
  const hh = Math.floor(total / 3600);
  const mm = Math.floor((total % 3600) / 60);
  const ss = total % 60;
  return [hh, mm, ss].map((x) => String(x).padStart(2, "0")).join(":");
};

const ClusterRunModal: React.FC<ClusterRunModalProps> = ({
  isOpen,
  onClose,
  onSubmit,
  isSubmitting,
  contextResources,
}) => {
  const [partition, setPartition] = useState("ccalc");
  const [walltime, setWalltime] = useState("00:30:00");
  const [cpus, setCpus] = useState("2");
  const [memGb, setMemGb] = useState("4");
  const [gpus, setGpus] = useState("1");

  // Prefill from the project's resource defaults each time the dialog opens.
  useEffect(() => {
    if (!isOpen) return;
    const r = (contextResources || {}) as Record<string, any>;
    const q = typeof r.queue === "string" ? r.queue : "";
    setPartition(KNOWN_PARTITIONS.has(q) ? q : "ccalc");
    setCpus(r.cpus != null ? String(r.cpus) : "2");
    setMemGb(r.memory_gb != null ? String(r.memory_gb) : "4");
    setWalltime(hoursToHHMMSS(r.walltime_hours) ?? "00:30:00");
    setGpus(r.gpus != null ? String(r.gpus) : "1");
  }, [isOpen, contextResources]);

  const isGpu = partition in GPU_PARTITIONS;

  const handleSubmit = () => {
    const rr: Record<string, unknown> = { partition, time: walltime };
    if (cpus.trim()) rr.cpus_per_task = Number(cpus);
    if (memGb.trim()) rr.mem = `${Number(memGb)}G`;
    if (isGpu && gpus.trim()) {
      rr.gres = `gpu:${GPU_PARTITIONS[partition]}:${Number(gpus)}`;
    }
    onSubmit(rr);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered>
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Run on compute cluster</ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          <Text fontSize="sm" color="gray.600" mb={4}>
            The workflow code is regenerated and submitted as a Slurm batch job
            on the RIKEN compute server. Progress and results appear in the Runs
            panel (bottom-right). Values are prefilled from the project's
            resource settings — adjust to override for this run.
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
          </VStack>
        </ModalBody>
        <ModalFooter>
          <Button variant="ghost" size="sm" mr={3} onClick={onClose}>
            Cancel
          </Button>
          <Button
            colorScheme="teal"
            size="sm"
            onClick={handleSubmit}
            isLoading={isSubmitting}
            loadingText="Submitting..."
          >
            Submit job
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default ClusterRunModal;
