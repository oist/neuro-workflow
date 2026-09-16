import { useEffect, useMemo, useState } from "react";
import { Node } from "@xyflow/react";
import {
  Alert, AlertIcon, Box, Button, Divider, FormControl, FormLabel, HStack, Heading, Modal,
  ModalBody, ModalCloseButton, ModalContent, ModalFooter, ModalHeader, ModalOverlay, Select,
  Text, Textarea, VStack, useToast,
} from "@chakra-ui/react";
import { CalculationNodeData, StudyObjective } from "../../type";
import { useFlowStore } from "../../../../stores/flowStore";
import {
  FlowNode, displayName, exploreRows, nodeDeclaredObjectives, studyObjectives,
} from "../../utils/studyAddress";
import { putParameterField } from "./studyApi";
import { DraftNumberInput, DraftTextInput } from "./draftInputs";
import { ExploreSection, SetParamField } from "./ExploreSection";
import { ObjectivesSection } from "./ObjectivesSection";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  optNodeId: string;
  workflowId?: string;
  updateNodeAPI: (nodeId: string, node: Partial<Node<CalculationNodeData>>) => Promise<void>;
}

const ALGORITHMS = ["random", "cmaes", "tpe", "nsga2", "nsga3", "optuna_random"];
const SINGLE_OBJECTIVE_ALGORITHMS = new Set(["cmaes"]);

const paramValue = (node: FlowNode | undefined, name: string): unknown =>
  node?.data.schema?.parameters?.[name]?.default_value;

const asInt = (v: unknown, fallback: number): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) && n >= 1 ? Math.floor(n) : fallback;
};

/** The study held on an NW_Optimization node: how to search (its own
 *  parameters), what to explore (flags on the other nodes' parameters) and what
 *  to hit (objectives kept on this node). Replaces the generic node modal. */
export const OptimizationStudyModal = ({ isOpen, onClose, optNodeId, workflowId, updateNodeAPI }: Props) => {
  const toast = useToast();
  const nodes = useFlowStore((s) => s.sharedNodes);
  const optNode = nodes.find((n) => n.id === optNodeId);

  const setParamField: SetParamField = async (nodeId, param, field, value) => {
    const node = useFlowStore.getState().sharedNodes.find((n) => n.id === nodeId);
    const schema = node?.data.schema;
    const current = schema?.parameters?.[param];
    if (!node || !schema || !current) {
      toast({ title: "Parameter not found", description: `${param} on ${nodeId}`, status: "error", duration: 3000 });
      return;
    }
    if (workflowId) {
      try {
        await putParameterField(workflowId, nodeId, param, field, value);
      } catch (e) {
        // The next Generate upserts every node with its full data, so a
        // store-only change is not lost; say so rather than fail.
        toast({
          title: "Kept locally",
          description: `Could not save ${param}.${field} now (${e instanceof Error ? e.message : "error"}); it is written with the next Generate.`,
          status: "warning",
          duration: 4000,
          isClosable: true,
        });
      }
    }
    // Mirror into the store at once: a later whole-node save reads the store.
    useFlowStore.getState().updateNodeData(nodeId, {
      schema: { ...schema, parameters: { ...schema.parameters, [param]: { ...current, [field]: value } } },
    });
  };

  const saveObjectives = async (objectives: StudyObjective[]) => {
    useFlowStore.getState().updateNodeData(optNodeId, { study: { objectives } });
    const node = useFlowStore.getState().sharedNodes.find((n) => n.id === optNodeId);
    if (node && workflowId) await updateNodeAPI(optNodeId, node);
  };

  // --- algorithm ----------------------------------------------------------------
  const algorithmField = optNode?.data.schema?.parameters?.algorithm;
  const allowed: string[] = Array.isArray(algorithmField?.constraints?.allowed_values)
    ? (algorithmField!.constraints!.allowed_values as unknown[]).map(String)
    : ALGORITHMS;
  const algorithm = String(paramValue(optNode, "algorithm") ?? "cmaes");
  const popSize = asInt(paramValue(optNode, "pop_size"), 16);
  const maxGenerations = asInt(paramValue(optNode, "max_generations"), 20);
  const seedRaw = paramValue(optNode, "seed");
  const seed = seedRaw === null || seedRaw === undefined ? "" : String(seedRaw);
  const resultsPath = String(paramValue(optNode, "results_path") ?? "results/optimization");
  const optionsRaw = paramValue(optNode, "options");
  const optionsText = typeof optionsRaw === "string" ? optionsRaw : JSON.stringify(optionsRaw ?? {});
  const [optionsDraft, setOptionsDraft] = useState(optionsText);
  const [optionsError, setOptionsError] = useState("");
  useEffect(() => setOptionsDraft(optionsText), [optionsText]);

  const commitOptions = () => {
    const text = optionsDraft.trim() || "{}";
    try {
      const parsed = JSON.parse(text);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) throw new Error("must be a JSON object");
      setOptionsError("");
      if (JSON.stringify(parsed) !== JSON.stringify(optionsRaw ?? {})) {
        setParamField(optNodeId, "options", "default_value", parsed);
      }
    } catch (e) {
      setOptionsError(e instanceof Error ? e.message : "invalid JSON");
    }
  };

  const setOwn = (name: string, value: unknown) => setParamField(optNodeId, name, "default_value", value);

  // --- summary ------------------------------------------------------------------
  const rows = useMemo(() => exploreRows(nodes), [nodes]);
  const objectiveCount = studyObjectives(optNode).length + nodeDeclaredObjectives(nodes).length;
  const budget = popSize * maxGenerations;
  const multiObjectiveWarning = SINGLE_OBJECTIVE_ALGORITHMS.has(algorithm) && objectiveCount >= 2;

  return (
    <Modal isOpen={isOpen} onClose={onClose} size="6xl" scrollBehavior="inside">
      <ModalOverlay />
      <ModalContent maxW="1200px" w="92vw">
        <ModalHeader>
          Optimization study{optNode ? `: ${displayName(optNode)}` : ""}
        </ModalHeader>
        <ModalCloseButton />
        <ModalBody>
          {!optNode ? (
            <Text color="red.400">The optimization node is no longer on the canvas.</Text>
          ) : (
            <VStack align="stretch" spacing={5}>
              <Box>
                <Heading size="sm" mb={2}>How to search</Heading>
                <HStack spacing={4} align="flex-end" flexWrap="wrap">
                  <FormControl w="150px">
                    <FormLabel fontSize="xs" mb={1}>algorithm</FormLabel>
                    <Select size="sm" value={algorithm} onChange={(e) => setOwn("algorithm", e.target.value)}>
                      {allowed.map((a) => <option key={a} value={a}>{a}</option>)}
                    </Select>
                  </FormControl>
                  <FormControl w="110px">
                    <FormLabel fontSize="xs" mb={1}>pop_size</FormLabel>
                    <DraftNumberInput value={popSize} w="100px" onCommit={(v) => v !== undefined && v >= 1 && setOwn("pop_size", Math.floor(v))} />
                  </FormControl>
                  <FormControl w="130px">
                    <FormLabel fontSize="xs" mb={1}>max_generations</FormLabel>
                    <DraftNumberInput value={maxGenerations} w="110px" onCommit={(v) => v !== undefined && v >= 1 && setOwn("max_generations", Math.floor(v))} />
                  </FormControl>
                  <FormControl w="110px">
                    <FormLabel fontSize="xs" mb={1}>seed (empty = unseeded)</FormLabel>
                    <DraftTextInput value={seed} w="100px" placeholder="none" onCommit={(v) => setOwn("seed", v)} />
                  </FormControl>
                  <FormControl w="240px">
                    <FormLabel fontSize="xs" mb={1}>results_path</FormLabel>
                    <DraftTextInput value={resultsPath} w="230px" onCommit={(v) => v && setOwn("results_path", v)} />
                  </FormControl>
                </HStack>
                <FormControl mt={2}>
                  <FormLabel fontSize="xs" mb={1}>options (JSON passed to the sampler)</FormLabel>
                  <Textarea
                    size="sm"
                    rows={2}
                    fontFamily="mono"
                    value={optionsDraft}
                    isInvalid={!!optionsError}
                    onChange={(e) => setOptionsDraft(e.target.value)}
                    onBlur={commitOptions}
                  />
                  {optionsError && <Text fontSize="xs" color="red.400">{optionsError}</Text>}
                </FormControl>
                <Text fontSize="sm" color="gray.600" mt={2}>
                  Budget: at most {popSize} × {maxGenerations} = {budget} runs, plus one baseline.
                </Text>
                {multiObjectiveWarning && (
                  <Alert status="warning" fontSize="sm" mt={2} py={2}>
                    <AlertIcon />
                    {algorithm} handles one objective; choose nsga2 or nsga3 for several.
                  </Alert>
                )}
              </Box>

              <Divider />
              <ExploreSection nodes={nodes} setParamField={setParamField} />

              <Divider />
              <ObjectivesSection nodes={nodes} optNode={optNode} saveObjectives={saveObjectives} />
            </VStack>
          )}
        </ModalBody>
        <ModalFooter justifyContent="space-between">
          <VStack align="flex-start" spacing={0}>
            <Text fontSize="sm" color="gray.600">
              {rows.length} parameter{rows.length === 1 ? "" : "s"} · {objectiveCount} objective{objectiveCount === 1 ? "" : "s"} · {algorithm} · ≤ {budget} runs
            </Text>
            {rows.length === 0 && (
              <Text fontSize="xs" color="orange.500">nothing is explored; the generated search has no dimension</Text>
            )}
            {objectiveCount === 0 && rows.length > 0 && (
              <Text fontSize="xs" color="orange.500">no objective; the engine has nothing to hit</Text>
            )}
          </VStack>
          <Button variant="ghost" onClick={onClose}>Close</Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );
};

export default OptimizationStudyModal;
