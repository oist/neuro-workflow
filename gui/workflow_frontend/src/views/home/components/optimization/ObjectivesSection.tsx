import { useMemo, useState } from "react";
import {
  Box, Button, HStack, Heading, IconButton, Input, Select, Table, Tbody, Td, Text, Th, Thead,
  Tr, Tooltip, Badge,
} from "@chakra-ui/react";
import { DeleteIcon } from "@chakra-ui/icons";
import { StudyObjective } from "../../type";
import {
  FlowNode, displayName, isOptimizationNode, measuresLabel, nodeDeclaredObjectives,
  objectiveAutoName, outputPorts, studyObjectives,
} from "../../utils/studyAddress";
import { DraftNumberInput, DraftTextInput } from "./draftInputs";

interface Props {
  nodes: FlowNode[];
  optNode: FlowNode;
  saveObjectives: (objectives: StudyObjective[]) => Promise<void>;
}

const GOALS: StudyObjective["goal"][] = ["in_range", "minimize", "maximize"];

/** The study's objectives, held on the NW_Optimization node. An objective is a
 *  name, a measurement address (node → output port → optional key) and a
 *  goal; the generator turns each into spec.add_objective(). */
export const ObjectivesSection = ({ nodes, optNode, saveObjectives }: Props) => {
  const objectives = studyObjectives(optNode);
  const declared = useMemo(() => nodeDeclaredObjectives(nodes), [nodes]);
  const nodeById = (id: string) => nodes.find((n) => n.id === id);

  const update = (index: number, patch: Partial<StudyObjective>) =>
    saveObjectives(objectives.map((o, i) => (i === index ? { ...o, ...patch } : o)));
  const remove = (index: number) => saveObjectives(objectives.filter((_, i) => i !== index));

  // --- add form ---------------------------------------------------------------
  const candidateNodes = nodes.filter((n) => !isOptimizationNode(n.data) && outputPorts(n).length > 0);
  const [nodeId, setNodeId] = useState("");
  const [port, setPort] = useState("");
  const [key, setKey] = useState("");
  const [goal, setGoal] = useState<StudyObjective["goal"]>("in_range");
  const [low, setLow] = useState<number | undefined>();
  const [high, setHigh] = useState<number | undefined>();
  const [unit, setUnit] = useState("");
  const [name, setName] = useState("");
  const [nameEdited, setNameEdited] = useState(false);
  const [adding, setAdding] = useState(false);

  const node = nodeById(nodeId);
  const autoName = node && port ? objectiveAutoName(displayName(node), port, key.trim() || undefined) : "";
  const effectiveName = nameEdited ? name.trim() : autoName;
  const duplicate = objectives.some((o) => o.name === effectiveName);
  const needsRange = goal === "in_range";
  const badRange = needsRange && (low === undefined || high === undefined || low >= high);
  const canAdd = !!node && !!port && !!effectiveName && !duplicate && !badRange;

  const add = async () => {
    if (!canAdd || !node) return;
    setAdding(true);
    try {
      const objective: StudyObjective = {
        node_id: node.id,
        port,
        key: key.trim() || undefined,
        name: effectiveName,
        goal,
        low: needsRange || low !== undefined ? low ?? null : null,
        high: needsRange || high !== undefined ? high ?? null : null,
        unit: unit.trim() || undefined,
      };
      await saveObjectives([...objectives, objective]);
      setPort("");
      setKey("");
      setLow(undefined);
      setHigh(undefined);
      setUnit("");
      setName("");
      setNameEdited(false);
    } finally {
      setAdding(false);
    }
  };

  return (
    <Box>
      <Heading size="sm" mb={2}>Objectives</Heading>
      {objectives.length === 0 ? (
        <Text fontSize="sm" color="gray.500" mb={2}>
          No objective yet. Pick the node and output port that produce the value to hit.
        </Text>
      ) : (
        <Table size="sm" variant="simple" mb={3}>
          <Thead>
            <Tr>
              <Th>Name</Th><Th>Measures</Th><Th>Goal</Th><Th>Low</Th><Th>High</Th><Th>Unit</Th><Th />
            </Tr>
          </Thead>
          <Tbody>
            {objectives.map((o, i) => {
              const target = nodeById(o.node_id);
              const inRange = o.goal === "in_range";
              const bad = inRange && (o.low == null || o.high == null || o.low >= o.high);
              return (
                <Tr key={`${o.node_id}.${o.port}.${o.key ?? ""}.${i}`}>
                  <Td>
                    <DraftTextInput
                      value={o.name}
                      w="150px"
                      isInvalid={objectives.some((x, j) => j !== i && x.name === o.name)}
                      onCommit={(v) => v && update(i, { name: v })}
                    />
                  </Td>
                  <Td>
                    {target ? (
                      <code>{measuresLabel(displayName(target), o.port, o.key)}</code>
                    ) : (
                      <Tooltip label="Its node is no longer on the canvas; the generator skips it" hasArrow>
                        <Badge colorScheme="red">node removed</Badge>
                      </Tooltip>
                    )}
                  </Td>
                  <Td>
                    <Select size="sm" w="120px" value={o.goal} onChange={(e) => update(i, { goal: e.target.value as StudyObjective["goal"] })}>
                      {GOALS.map((g) => <option key={g} value={g}>{g}</option>)}
                    </Select>
                  </Td>
                  <Td>
                    <DraftNumberInput value={o.low ?? undefined} allowEmpty={!inRange} isInvalid={bad} onCommit={(v) => update(i, { low: v ?? null })} />
                  </Td>
                  <Td>
                    <DraftNumberInput value={o.high ?? undefined} allowEmpty={!inRange} isInvalid={bad} onCommit={(v) => update(i, { high: v ?? null })} />
                  </Td>
                  <Td>
                    <DraftTextInput value={o.unit ?? ""} w="70px" placeholder="unit" onCommit={(v) => update(i, { unit: v || undefined })} />
                  </Td>
                  <Td>
                    <IconButton aria-label="Remove objective" size="xs" variant="ghost" icon={<DeleteIcon />} onClick={() => remove(i)} />
                  </Td>
                </Tr>
              );
            })}
          </Tbody>
        </Table>
      )}

      <HStack spacing={2} align="flex-end" flexWrap="wrap">
        <Select size="sm" w="160px" placeholder="Node" value={nodeId} onChange={(e) => { setNodeId(e.target.value); setPort(""); }}>
          {candidateNodes.map((n) => <option key={n.id} value={n.id}>{displayName(n)}</option>)}
        </Select>
        <Select size="sm" w="150px" placeholder="Output port" value={port} isDisabled={!node} onChange={(e) => setPort(e.target.value)}>
          {(node ? outputPorts(node) : []).map((p) => <option key={p} value={p}>{p}</option>)}
        </Select>
        <Tooltip label="Key inside a dict output, e.g. the population name: firing_rate_hz.exc" hasArrow>
          <Input size="sm" w="110px" placeholder="key (opt.)" value={key} onChange={(e) => setKey(e.target.value)} />
        </Tooltip>
        <Select size="sm" w="115px" value={goal} onChange={(e) => setGoal(e.target.value as StudyObjective["goal"])}>
          {GOALS.map((g) => <option key={g} value={g}>{g}</option>)}
        </Select>
        <DraftNumberInput value={low} placeholder="low" allowEmpty isInvalid={badRange && !!port} onCommit={setLow} />
        <DraftNumberInput value={high} placeholder="high" allowEmpty isInvalid={badRange && !!port} onCommit={setHigh} />
        <Input size="sm" w="70px" placeholder="unit" value={unit} onChange={(e) => setUnit(e.target.value)} />
        <Input
          size="sm"
          w="170px"
          placeholder="name"
          value={nameEdited ? name : autoName}
          isInvalid={duplicate}
          onChange={(e) => { setName(e.target.value); setNameEdited(true); }}
        />
        <Button size="sm" colorScheme="purple" onClick={add} isDisabled={!canAdd} isLoading={adding}>
          Add objective
        </Button>
      </HStack>
      {duplicate && <Text fontSize="xs" color="red.400" mt={1}>an objective with this name already exists</Text>}
      {port && badRange && <Text fontSize="xs" color="red.400" mt={1}>in_range needs low below high (the target range)</Text>}
      <Text fontSize="xs" color="gray.500" mt={1}>
        A dict output's keys are only known from data; run the workflow once and read the log,
        or type the key. A wrong address is reported by the baseline run with the available ones.
      </Text>

      {declared.length > 0 && (
        <Box mt={3}>
          <Text fontSize="xs" fontWeight="bold" color="gray.600">Declared on nodes</Text>
          <Text fontSize="xs" color="gray.500" mb={1}>
            Objectives a node declares on a parameter (is_objective); the engine discovers them itself.
            Edit them in that node's own panel.
          </Text>
          {declared.map((d) => (
            <Text key={`${d.nodeId}.${d.param}`} fontSize="xs">
              {d.instanceName}.{d.param}: {d.measures || "(no measures address)"}
              {d.range ? ` in [${d.range[0]}, ${d.range[1]}]` : ""} {d.unit}
            </Text>
          ))}
        </Box>
      )}
    </Box>
  );
};
