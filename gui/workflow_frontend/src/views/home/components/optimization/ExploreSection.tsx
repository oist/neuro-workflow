import { useMemo, useState } from "react";
import {
  Badge, Box, Button, HStack, IconButton, Select, Table, Tbody, Td, Text, Th, Thead, Tr,
  Tooltip, Input, Heading,
} from "@chakra-ui/react";
import { DeleteIcon } from "@chakra-ui/icons";
import {
  ExplorableParameter, ExploreRow, FlowNode, addressLabel, clampToBounds, defaultRangeFor,
  displayName, explorableParameters, exploreRows, hardBounds, isOptimizationNode, rowId,
} from "../../utils/studyAddress";
import { DraftNumberInput, DraftTextInput } from "./draftInputs";
import { ParameterFieldName } from "./studyApi";

export type SetParamField = (
  nodeId: string, param: string, field: ParameterFieldName, value: unknown
) => Promise<void>;

interface Props {
  nodes: FlowNode[];
  setParamField: SetParamField;
}

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const rangeDict = (node: FlowNode | undefined, param: string): Record<string, unknown> => {
  const r = node?.data.schema?.parameters?.[param]?.optimization_range;
  return isPlainObject(r) ? { ...r } : {};
};

/** The parameters explored by the study: one row per parameter (or dict key)
 *  marked optimizable on any node. Edits go straight to that node's fields. */
export const ExploreSection = ({ nodes, setParamField }: Props) => {
  const rows = useMemo(() => exploreRows(nodes), [nodes]);
  const explored = useMemo(() => new Set(rows.map((r) => r.id)), [rows]);
  const nodeById = (id: string) => nodes.find((n) => n.id === id);

  const commitRange = async (row: ExploreRow, low: number | undefined, high: number | undefined) => {
    if (low === undefined || high === undefined) return;
    if (row.key) {
      const dict = rangeDict(nodeById(row.nodeId), row.param);
      dict[row.key] = [low, high];
      await setParamField(row.nodeId, row.param, "optimization_range", dict);
    } else {
      await setParamField(row.nodeId, row.param, "optimization_range", [low, high]);
    }
  };

  const removeRow = async (row: ExploreRow) => {
    if (row.key) {
      const dict = rangeDict(nodeById(row.nodeId), row.param);
      delete dict[row.key];
      // The endpoint refuses null, so an emptied dict is written as {}
      await setParamField(row.nodeId, row.param, "optimization_range", dict);
      if (Object.keys(dict).length > 0) return;
    }
    await setParamField(row.nodeId, row.param, "optimizable", false);
  };

  // --- add form ---------------------------------------------------------------
  const candidateNodes = nodes.filter((n) => !isOptimizationNode(n.data));
  const [nodeId, setNodeId] = useState("");
  const [choice, setChoice] = useState("");
  const [low, setLow] = useState<number | undefined>();
  const [high, setHigh] = useState<number | undefined>();
  const [unit, setUnit] = useState("");
  const [adding, setAdding] = useState(false);

  const candidates: ExplorableParameter[] = useMemo(() => {
    const node = nodeById(nodeId);
    if (!node) return [];
    return explorableParameters(node).filter((p) => !explored.has(rowId(node.id, p.param, p.key)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeId, nodes, explored]);
  const selected = candidates.find((p) => addressLabel(p.param, p.key) === choice);

  const pick = (label: string) => {
    setChoice(label);
    const p = candidates.find((c) => addressLabel(c.param, c.key) === label);
    if (!p) return;
    const [lo, hi] = defaultRangeFor(p);
    setLow(lo);
    setHigh(hi);
    setUnit(typeof p.field.unit === "string" ? p.field.unit : "");
  };

  const bounds = selected ? hardBounds(selected) : {};
  const outsideBounds =
    selected !== undefined && low !== undefined && high !== undefined &&
    ((bounds.min !== undefined && low < bounds.min) || (bounds.max !== undefined && high > bounds.max));
  const badOrder = low !== undefined && high !== undefined && low >= high;
  const canAdd = selected !== undefined && low !== undefined && high !== undefined && !badOrder && !outsideBounds;

  const add = async () => {
    if (!selected || low === undefined || high === undefined) return;
    setAdding(true);
    try {
      const range: [number, number] = clampToBounds([low, high], selected);
      await setParamField(nodeId, selected.param, "optimizable", true);
      if (selected.key) {
        const dict = rangeDict(nodeById(nodeId), selected.param);
        dict[selected.key] = range;
        await setParamField(nodeId, selected.param, "optimization_range", dict);
      } else {
        await setParamField(nodeId, selected.param, "optimization_range", range);
      }
      const declaredUnit = typeof selected.field.unit === "string" ? selected.field.unit : "";
      if (unit.trim() && unit.trim() !== declaredUnit) {
        await setParamField(nodeId, selected.param, "unit", unit.trim());
      }
      setChoice("");
      setLow(undefined);
      setHigh(undefined);
      setUnit("");
    } finally {
      setAdding(false);
    }
  };

  return (
    <Box>
      <Heading size="sm" mb={2}>Parameters to explore</Heading>
      {rows.length === 0 ? (
        <Text fontSize="sm" color="gray.500" mb={2}>
          Nothing is explored yet. Pick a node and a numeric parameter below.
        </Text>
      ) : (
        <Table size="sm" variant="simple" mb={3}>
          <Thead>
            <Tr>
              <Th>Node</Th><Th>Parameter</Th><Th>Low</Th><Th>High</Th><Th>Unit</Th><Th /><Th />
            </Tr>
          </Thead>
          <Tbody>
            {rows.map((row) => (
              <Tr key={row.id}>
                <Td>{row.instanceName}</Td>
                <Td><code>{addressLabel(row.param, row.key)}</code></Td>
                <Td>
                  <DraftNumberInput
                    value={row.low}
                    isInvalid={row.low !== undefined && row.high !== undefined && row.low >= row.high}
                    onCommit={(v) => commitRange(row, v, row.high)}
                  />
                </Td>
                <Td>
                  <DraftNumberInput
                    value={row.high}
                    isInvalid={row.low !== undefined && row.high !== undefined && row.low >= row.high}
                    onCommit={(v) => commitRange(row, row.low, v)}
                  />
                </Td>
                <Td>
                  <DraftTextInput
                    value={row.unit}
                    w="70px"
                    placeholder="unit"
                    onCommit={(v) => setParamField(row.nodeId, row.param, "unit", v)}
                  />
                </Td>
                <Td>
                  <HStack spacing={1}>
                    {row.integer && <Badge colorScheme="blue" fontSize="9px">integer</Badge>}
                    {row.warning && (
                      <Tooltip label={row.warning} hasArrow>
                        <Badge colorScheme="red" fontSize="9px">check</Badge>
                      </Tooltip>
                    )}
                  </HStack>
                </Td>
                <Td>
                  <IconButton
                    aria-label="Stop exploring this parameter"
                    size="xs"
                    variant="ghost"
                    icon={<DeleteIcon />}
                    onClick={() => removeRow(row)}
                  />
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
      )}

      <HStack spacing={2} align="flex-end" flexWrap="wrap">
        <Select
          size="sm"
          w="180px"
          placeholder="Node"
          value={nodeId}
          onChange={(e) => { setNodeId(e.target.value); setChoice(""); }}
        >
          {candidateNodes.map((n) => (
            <option key={n.id} value={n.id}>{displayName(n)}</option>
          ))}
        </Select>
        <Select
          size="sm"
          w="200px"
          placeholder={nodeId ? (candidates.length ? "Parameter" : "No numeric parameter left") : "Parameter"}
          value={choice}
          isDisabled={!nodeId || candidates.length === 0}
          onChange={(e) => pick(e.target.value)}
        >
          {candidates.map((p) => {
            const label = addressLabel(p.param, p.key);
            return <option key={label} value={label}>{label} = {p.declaredDefault}</option>;
          })}
        </Select>
        <DraftNumberInput value={low} placeholder="low" isInvalid={badOrder || outsideBounds} onCommit={setLow} />
        <DraftNumberInput value={high} placeholder="high" isInvalid={badOrder || outsideBounds} onCommit={setHigh} />
        <Input size="sm" w="70px" placeholder="unit" value={unit} onChange={(e) => setUnit(e.target.value)} />
        <Button size="sm" colorScheme="yellow" onClick={add} isDisabled={!canAdd} isLoading={adding}>
          Add parameter
        </Button>
      </HStack>
      {badOrder && <Text fontSize="xs" color="red.400" mt={1}>low must be below high</Text>}
      {outsideBounds && (
        <Text fontSize="xs" color="red.400" mt={1}>
          outside the parameter's constraints [{bounds.min ?? "-∞"}, {bounds.max ?? "∞"}]
        </Text>
      )}
    </Box>
  );
};
