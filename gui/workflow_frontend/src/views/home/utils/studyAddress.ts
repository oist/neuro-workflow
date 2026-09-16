// Pure helpers for the optimization study: which parameters an address can
// point at, which range to propose for one, and how the study on an
// NW_Optimization node relates to the other nodes on the canvas.
//
// The explore side has a single source of truth: the target node's own
// parameter fields (optimizable / optimization_range / unit), exactly what the
// engine reads. The objectives live on the NW_Optimization node as
// data.study.objectives, keyed by node id so a rename cannot break them.
import { Node } from "@xyflow/react";
import { CalculationNodeData, ParameterField, StudyObjective } from "../type";

export const OPTIMIZATION_NODE_LABEL = "NW_Optimization";

export type FlowNode = Node<CalculationNodeData>;

export const isOptimizationNode = (data: CalculationNodeData | undefined): boolean =>
  data?.label === OPTIMIZATION_NODE_LABEL;

const isNumber = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

export const displayName = (node: FlowNode): string =>
  node.data.instanceName || node.data.label || node.id;

/** A parameter, or one numeric key of a dict-valued parameter, a study can explore. */
export interface ExplorableParameter {
  param: string;
  key?: string;
  declaredDefault: number;
  field: ParameterField;
}

export const addressLabel = (param: string, key?: string): string =>
  key ? `${param}.${key}` : param;

/** Numeric scalars and the numeric leaf keys of dict-valued parameters. Strings,
 *  booleans and lists are not offered: an address cannot index them. */
export function explorableParameters(node: FlowNode): ExplorableParameter[] {
  const params = node.data.schema?.parameters ?? {};
  const out: ExplorableParameter[] = [];
  for (const [param, field] of Object.entries(params)) {
    const d = field?.default_value;
    if (isNumber(d)) {
      out.push({ param, declaredDefault: d, field });
    } else if (isPlainObject(d)) {
      for (const [key, v] of Object.entries(d)) {
        if (isNumber(v)) out.push({ param, key, declaredDefault: v, field });
      }
    }
  }
  return out;
}

export const outputPorts = (node: FlowNode): string[] =>
  Object.keys(node.data.schema?.outputs ?? {});

const asPair = (v: unknown): [number, number] | undefined =>
  Array.isArray(v) && v.length === 2 && isNumber(v[0]) && isNumber(v[1])
    ? [v[0], v[1]]
    : undefined;

/** The declared range of a parameter (or of one key of a dict-valued one). */
export function declaredRange(field: ParameterField, key?: string): [number, number] | undefined {
  const r = field.optimization_range;
  if (key) return isPlainObject(r) ? asPair(r[key]) : undefined;
  return asPair(r);
}

/** Mirrors the engine's inference: constraints.integer (per key for a dict),
 *  else the declared default is a whole number. */
export function isIntegerAxis(field: ParameterField, declaredDefault: number, key?: string): boolean {
  const c = field.constraints?.integer;
  if (typeof c === "boolean") return c;
  if (isPlainObject(c) && key && typeof c[key] === "boolean") return c[key];
  return Number.isInteger(declaredDefault);
}

/** Hard bounds the engine clips to. Only a scalar parameter has them: a
 *  constraint belongs to the parameter as a whole and cannot bound one key. */
export function hardBounds(p: ExplorableParameter): { min?: number; max?: number } {
  if (p.key) return {};
  const { min, max } = p.field.constraints ?? {};
  return { min: isNumber(min) ? min : undefined, max: isNumber(max) ? max : undefined };
}

export function clampToBounds(range: [number, number], p: ExplorableParameter): [number, number] {
  const { min, max } = hardBounds(p);
  let [lo, hi] = range;
  if (min !== undefined) lo = Math.max(lo, min);
  if (max !== undefined) hi = Math.min(hi, max);
  return [lo, hi];
}

/** The range to prefill: the author's optimization_range, else constraints
 *  min/max, else the current value ± 50 %, clamped to the constraints. */
export function defaultRangeFor(p: ExplorableParameter): [number, number] {
  const declared = declaredRange(p.field, p.key);
  if (declared) return declared;
  const { min, max } = hardBounds(p);
  if (min !== undefined && max !== undefined) return [min, max];
  const d = p.declaredDefault;
  const guess: [number, number] =
    d === 0 ? [0, 1] : [Math.min(d * 0.5, d * 1.5), Math.max(d * 0.5, d * 1.5)];
  return clampToBounds(guess, p);
}

/** One row of the explore table: a parameter (or dict key) marked optimizable. */
export interface ExploreRow {
  id: string;
  nodeId: string;
  instanceName: string;
  param: string;
  key?: string;
  low?: number;
  high?: number;
  unit: string;
  integer: boolean;
  warning?: string;
}

export const rowId = (nodeId: string, param: string, key?: string): string =>
  `${nodeId}.${addressLabel(param, key)}`;

/** Every parameter marked optimizable on the canvas, in node order. */
export function exploreRows(nodes: FlowNode[]): ExploreRow[] {
  const rows: ExploreRow[] = [];
  for (const node of nodes) {
    if (isOptimizationNode(node.data)) continue;
    const params = node.data.schema?.parameters ?? {};
    const instanceName = displayName(node);
    for (const [param, field] of Object.entries(params)) {
      if (!field || field.optimizable !== true) continue;
      const unit = typeof field.unit === "string" ? field.unit : "";
      const d = field.default_value;
      const r = field.optimization_range;
      if (isPlainObject(d)) {
        const keys = isPlainObject(r) ? Object.keys(r) : [];
        if (keys.length === 0) {
          rows.push({
            id: rowId(node.id, param), nodeId: node.id, instanceName, param, unit,
            integer: false,
            warning: "a dict-valued parameter needs a range per key; remove and add a key",
          });
          continue;
        }
        for (const key of keys) {
          const pair = asPair((r as Record<string, unknown>)[key]);
          const declaredDefault = isNumber(d[key]) ? d[key] : undefined;
          rows.push({
            id: rowId(node.id, param, key), nodeId: node.id, instanceName, param, key,
            low: pair?.[0], high: pair?.[1], unit,
            integer: declaredDefault !== undefined && isIntegerAxis(field, declaredDefault, key),
            warning: declaredDefault === undefined
              ? "this key is not in the parameter's current value"
              : pair ? undefined : "range is not [low, high]",
          });
        }
      } else {
        const pair = asPair(r);
        rows.push({
          id: rowId(node.id, param), nodeId: node.id, instanceName, param,
          low: pair?.[0], high: pair?.[1], unit,
          integer: isNumber(d) && isIntegerAxis(field, d),
          warning: pair
            ? undefined
            : isNumber(field.constraints?.min) && isNumber(field.constraints?.max)
              ? undefined // the engine falls back to constraints.min/max
              : "no range; the engine will skip this parameter",
        });
      }
    }
  }
  return rows;
}

export const objectiveAutoName = (instanceName: string, port: string, key?: string): string =>
  `${instanceName}_${port}${key ? `_${key}` : ""}`.replace(/[^A-Za-z0-9_]/g, "_");

export const measuresLabel = (instanceName: string, port: string, key?: string): string =>
  `${instanceName}.${port}${key ? `.${key}` : ""}`;

export const studyObjectives = (node: FlowNode | undefined): StudyObjective[] =>
  node?.data.study?.objectives ?? [];

/** Ids of the nodes some NW_Optimization node's objectives measure. */
export function targetedNodeIds(nodes: FlowNode[]): Set<string> {
  const ids = new Set<string>();
  for (const node of nodes) {
    if (!isOptimizationNode(node.data)) continue;
    for (const o of studyObjectives(node)) ids.add(o.node_id);
  }
  return ids;
}

/** Objectives a node author declared on a parameter (is_objective); the engine
 *  discovers them itself, the study only shows them. */
export interface NodeDeclaredObjective {
  nodeId: string;
  instanceName: string;
  param: string;
  range?: [number, number];
  measures: string;
  unit: string;
}

export function nodeDeclaredObjectives(nodes: FlowNode[]): NodeDeclaredObjective[] {
  const out: NodeDeclaredObjective[] = [];
  for (const node of nodes) {
    if (isOptimizationNode(node.data)) continue;
    for (const [param, field] of Object.entries(node.data.schema?.parameters ?? {})) {
      if (!field || field.is_objective !== true) continue;
      out.push({
        nodeId: node.id, instanceName: displayName(node), param,
        range: asPair(field.objective_range),
        measures: typeof field.measures === "string" ? field.measures : "",
        unit: typeof field.unit === "string" ? field.unit : "",
      });
    }
  }
  return out;
}
