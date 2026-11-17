export type ConfigureValue =
  | string
  | number
  | boolean
  | null
  | ConfigureValue[]
  | { [key: string]: ConfigureValue };

export interface ImportedNode {
  className: string;
  category: string;
  instanceName: string;
  parameters: Record<string, ConfigureValue>;
}

export interface ImportedEdge {
  source: string;
  sourcePort: string;
  target: string;
  targetPort: string;
}

const IMPORT_RE =
  /from\s+nodes\.([A-Za-z0-9_]+)\.([A-Za-z_][A-Za-z0-9_]*)\s+import\s+\2\b/g;
const CONSTRUCTOR_RE =
  /([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\(\s*["']([^"']+)["']\s*\)/g;
const CONNECT_RE =
  /workflow_builder\.connect\(\s*["']([^"']+)["']\s*,\s*["']([^"']+)["']\s*,\s*["']([^"']+)["']\s*,\s*["']([^"']+)["']\s*\)/g;

function skipSpace(source: string, index: number): number {
  let cursor = index;
  while (cursor < source.length && /\s/.test(source[cursor])) {
    cursor += 1;
  }
  return cursor;
}

function isIdentifierChar(char: string | undefined): boolean {
  return Boolean(char && /[A-Za-z0-9_]/.test(char));
}

function parseString(source: string, index: number): [string, number] | null {
  const quote = source[index];
  if (quote !== "'" && quote !== '"') {
    return null;
  }
  let text = "";
  let cursor = index + 1;
  while (cursor < source.length) {
    const char = source[cursor];
    if (char === "\\") {
      const escaped = source[cursor + 1] ?? "";
      const decoded: Record<string, string> = {
        n: "\n",
        t: "\t",
        r: "\r",
        "\\": "\\",
        "'": "'",
        '"': '"',
      };
      text += decoded[escaped] ?? escaped;
      cursor += 2;
      continue;
    }
    if (char === quote) {
      return [text, cursor + 1];
    }
    text += char;
    cursor += 1;
  }
  return null;
}

function parseNumber(source: string, index: number): [number, number] | null {
  const match = /^-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?/.exec(source.slice(index));
  if (!match) {
    return null;
  }
  return [Number(match[0]), index + match[0].length];
}

function parseLiteral(source: string, index: number): [ConfigureValue, number] | null {
  const cursor = skipSpace(source, index);
  if (source.startsWith("True", cursor) && !isIdentifierChar(source[cursor + 4])) {
    return [true, cursor + 4];
  }
  if (source.startsWith("False", cursor) && !isIdentifierChar(source[cursor + 5])) {
    return [false, cursor + 5];
  }
  if (source.startsWith("None", cursor) && !isIdentifierChar(source[cursor + 4])) {
    return [null, cursor + 4];
  }
  const text = parseString(source, cursor);
  if (text) {
    return text;
  }
  const number = parseNumber(source, cursor);
  if (number) {
    return number;
  }
  if (source[cursor] === "[") {
    return parseList(source, cursor);
  }
  if (source[cursor] === "{") {
    return parseDict(source, cursor);
  }
  return null;
}

function parseList(source: string, index: number): [ConfigureValue[], number] | null {
  const items: ConfigureValue[] = [];
  let cursor = index + 1;
  while (cursor < source.length) {
    cursor = skipSpace(source, cursor);
    if (source[cursor] === "]") {
      return [items, cursor + 1];
    }
    const item = parseLiteral(source, cursor);
    if (!item) {
      return null;
    }
    items.push(item[0]);
    cursor = skipSpace(source, item[1]);
    if (source[cursor] === ",") {
      cursor += 1;
    }
  }
  return null;
}

function parseDict(source: string, index: number): [Record<string, ConfigureValue>, number] | null {
  const record: Record<string, ConfigureValue> = {};
  let cursor = index + 1;
  while (cursor < source.length) {
    cursor = skipSpace(source, cursor);
    if (source[cursor] === "}") {
      return [record, cursor + 1];
    }
    const quoted = parseString(source, cursor);
    let key = "";
    if (quoted) {
      key = quoted[0];
      cursor = quoted[1];
    } else {
      const bare = /^([A-Za-z_][A-Za-z0-9_]*)/.exec(source.slice(cursor));
      if (!bare) {
        return null;
      }
      key = bare[1];
      cursor += bare[1].length;
    }
    cursor = skipSpace(source, cursor);
    if (source[cursor] !== ":") {
      return null;
    }
    const value = parseLiteral(source, cursor + 1);
    if (!value) {
      return null;
    }
    record[key] = value[0];
    cursor = skipSpace(source, value[1]);
    if (source[cursor] === ",") {
      cursor += 1;
    }
  }
  return null;
}

function readArgument(source: string, index: number): [ConfigureValue, number] | null {
  const cursor = skipSpace(source, index);
  if (source.startsWith("lambda", cursor) && !isIdentifierChar(source[cursor + 6])) {
    let depth = 0;
    let end = cursor;
    while (end < source.length) {
      const char = source[end];
      if (char === "'" || char === '"') {
        const text = parseString(source, end);
        if (!text) {
          return null;
        }
        end = text[1];
        continue;
      }
      if (char === "(" || char === "[" || char === "{") {
        depth += 1;
      } else if (char === ")" || char === "]" || char === "}") {
        if (depth === 0) {
          break;
        }
        depth -= 1;
      } else if (char === "," && depth === 0) {
        break;
      }
      end += 1;
    }
    return [source.slice(cursor, end).trim(), end];
  }
  return parseLiteral(source, cursor);
}

function parseConfigureArguments(body: string): Record<string, ConfigureValue> {
  const values: Record<string, ConfigureValue> = {};
  let cursor = 0;
  while (cursor < body.length) {
    cursor = skipSpace(body, cursor);
    if (cursor >= body.length) {
      break;
    }
    const key = /^([A-Za-z_][A-Za-z0-9_]*)\s*=/.exec(body.slice(cursor));
    if (!key) {
      break;
    }
    cursor += key[0].length;
    const parsed = readArgument(body, cursor);
    if (!parsed) {
      break;
    }
    values[key[1]] = parsed[0];
    cursor = skipSpace(body, parsed[1]);
    if (body[cursor] === ",") {
      cursor += 1;
    }
  }
  return values;
}

function extractCallBody(source: string, openParen: number): string | null {
  let depth = 0;
  let cursor = openParen;
  while (cursor < source.length) {
    const char = source[cursor];
    if (char === "'" || char === '"') {
      const text = parseString(source, cursor);
      if (!text) {
        return null;
      }
      cursor = text[1];
      continue;
    }
    if (char === "(") {
      depth += 1;
    } else if (char === ")") {
      depth -= 1;
      if (depth === 0) {
        return source.slice(openParen + 1, cursor);
      }
    }
    cursor += 1;
  }
  return null;
}

export function parseWorkflowSource(source: string): {
  nodes: ImportedNode[];
  edges: ImportedEdge[];
} {
  const classes = new Map<string, string>();
  for (const match of source.matchAll(IMPORT_RE)) {
    classes.set(match[2], match[1]);
  }

  const nodes: ImportedNode[] = [];
  const seen = new Set<string>();
  const variableToInstance = new Map<string, string>();
  for (const match of source.matchAll(CONSTRUCTOR_RE)) {
    const className = match[2];
    const category = classes.get(className);
    if (!category) {
      continue;
    }
    const instanceName = match[3];
    if (seen.has(instanceName)) {
      continue;
    }
    seen.add(instanceName);
    variableToInstance.set(match[1], instanceName);
    nodes.push({ className, category, instanceName, parameters: {} });
  }

  const configureRe = /([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*configure\s*\(/g;
  for (const match of source.matchAll(configureRe)) {
    const instanceName = variableToInstance.get(match[1]);
    const body = extractCallBody(source, match.index + match[0].length - 1);
    if (!instanceName || body === null) {
      continue;
    }
    const node = nodes.find((item) => item.instanceName === instanceName);
    if (!node) {
      continue;
    }
    node.parameters = { ...node.parameters, ...parseConfigureArguments(body) };
  }

  const edges: ImportedEdge[] = [];
  for (const match of source.matchAll(CONNECT_RE)) {
    edges.push({
      source: match[1],
      sourcePort: match[2],
      target: match[3],
      targetPort: match[4],
    });
  }
  return { nodes, edges };
}

export function nextInstanceName(className: string, usedNames: Iterable<string>): string {
  const used = new Set(usedNames);
  let number = 1;
  while (used.has(`${className}_${number}`)) {
    number += 1;
  }
  return `${className}_${number}`;
}

export function replacesGeneratedName(className: string, instanceName: string): boolean {
  if (instanceName === className) {
    return true;
  }
  const escaped = className.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^instance_${escaped}_\\d+$`).test(instanceName);
}

export function assignImportedNames(
  nodes: ImportedNode[],
  edges: ImportedEdge[],
  usedNames: Iterable<string>,
): { nodes: ImportedNode[]; edges: ImportedEdge[] } {
  const used = new Set(usedNames);
  const renamed = new Map<string, string>();
  const assigned = nodes.map((node) => {
    if (!replacesGeneratedName(node.className, node.instanceName)) {
      used.add(node.instanceName);
      return node;
    }
    const instanceName = nextInstanceName(node.className, used);
    used.add(instanceName);
    renamed.set(node.instanceName, instanceName);
    return { ...node, instanceName };
  });
  const nameOf = (name: string) => renamed.get(name) ?? name;
  return {
    nodes: assigned,
    edges: edges.map((edge) => ({
      ...edge,
      source: nameOf(edge.source),
      target: nameOf(edge.target),
    })),
  };
}

export interface ParameterModification {
  is_modified: boolean;
  field_modifications: { default_value_original: unknown };
}

export function applyConfigureValues(
  schema: unknown,
  values: Record<string, ConfigureValue>,
): {
  schema: unknown;
  parameterModifications: Record<string, ParameterModification>;
  unknown: string[];
} {
  const copy = schema && typeof schema === "object" ? JSON.parse(JSON.stringify(schema)) : {};
  const parameters =
    copy.parameters && typeof copy.parameters === "object"
      ? (copy.parameters as Record<string, { default_value?: unknown }>)
      : {};
  const parameterModifications: Record<string, ParameterModification> = {};
  const unknown: string[] = [];
  for (const [key, value] of Object.entries(values)) {
    const parameter = parameters[key];
    if (!parameter || typeof parameter !== "object") {
      unknown.push(key);
      continue;
    }
    const original = parameter.default_value;
    parameter.default_value = value;
    parameterModifications[key] = {
      is_modified: true,
      field_modifications: { default_value_original: original },
    };
  }
  return { schema: copy, parameterModifications, unknown };
}

export interface CatalogEntry {
  class_name?: string;
  label?: string;
  schema?: unknown;
}

export function pickCatalogEntry<T extends CatalogEntry>(
  catalog: T[],
  className: string,
): T | undefined {
  return (
    catalog.find((node) => node.class_name === className) ??
    catalog.find((node) => node.label === className)
  );
}

type PortInfo = { type?: string; fan_in?: boolean };

function portTable(schema: unknown, direction: "input" | "output"): Record<string, PortInfo> {
  if (!schema || typeof schema !== "object") {
    return {};
  }
  const key = direction === "input" ? "inputs" : "outputs";
  const table = (schema as { inputs?: unknown; outputs?: unknown })[key];
  if (!table || typeof table !== "object") {
    return {};
  }
  return table as Record<string, PortInfo>;
}

export function portOnSchema(
  schema: unknown,
  direction: "input" | "output",
  portName: string,
): { type: string; fanIn: boolean } | null {
  const port = portTable(schema, direction)[portName];
  if (!port || typeof port !== "object") {
    return null;
  }
  return { type: port.type || "any", fanIn: port.fan_in === true };
}

export interface CanvasEdgeRef {
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
}

export function portsCanConnect(sourceType: string, targetType: string): boolean {
  return sourceType.toUpperCase() === targetType.toUpperCase();
}

export function connectionAlreadyOnCanvas(
  edges: CanvasEdgeRef[],
  sourceId: string,
  targetId: string,
  sourceHandle: string,
  targetHandle: string,
  targetAllowsFanIn: boolean,
): boolean {
  const sameConnection = edges.some(
    (edge) =>
      edge.source === sourceId &&
      edge.target === targetId &&
      edge.sourceHandle === sourceHandle &&
      edge.targetHandle === targetHandle,
  );
  if (sameConnection) {
    return true;
  }
  if (targetAllowsFanIn) {
    return false;
  }
  return edges.some(
    (edge) => edge.target === targetId && edge.targetHandle === targetHandle,
  );
}
