export interface ImportedNode {
  className: string;
  category: string;
  instanceName: string;
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
    nodes.push({ className, category, instanceName });
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
