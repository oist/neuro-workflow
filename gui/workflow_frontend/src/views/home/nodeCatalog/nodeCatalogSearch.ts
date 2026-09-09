import type { InputField, OutputField, ParameterField, SchemaFields } from "../type";

export type CatalogNode = {
  id?: string;
  label: string;
  description: string;
  category: string;
  file_name: string;
  class_name: string;
  schema?: SchemaFields | null;
};

const ALL_CATEGORIES = "all";

export function nodeCatalogKey(node: CatalogNode): string {
  if (node.id) {
    return node.id;
  }
  return `${node.class_name}::${node.file_name}`;
}

function fieldHaystack(
  name: string,
  field: InputField | OutputField | ParameterField | undefined
): string {
  if (!field) {
    return name;
  }
  return [name, field.type ?? "", field.description ?? ""].join(" ");
}

export function nodeCatalogHaystack(node: CatalogNode): string {
  const parts: string[] = [
    node.label ?? "",
    node.description ?? "",
    node.category ?? "",
    node.file_name ?? "",
    node.class_name ?? "",
  ];
  const schema = node.schema;
  if (schema) {
    for (const [name, field] of Object.entries(schema.inputs ?? {})) {
      parts.push(fieldHaystack(name, field));
    }
    for (const [name, field] of Object.entries(schema.outputs ?? {})) {
      parts.push(fieldHaystack(name, field));
    }
    for (const [name, field] of Object.entries(schema.parameters ?? {})) {
      parts.push(fieldHaystack(name, field));
    }
  }
  return parts.join(" ").toLowerCase();
}

export function filterNodeCatalog(
  nodes: CatalogNode[],
  query: string,
  category: string = ALL_CATEGORIES
): CatalogNode[] {
  const q = query.trim().toLowerCase();
  const filtered = nodes.filter((node) => {
    if (category && category !== ALL_CATEGORIES && node.category !== category) {
      return false;
    }
    if (!q) {
      return true;
    }
    return nodeCatalogHaystack(node).includes(q);
  });
  return [...filtered].sort((a, b) => {
    const byCategory = (a.category || "").localeCompare(b.category || "");
    if (byCategory !== 0) {
      return byCategory;
    }
    return (a.label || "").localeCompare(b.label || "");
  });
}

export function uniqueCatalogCategories(nodes: CatalogNode[]): string[] {
  const seen = new Set<string>();
  for (const node of nodes) {
    if (node.category) {
      seen.add(node.category);
    }
  }
  return [...seen].sort((a, b) => a.localeCompare(b));
}

export function groupNodesByCategory(
  nodes: CatalogNode[]
): Array<[string, CatalogNode[]]> {
  const map = new Map<string, CatalogNode[]>();
  for (const node of nodes) {
    const category = node.category || "Uncategorized";
    const list = map.get(category);
    if (list) {
      list.push(node);
    } else {
      map.set(category, [node]);
    }
  }
  return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

export function isPortRequired(field: {
  required?: boolean;
  optional?: boolean;
}): boolean {
  if (field.optional === false) {
    return true;
  }
  return field.required === true;
}

export function formatParamDefault(value: unknown): string | null {
  if (value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
