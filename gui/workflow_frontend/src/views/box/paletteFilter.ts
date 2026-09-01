export type PaletteScope = "all" | "mine" | "shared";

export type PaletteNode = {
  label: string;
  description?: string;
  file_name?: string;
  category?: string;
  category_key?: string;
  is_own?: boolean;
  parse_ok?: boolean;
  draggable?: boolean;
};

export function countPaletteByScope(nodes: PaletteNode[]): {
  all: number;
  mine: number;
  shared: number;
} {
  let mine = 0;
  let shared = 0;
  for (const node of nodes) {
    if (node.is_own) {
      mine += 1;
    } else {
      shared += 1;
    }
  }
  return { all: nodes.length, mine, shared };
}

export function filterPaletteNodes<T extends PaletteNode>(
  nodes: T[],
  opts: { scope: PaletteScope; query: string }
): T[] {
  const q = opts.query.trim().toLowerCase();
  return nodes.filter((node) => {
    if (opts.scope === "mine" && !node.is_own) {
      return false;
    }
    if (opts.scope === "shared" && node.is_own) {
      return false;
    }
    if (!q) {
      return true;
    }
    const hay = [
      node.label,
      node.description,
      node.file_name,
      node.category,
      node.category_key,
    ]
      .filter((part): part is string => Boolean(part))
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });
}
