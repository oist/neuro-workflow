import { describe, expect, it } from "vitest";
import type { SchemaFields } from "../type";
import {
  filterNodeCatalog,
  formatParamDefault,
  isPortRequired,
  uniqueCatalogCategories,
  type CatalogNode,
} from "./nodeCatalogSearch";

const schema: SchemaFields = {
  inputs: {
    adjacency_matrix: {
      type: "object",
      description: "Square connectivity matrix",
      optional: false,
    },
  },
  outputs: {
    firing_rate_hz: {
      type: "float",
      description: "Mean rate",
    },
  },
  parameters: {
    n_neurons: {
      type: "int",
      description: "Population size",
      default_value: 100,
    },
  },
  methods: {},
};

const nodes: CatalogNode[] = [
  {
    id: "a",
    label: "AsperaSharesLoaderNode",
    description: "Loads mat file from Aspera Shares",
    category: "I/O",
    file_name: "AsperaSharesLoaderNode.py",
    class_name: "AsperaSharesLoaderNode",
    schema: {
      inputs: {},
      outputs: { mat_data: { type: "object", description: "Loaded mat" } },
      parameters: { remote_path: { type: "string", description: "mat path" } },
      methods: {},
    },
  },
  {
    id: "b",
    label: "NW_Connectivity",
    description: "Build a structural network",
    category: "Network",
    file_name: "NW_Connectivity.py",
    class_name: "NW_Connectivity",
    schema,
  },
];

describe("filterNodeCatalog", () => {
  it("returns all nodes sorted by category then label when the query is empty", () => {
    const result = filterNodeCatalog(nodes, "   ", "all");
    expect(result.map((n) => n.label)).toEqual([
      "AsperaSharesLoaderNode",
      "NW_Connectivity",
    ]);
  });

  it("matches by label", () => {
    const result = filterNodeCatalog(nodes, "aspera");
    expect(result.map((n) => n.label)).toEqual(["AsperaSharesLoaderNode"]);
  });

  it("matches by input port name", () => {
    const result = filterNodeCatalog(nodes, "adjacency_matrix");
    expect(result.map((n) => n.label)).toEqual(["NW_Connectivity"]);
  });

  it("matches by description", () => {
    const result = filterNodeCatalog(nodes, "aspera shares");
    expect(result.map((n) => n.label)).toEqual(["AsperaSharesLoaderNode"]);
  });

  it("filters by category equality", () => {
    const result = filterNodeCatalog(nodes, "", "Network");
    expect(result.map((n) => n.label)).toEqual(["NW_Connectivity"]);
  });
});

describe("uniqueCatalogCategories", () => {
  it("returns sorted unique categories", () => {
    expect(uniqueCatalogCategories(nodes)).toEqual(["I/O", "Network"]);
  });
});

describe("isPortRequired", () => {
  it("treats optional === false as required", () => {
    expect(isPortRequired({ optional: false })).toBe(true);
    expect(isPortRequired({ optional: true })).toBe(false);
    expect(isPortRequired({ required: true })).toBe(true);
  });
});

describe("formatParamDefault", () => {
  it("stringifies non-strings", () => {
    expect(formatParamDefault(100)).toBe("100");
    expect(formatParamDefault(undefined)).toBeNull();
    expect(formatParamDefault("hz")).toBe("hz");
  });
});
