import { describe, expect, it } from "vitest";
import {
  countPaletteByScope,
  filterPaletteNodes,
  type PaletteNode,
} from "./paletteFilter";

const nodes: PaletteNode[] = [
  {
    label: "Foo",
    description: "owner analysis node",
    file_name: "foo.py",
    category: "Analysis",
    category_key: "analysis",
    is_own: true,
    parse_ok: true,
    draggable: true,
  },
  {
    label: "Cat",
    description: "catalog io node",
    file_name: "cat.py",
    category: "I/O",
    category_key: "io",
    is_own: false,
    parse_ok: true,
    draggable: true,
  },
  {
    label: "broken",
    description: "No NODE_DEFINITION found — this file is not a palette node.",
    file_name: "broken.py",
    category: "Analysis",
    category_key: "analysis",
    is_own: true,
    parse_ok: false,
    draggable: false,
  },
];

describe("countPaletteByScope", () => {
  it("splits mine vs shared", () => {
    expect(countPaletteByScope(nodes)).toEqual({ all: 3, mine: 2, shared: 1 });
  });
});

describe("filterPaletteNodes", () => {
  it("returns all nodes for scope all", () => {
    const result = filterPaletteNodes(nodes, { scope: "all", query: "" });
    expect(result.map((n) => n.label)).toEqual(["Foo", "Cat", "broken"]);
  });

  it("keeps only is_own for mine, including parse_ok false", () => {
    const result = filterPaletteNodes(nodes, { scope: "mine", query: "" });
    expect(result.map((n) => n.label)).toEqual(["Foo", "broken"]);
    expect(result.some((n) => n.parse_ok === false)).toBe(true);
  });

  it("keeps only shared catalog nodes", () => {
    const result = filterPaletteNodes(nodes, { scope: "shared", query: "" });
    expect(result.map((n) => n.label)).toEqual(["Cat"]);
  });

  it("matches category display and key", () => {
    const byDisplay = filterPaletteNodes(nodes, { scope: "all", query: "Analysis" });
    expect(byDisplay.map((n) => n.label)).toEqual(["Foo", "broken"]);
    const byKey = filterPaletteNodes(nodes, { scope: "all", query: "I/O" });
    expect(byKey.map((n) => n.label)).toEqual(["Cat"]);
  });
});
