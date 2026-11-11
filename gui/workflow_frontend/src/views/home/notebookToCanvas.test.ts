import { describe, expect, it } from "vitest";
import {
  connectionAlreadyOnCanvas,
  parseWorkflowSource,
  pickCatalogEntry,
  portOnSchema,
  portsCanConnect,
} from "./notebookToCanvas";

describe("parseWorkflowSource", () => {
  it("reads generated constructors, configure calls, and connections", () => {
    const source = `
from nodes.network.ExcNode import ExcNode
from nodes.simulation.SimNode import SimNode
    exc_pop = ExcNode("exc_pop")
    exc_pop.configure()
    sim = SimNode("sim")
    sim.configure()
    workflow_builder.add_node(exc_pop)
    workflow_builder.connect("exc_pop", "spikes", "sim", "input")
`;
    expect(parseWorkflowSource(source)).toEqual({
      nodes: [
        { className: "ExcNode", category: "network", instanceName: "exc_pop" },
        { className: "SimNode", category: "simulation", instanceName: "sim" },
      ],
      edges: [
        { source: "exc_pop", sourcePort: "spikes", target: "sim", targetPort: "input" },
      ],
    });
  });

  it("ignores a constructor that is not a catalog node import", () => {
    expect(parseWorkflowSource('x = Other("x")').nodes).toEqual([]);
  });
});

describe("pickCatalogEntry", () => {
  it("prefers the class name when another entry shares the label", () => {
    const catalog = [
      { class_name: "Other", label: "ExcNode", schema: { id: "label" } },
      { class_name: "ExcNode", label: "ExcNode", schema: { id: "class" } },
    ];
    expect(pickCatalogEntry(catalog, "ExcNode")?.schema).toEqual({ id: "class" });
  });
});

describe("canvas connections", () => {
  const sourceSchema = { outputs: { spikes: { type: "float" } } };
  const targetSchema = { inputs: { input: { type: "float" }, many: { type: "float", fan_in: true } } };

  it("reads a port that exists and rejects one that does not", () => {
    expect(portOnSchema(sourceSchema, "output", "spikes")).toEqual({
      type: "float",
      fanIn: false,
    });
    expect(portOnSchema(targetSchema, "input", "missing")).toBeNull();
  });

  it("treats the same connection, and a busy target without fan-in, as already present", () => {
    const edges = [
      {
        source: "a",
        target: "b",
        sourceHandle: "a::spikes::output::float",
        targetHandle: "b::input::input::float",
      },
    ];
    expect(
      connectionAlreadyOnCanvas(
        edges,
        "a",
        "b",
        "a::spikes::output::float",
        "b::input::input::float",
        false,
      ),
    ).toBe(true);
    expect(
      connectionAlreadyOnCanvas(
        edges,
        "c",
        "b",
        "c::spikes::output::float",
        "b::input::input::float",
        false,
      ),
    ).toBe(true);
    expect(
      connectionAlreadyOnCanvas(
        edges,
        "c",
        "b",
        "c::spikes::output::float",
        "b::many::input::float",
        true,
      ),
    ).toBe(false);
  });

  it("refuses different port types and accepts the same type in either case", () => {
    expect(portsCanConnect("float", "int")).toBe(false);
    expect(portsCanConnect("float", "Float")).toBe(true);
    expect(portsCanConnect("float", "float")).toBe(true);
  });
});
