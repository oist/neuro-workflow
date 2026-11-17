import { describe, expect, it } from "vitest";
import {
  applyConfigureValues,
  assignImportedNames,
  canvasNodeForImport,
  connectionAlreadyOnCanvas,
  nextInstanceName,
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
        { className: "ExcNode", category: "network", instanceName: "exc_pop", parameters: {} },
        { className: "SimNode", category: "simulation", instanceName: "sim", parameters: {} },
      ],
      edges: [
        { source: "exc_pop", sourcePort: "spikes", target: "sim", targetPort: "input" },
      ],
    });
  });

  it("ignores a constructor that is not a catalog node import", () => {
    expect(parseWorkflowSource('x = Other("x")').nodes).toEqual([]);
  });

  it("reads configure values, including a lambda as source text", () => {
    const source = `
from nodes.network.NW_Population import NW_Population
pop = NW_Population("exc_pop")
pop.configure(N=100, name="v1", flag=True, items=[1, 2], table={"a": 1}, rule=lambda x: x + 1)
`;
    expect(parseWorkflowSource(source).nodes[0].parameters).toEqual({
      N: 100,
      name: "v1",
      flag: true,
      items: [1, 2],
      table: { a: 1 },
      rule: "lambda x: x + 1",
    });
  });
});

describe("instance names", () => {
  it("uses the smallest free Class_n", () => {
    expect(nextInstanceName("NW_Population", [])).toBe("NW_Population_1");
    expect(nextInstanceName("NW_Population", ["NW_Population_1"])).toBe("NW_Population_2");
  });

  it("replaces a class name and a generated instance name, and the connections follow", () => {
    const nodes = [
      {
        className: "NW_Population",
        category: "network",
        instanceName: "NW_Population",
        parameters: {},
      },
      {
        className: "SimNode",
        category: "simulation",
        instanceName: "instance_SimNode_002",
        parameters: {},
      },
      {
        className: "ExcNode",
        category: "network",
        instanceName: "exc_pop",
        parameters: {},
      },
    ];
    const edges = [
      {
        source: "NW_Population",
        sourcePort: "spikes",
        target: "instance_SimNode_002",
        targetPort: "input",
      },
    ];
    expect(assignImportedNames(nodes, edges, ["NW_Population_1"])).toEqual({
      nodes: [
        { ...nodes[0], instanceName: "NW_Population_2" },
        { ...nodes[1], instanceName: "SimNode_1" },
        nodes[2],
      ],
      edges: [
        {
          source: "NW_Population_2",
          sourcePort: "spikes",
        target: "SimNode_1",
        targetPort: "input",
      },
    ],
    });
  });

  it("keeps an old generated name on the class-named node at that saved position", () => {
    const canvas = [
      { instanceName: "NW_Population", className: "NW_Population" },
      { instanceName: "exc_pop", className: "ExcNode" },
      { instanceName: "SimNode", className: "SimNode" },
    ];
    const nodes = [
      {
        className: "NW_Population",
        category: "network",
        instanceName: "instance_NW_Population_001",
        parameters: {},
      },
      {
        className: "SimNode",
        category: "simulation",
        instanceName: "instance_SimNode_003",
        parameters: {},
      },
      {
        className: "Other",
        category: "analysis",
        instanceName: "instance_Other_002",
        parameters: {},
      },
    ];
    const edges = [
      {
        source: "instance_NW_Population_001",
        sourcePort: "spikes",
        target: "instance_SimNode_003",
        targetPort: "input",
      },
    ];
    expect(
      assignImportedNames(
        nodes,
        edges,
        canvas.map((node) => node.instanceName),
        canvas,
      ),
    ).toEqual({
      nodes: [
        { ...nodes[0], instanceName: "NW_Population" },
        { ...nodes[1], instanceName: "SimNode" },
        { ...nodes[2], instanceName: "Other_1" },
      ],
      edges: [
        {
          source: "NW_Population",
          sourcePort: "spikes",
          target: "SimNode",
          targetPort: "input",
          sourceCanvasIndex: 0,
          targetCanvasIndex: 2,
        },
      ],
    });
  });

  it("keeps each class-named node when several share that name", () => {
    const canvas = [
      { instanceName: "NW_Population", className: "NW_Population" },
      { instanceName: "exc_pop", className: "ExcNode" },
      { instanceName: "NW_Population", className: "NW_Population" },
    ];
    const nodes = [
      {
        className: "NW_Population",
        category: "network",
        instanceName: "instance_NW_Population_001",
        parameters: {},
      },
      {
        className: "NW_Population",
        category: "network",
        instanceName: "instance_NW_Population_003",
        parameters: {},
      },
    ];
    const edges = [
      {
        source: "instance_NW_Population_001",
        sourcePort: "spikes",
        target: "instance_NW_Population_003",
        targetPort: "input",
      },
    ];
    const assigned = assignImportedNames(
      nodes,
      edges,
      canvas.map((node) => node.instanceName),
      canvas,
    );
    expect(assigned).toEqual({
      nodes: [
        { ...nodes[0], instanceName: "NW_Population" },
        { ...nodes[1], instanceName: "NW_Population" },
      ],
      edges: [
        {
          source: "NW_Population",
          sourcePort: "spikes",
          target: "NW_Population",
          targetPort: "input",
          sourceCanvasIndex: 0,
          targetCanvasIndex: 2,
        },
      ],
    });
    const canvasNodes = canvas.map((node, index) => ({
      id: `n${index}`,
      data: { instanceName: node.instanceName },
    }));
    const edge = assigned.edges[0];
    expect(canvasNodeForImport(canvasNodes, edge.source, edge.sourceCanvasIndex)?.id).toBe("n0");
    expect(canvasNodeForImport(canvasNodes, edge.target, edge.targetCanvasIndex)?.id).toBe("n2");
  });
});

describe("applyConfigureValues", () => {
  it("stores known parameters and reports a keyword the node does not have", () => {
    const schema = {
      parameters: {
        N: { default_value: 1 },
        name: { default_value: "" },
        flag: { default_value: false },
      },
    };
    const applied = applyConfigureValues(schema, {
      N: 100,
      name: "v1",
      flag: true,
      extra: 1,
    });
    expect(applied.unknown).toEqual(["extra"]);
    expect(applied.parameterModifications).toEqual({
      N: { is_modified: true, field_modifications: { default_value_original: 1 } },
      name: { is_modified: true, field_modifications: { default_value_original: "" } },
      flag: { is_modified: true, field_modifications: { default_value_original: false } },
    });
    expect(
      (applied.schema as { parameters: { N: { default_value: number } } }).parameters.N
        .default_value,
    ).toBe(100);
    expect(
      (schema.parameters.N as { default_value: number }).default_value,
    ).toBe(1);
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

  it("treats a legacy hyphen handle as the same connection", () => {
    const edges = [
      {
        source: "a",
        target: "b",
        sourceHandle: "a-spikes-output-float",
        targetHandle: "b-input-input-float",
      },
    ];
    expect(
      connectionAlreadyOnCanvas(
        edges,
        "a",
        "b",
        "a::spikes::output::float",
        "b::input::input::float",
        true,
      ),
    ).toBe(true);
    expect(
      connectionAlreadyOnCanvas(
        edges,
        "a",
        "b",
        "a::other::output::float",
        "b::input::input::float",
        true,
      ),
    ).toBe(false);
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
  });

  it("refuses different port types and accepts the same type in either case", () => {
    expect(portsCanConnect("float", "int")).toBe(false);
    expect(portsCanConnect("float", "Float")).toBe(true);
    expect(portsCanConnect("float", "float")).toBe(true);
  });
});
