import { describe, expect, it } from "vitest";
import { parseWorkflowSource } from "./notebookToCanvas";

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
