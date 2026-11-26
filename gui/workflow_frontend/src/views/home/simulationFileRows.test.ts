import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { simulationFileRows } from "./simulationFileRows";

describe("simulationFileRows", () => {
  it("titles the window Simulation files, not inputs and outputs", () => {
    const modal = readFileSync(new URL("./components/SimulationIoModal.tsx", import.meta.url), "utf8");
    expect(modal).toContain("<ModalHeader>Simulation files</ModalHeader>");
    expect(modal).not.toContain("inputs and outputs");
  });

  it("keeps a saved credit and starts a new file empty", () => {
    const rows = simulationFileRows(
      [{ filename: "connectivity.npz" }, { filename: "spikes.csv" }],
      [{ path: "results/rates.npz", filename: "rates.npz" }],
      [
        {
          path: "connectivity.npz",
          kind: "project_file",
          source: "Allen Institute",
          owner: "Lab",
          paper: "doi:10.1/example",
          acknowledgment: "Thanks",
        },
      ],
    );
    expect(rows).toEqual([
      {
        path: "connectivity.npz",
        kind: "project_file",
        source: "Allen Institute",
        owner: "Lab",
        paper: "doi:10.1/example",
        acknowledgment: "Thanks",
      },
      {
        path: "spikes.csv",
        kind: "project_file",
        source: "",
        owner: "",
        paper: "",
        acknowledgment: "",
      },
      {
        path: "results/rates.npz",
        kind: "output",
        source: "",
        owner: "",
        paper: "",
        acknowledgment: "",
      },
    ]);
  });
});
