import { describe, expect, it } from "vitest";
import { simulationFileRows } from "./simulationFileRows";

describe("simulationFileRows", () => {
  it("keeps a saved credit and starts a new file empty", () => {
    const rows = simulationFileRows(
      [{ filename: "connectivity.npz" }, { filename: "spikes.csv" }],
      [{ path: "results/rates.npz", filename: "rates.npz" }],
      [
        {
          path: "connectivity.npz",
          kind: "input",
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
        kind: "input",
        source: "Allen Institute",
        owner: "Lab",
        paper: "doi:10.1/example",
        acknowledgment: "Thanks",
      },
      {
        path: "spikes.csv",
        kind: "input",
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
