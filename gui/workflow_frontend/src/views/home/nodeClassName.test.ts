import { describe, expect, it } from "vitest";
import { nodeClassName } from "./nodeClassName";

describe("nodeClassName", () => {
  it("keeps the catalog label after the instance is renamed", () => {
    expect(
      nodeClassName({
        label: "HybridBenchmark",
        instanceName: "my_run",
        file_name: "HybridBenchmark.py",
      }),
    ).toBe("HybridBenchmark");
  });

  it("falls back to the file stem when the label is empty", () => {
    expect(
      nodeClassName({
        label: "  ",
        instanceName: "renamed",
        file_name: "Evaluate.py",
      }),
    ).toBe("Evaluate");
  });

  it("does not use the instance name as the class", () => {
    expect(
      nodeClassName({
        label: "",
        instanceName: "only_instance",
        file_name: "",
      }),
    ).toBe("");
  });
});
