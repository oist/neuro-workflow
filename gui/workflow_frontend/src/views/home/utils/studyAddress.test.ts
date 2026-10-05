import { describe, expect, it } from "vitest";
import { ParameterField } from "../type";
import { FlowNode, exploreRows, rangeDictWithKey } from "./studyAddress";

const nestParams = (field: ParameterField): FlowNode =>
  ({
    id: "n1",
    position: { x: 0, y: 0 },
    data: {
      file_name: "probe.py",
      label: "Probe",
      instanceName: "probe",
      color: "",
      schema: { parameters: { nest_params: field } },
    },
  }) as unknown as FlowNode;

const declared = {
  default_value: { C_m: 250.0, V_th: -55.0, I_e: 0.0 },
  optimization_range: { I_e: [0, 800], V_th: [-60, -45] },
};

describe("rangeDictWithKey", () => {
  it("starts a fresh dict when the declared ranges are only hints", () => {
    const range = rangeDictWithKey({ ...declared, optimizable: false }, "I_e", [0, 500]);
    expect(range).toEqual({ I_e: [0, 500] });
    const rows = exploreRows([nestParams({ ...declared, optimizable: true, optimization_range: range })]);
    expect(rows.map((r) => r.key)).toEqual(["I_e"]);
  });

  it("extends the explored dict once the parameter is optimizable", () => {
    const field = { ...declared, optimizable: true, optimization_range: { I_e: [0, 500] } };
    expect(rangeDictWithKey(field, "V_th", [-60, -50])).toEqual({ I_e: [0, 500], V_th: [-60, -50] });
  });

  it("works without any declared range", () => {
    expect(rangeDictWithKey({ default_value: { I_e: 0 } }, "I_e", [0, 1])).toEqual({ I_e: [0, 1] });
  });
});
