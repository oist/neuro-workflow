import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const source = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "calculationNode.tsx"),
  "utf8",
);

describe("disable control", () => {
  it("uses the boolean true that code generation ignores", () => {
    expect(source).toContain("Disable");
    expect(source).toContain("Enable");
    expect(source).toContain("disabled: !data.disabled");
    expect(source).toContain("data.disabled === true");
  });
});
