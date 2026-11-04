import { describe, expect, it } from "vitest";
import { projectFolderTreePath } from "./projectFolderPath";

describe("projectFolderTreePath", () => {
  it("opens the project notebook path Jupyter already uses", () => {
    expect(projectFolderTreePath("5ad77111-c21d-49b4-b948-c55bb8da8563")).toBe(
      "codes/projects/5ad77111-c21d-49b4-b948-c55bb8da8563/workflow.py",
    );
  });
});
