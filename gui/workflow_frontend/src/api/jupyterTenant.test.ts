import { describe, expect, it } from "vitest";
import { projectTreePath, type JupyterSession } from "./jupyterTenant";

function session(tenant: "project" | "community"): JupyterSession {
  return {
    tenant,
    hub_user: tenant === "project" ? "internal" : "external",
    jupyter_path: "",
    viewer_token: "",
    is_node_reviewer: false,
    notice: "",
  };
}

describe("projectTreePath", () => {
  it("opens a community project's files from the read-only community folder for a project user", () => {
    expect(projectTreePath("codes/projects/p1/workflow.py", "community", session("project"))).toBe(
      "codes-community/projects/p1/workflow.py",
    );
    expect(projectTreePath("codes/nodes/network/Pop.py", "hackathon", session("project"))).toBe(
      "codes-community/nodes/network/Pop.py",
    );
  });

  it("keeps paths in the viewer's own space", () => {
    const path = "codes/projects/p1/workflow.py";
    expect(projectTreePath(path, "project", session("project"))).toBe(path);
    expect(projectTreePath(path, "community", session("community"))).toBe(path);
    expect(projectTreePath(path, undefined, session("project"))).toBe(path);
    expect(projectTreePath(`/${path}`, "project", session("project"))).toBe(path);
  });
});
