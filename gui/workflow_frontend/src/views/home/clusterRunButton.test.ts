import { describe, expect, it } from "vitest";
import { CLUSTER_PROJECT_SPACE_LABEL, clusterRunButton } from "./clusterRunButton";

describe("clusterRunButton", () => {
  it("disables cluster runs for a community user", () => {
    expect(
      clusterRunButton({ userTenant: "community", hasProject: true, nodeCount: 3 }),
    ).toEqual({ disabled: true, label: CLUSTER_PROJECT_SPACE_LABEL });
  });

  it("keeps the community reason when there is no project and when there are no nodes", () => {
    expect(
      clusterRunButton({ userTenant: "community", hasProject: false, nodeCount: 0 }),
    ).toEqual({ disabled: true, label: CLUSTER_PROJECT_SPACE_LABEL });
    expect(
      clusterRunButton({ userTenant: "community", hasProject: true, nodeCount: 0 }),
    ).toEqual({ disabled: true, label: CLUSTER_PROJECT_SPACE_LABEL });
  });

  it("keeps a project user with no project on Select Project First", () => {
    expect(
      clusterRunButton({ userTenant: "project", hasProject: false, nodeCount: 3 }),
    ).toEqual({ disabled: true, label: "Select Project First" });
  });

  it("enables a project user who has a project and nodes", () => {
    expect(
      clusterRunButton({ userTenant: "project", hasProject: true, nodeCount: 2 }),
    ).toEqual({ disabled: false, label: "☁️ Run on Compute Cluster" });
  });

  it("keeps a project user with no nodes disabled", () => {
    expect(
      clusterRunButton({ userTenant: "project", hasProject: true, nodeCount: 0 }),
    ).toEqual({ disabled: true, label: "Add Nodes to Run" });
  });

  it("leaves today's rules in place when the session is unknown", () => {
    expect(
      clusterRunButton({ userTenant: null, hasProject: false, nodeCount: 4 }),
    ).toEqual({ disabled: true, label: "Select Project First" });
  });
});
