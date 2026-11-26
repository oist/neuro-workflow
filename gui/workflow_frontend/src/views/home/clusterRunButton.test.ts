import { describe, expect, it } from "vitest";
import { CLUSTER_PROJECT_SPACE_LABEL, VIEW_ONLY_RUNS_LABEL, clusterRunButton, localRunButton } from "./clusterRunButton";

describe("clusterRunButton", () => {
  it("disables cluster runs for a community user", () => {
    expect(
      clusterRunButton({ canEdit: true, userTenant: "community", hasProject: true, nodeCount: 3 }),
    ).toEqual({ disabled: true, label: CLUSTER_PROJECT_SPACE_LABEL });
  });

  it("keeps the community reason when there is no project and when there are no nodes", () => {
    expect(
      clusterRunButton({ canEdit: true, userTenant: "community", hasProject: false, nodeCount: 0 }),
    ).toEqual({ disabled: true, label: CLUSTER_PROJECT_SPACE_LABEL });
    expect(
      clusterRunButton({ canEdit: true, userTenant: "community", hasProject: true, nodeCount: 0 }),
    ).toEqual({ disabled: true, label: CLUSTER_PROJECT_SPACE_LABEL });
  });

  it("keeps a project user with no project on Select Project First", () => {
    expect(
      clusterRunButton({ canEdit: true, userTenant: "project", hasProject: false, nodeCount: 3 }),
    ).toEqual({ disabled: true, label: "Select Project First" });
  });

  it("enables a project user who has a project and nodes", () => {
    expect(
      clusterRunButton({ canEdit: true, userTenant: "project", hasProject: true, nodeCount: 2 }),
    ).toEqual({ disabled: false, label: "☁️ Run on Compute Cluster" });
  });

  it("keeps a project user with no nodes disabled", () => {
    expect(
      clusterRunButton({ canEdit: true, userTenant: "project", hasProject: true, nodeCount: 0 }),
    ).toEqual({ disabled: true, label: "Add Nodes to Run" });
  });

  it("leaves today's rules in place when the session is unknown", () => {
    expect(
      clusterRunButton({ canEdit: true, userTenant: null, hasProject: false, nodeCount: 4 }),
    ).toEqual({ disabled: true, label: "Select Project First" });
  });

  it("disables a view-only project before the community and empty-canvas labels", () => {
    expect(
      clusterRunButton({ canEdit: false, userTenant: "project", hasProject: true, nodeCount: 3 }),
    ).toEqual({ disabled: true, label: VIEW_ONLY_RUNS_LABEL });
    expect(
      clusterRunButton({ canEdit: false, userTenant: "community", hasProject: true, nodeCount: 3 }),
    ).toEqual({ disabled: true, label: VIEW_ONLY_RUNS_LABEL });
  });
});

describe("localRunButton", () => {
  it("disables the local run only when the project cannot be edited", () => {
    expect(localRunButton(false)).toEqual({ disabled: true, label: VIEW_ONLY_RUNS_LABEL });
    expect(localRunButton(true)).toEqual({ disabled: false, label: "Execute" });
  });
});
