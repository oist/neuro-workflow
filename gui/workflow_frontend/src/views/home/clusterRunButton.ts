export const CLUSTER_PROJECT_SPACE_LABEL = "Cluster runs are for the project space";
export const VIEW_ONLY_RUNS_LABEL = "View only — runs are not available";

export function clusterRunButton(input: {
  canEdit: boolean;
  userTenant: "project" | "community" | null;
  hasProject: boolean;
  nodeCount: number;
}): { disabled: boolean; label: string } {
  if (!input.canEdit) {
    return { disabled: true, label: VIEW_ONLY_RUNS_LABEL };
  }
  if (input.userTenant === "community") {
    return { disabled: true, label: CLUSTER_PROJECT_SPACE_LABEL };
  }
  if (!input.hasProject) {
    return { disabled: true, label: "Select Project First" };
  }
  if (input.nodeCount === 0) {
    return { disabled: true, label: "Add Nodes to Run" };
  }
  return { disabled: false, label: "☁️ Run on Compute Cluster" };
}

export function localRunButton(canEdit: boolean): { disabled: boolean; label: string } {
  if (!canEdit) {
    return { disabled: true, label: VIEW_ONLY_RUNS_LABEL };
  }
  return { disabled: false, label: "Execute" };
}
