export const CLUSTER_PROJECT_SPACE_LABEL = "Cluster runs are for the project space";

export function clusterRunButton(input: {
  userTenant: "project" | "community" | null;
  hasProject: boolean;
  nodeCount: number;
}): { disabled: boolean; label: string } {
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
