export function refreshResultIsCurrent(
  startedAt: number,
  currentAt: number,
  startedProjectId: string,
  currentProjectId: string | null,
): boolean {
  return startedAt !== 0
    && startedAt === currentAt
    && currentProjectId === startedProjectId;
}
