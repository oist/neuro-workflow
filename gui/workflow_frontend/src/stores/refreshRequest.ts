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

export function refreshFailureMessage(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const record = body as { error?: unknown; detail?: unknown };
    if (typeof record.error === "string" && record.error) {
      return record.error;
    }
    if (typeof record.detail === "string" && record.detail) {
      return record.detail;
    }
  }
  return `HTTP ${status}`;
}
