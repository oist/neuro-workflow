export function archiveFailureMessage(body: unknown, status: number): string {
  const record =
    body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  if (typeof record.error === "string" && record.error) {
    return record.error;
  }
  if (typeof record.detail === "string" && record.detail) {
    return record.detail;
  }
  return `HTTP ${status}`;
}
