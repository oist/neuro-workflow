export function messageFromProjectCreateError(body: unknown, status: number): string {
  const record =
    body && typeof body === "object" ? (body as Record<string, unknown>) : {};
  const name = record.name;
  const nameText = Array.isArray(name)
    ? name.filter((item): item is string => typeof item === "string").join(" ")
    : typeof name === "string"
      ? name
      : "";
  if (nameText) {
    return nameText;
  }
  if (typeof record.error === "string" && record.error) {
    return record.error;
  }
  if (status === 401) {
    return "Authentication required. Please login first.";
  }
  if (status === 403) {
    return "Permission denied. You do not have access to create projects.";
  }
  if (status === 400) {
    return "Invalid project data. Please check your input.";
  }
  if (status === 500) {
    return "Server error. Please try again later.";
  }
  return "An unexpected error occurred";
}
