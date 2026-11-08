export function nodeClassName(data: {
  label?: string | null;
  file_name?: string | null;
  instanceName?: string | null;
}): string {
  const label = (data.label || "").trim();
  if (label) {
    return label;
  }
  return (data.file_name || "").replace(/\.py$/i, "");
}
