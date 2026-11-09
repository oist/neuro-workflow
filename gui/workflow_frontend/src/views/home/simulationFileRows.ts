export interface SimulationFileNote {
  path: string;
  kind: "input" | "output";
  source: string;
  owner: string;
  paper: string;
  acknowledgment: string;
}

interface ListedFile {
  filename?: string;
  path?: string;
}

function blank(path: string, kind: SimulationFileNote["kind"]): SimulationFileNote {
  return { path, kind, source: "", owner: "", paper: "", acknowledgment: "" };
}

export function simulationFileRows(
  files: ListedFile[],
  results: ListedFile[],
  saved: SimulationFileNote[],
): SimulationFileNote[] {
  const savedByPath = new Map(saved.map((row) => [`${row.kind}:${row.path}`, row]));
  const rows: SimulationFileNote[] = [];
  for (const file of files) {
    const path = file.filename || "";
    if (!path) continue;
    const prior = savedByPath.get(`input:${path}`);
    rows.push(prior ? { ...blank(path, "input"), ...prior, path, kind: "input" } : blank(path, "input"));
  }
  for (const file of results) {
    const path = file.path || file.filename || "";
    if (!path) continue;
    const prior = savedByPath.get(`output:${path}`);
    rows.push(prior ? { ...blank(path, "output"), ...prior, path, kind: "output" } : blank(path, "output"));
  }
  return rows;
}
