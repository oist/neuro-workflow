export type ProjectVisibility = "public" | "private";

export type ProjectOption = {
  value: string;
  label: string;
  description?: string;
  visibility: ProjectVisibility;
};

type ProjectLike = {
  id: string;
  name: string;
  description?: string | null;
  visibility: ProjectVisibility;
};

export function toProjectOptions(projects: ProjectLike[]): ProjectOption[] {
  return projects.map((project) => ({
    value: project.id,
    label: project.name,
    description: project.description ?? undefined,
    visibility: project.visibility,
  }));
}

function matchesVisibilityWord(visibility: ProjectVisibility, q: string): boolean {
  if (q.length < 3) {
    return false;
  }
  return visibility.startsWith(q);
}

export function matchProjectOption(option: ProjectOption, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) {
    return true;
  }
  if (option.label.toLowerCase().includes(q)) {
    return true;
  }
  if ((option.description ?? "").toLowerCase().includes(q)) {
    return true;
  }
  return matchesVisibilityWord(option.visibility, q);
}

export function filterProjectOptions(
  options: ProjectOption[],
  query: string
): ProjectOption[] {
  return options.filter((option) => matchProjectOption(option, query));
}

/** react-select filterOption adapter. */
export function matchProjectFilterOption(
  option: { data: ProjectOption },
  query: string
): boolean {
  return matchProjectOption(option.data, query);
}
