export type ProjectVisibility = "public" | "private";

export type ProjectOption = {
  value: string;
  label: string;
  description?: string;
  visibility: ProjectVisibility;
};

type ProjectOwnerLike = {
  first_name?: string;
  last_name?: string;
  email?: string;
};

type ProjectLike = {
  id: string;
  name: string;
  description?: string | null;
  visibility: ProjectVisibility;
  owner?: ProjectOwnerLike | null;
};

function ownerLabel(owner: ProjectOwnerLike | null | undefined): string {
  if (!owner) {
    return "";
  }
  const name = [owner.first_name, owner.last_name].filter(Boolean).join(" ").trim();
  return name || (owner.email ?? "").trim();
}

export function toProjectOptions(projects: ProjectLike[]): ProjectOption[] {
  const nameCounts = new Map<string, number>();
  for (const project of projects) {
    const key = project.name.trim().toLowerCase();
    nameCounts.set(key, (nameCounts.get(key) ?? 0) + 1);
  }
  return projects.map((project) => {
    const shared = (nameCounts.get(project.name.trim().toLowerCase()) ?? 0) > 1;
    const who = shared ? ownerLabel(project.owner) : "";
    return {
      value: project.id,
      label: who ? `${project.name} (${who})` : project.name,
      description: project.description ?? undefined,
      visibility: project.visibility,
    };
  });
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
