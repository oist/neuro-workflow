import { describe, expect, it } from "vitest";
import {
  filterProjectOptions,
  matchProjectOption,
  toProjectOptions,
  type ProjectOption,
} from "./projectFilter";

const projects = [
  {
    id: "1",
    name: "Human fMRI analysis Visualization",
    description: "Bold maps from resting-state scans",
    visibility: "public" as const,
  },
  {
    id: "2",
    name: "Test_project",
    description: "sandbox for kirill",
    visibility: "private" as const,
  },
  {
    id: "3",
    name: "Ring Attractor (Hatsuta)",
    description: "",
    visibility: "public" as const,
  },
];

const options: ProjectOption[] = toProjectOptions(projects);

describe("toProjectOptions", () => {
  it("maps id to value and name to label", () => {
    expect(options[0]).toEqual({
      value: "1",
      label: "Human fMRI analysis Visualization",
      description: "Bold maps from resting-state scans",
      visibility: "public",
    });
  });
});

describe("filterProjectOptions", () => {
  it("returns all options for an empty query", () => {
    expect(filterProjectOptions(options, "  ")).toEqual(options);
  });

  it("matches a name substring", () => {
    const result = filterProjectOptions(options, "fMRI");
    expect(result.map((o) => o.value)).toEqual(["1"]);
  });

  it("matches a description that is not in the name", () => {
    const result = filterProjectOptions(options, "Bold maps");
    expect(result.map((o) => o.value)).toEqual(["1"]);
    expect(matchProjectOption(options[1], "Bold maps")).toBe(false);
  });

  it("matches visibility public vs private as a word", () => {
    expect(filterProjectOptions(options, "public").map((o) => o.value)).toEqual([
      "1",
      "3",
    ]);
    expect(filterProjectOptions(options, "private").map((o) => o.value)).toEqual([
      "2",
    ]);
  });

  it("matches visibility prefixes of length 3 or more", () => {
    expect(filterProjectOptions(options, "pub").map((o) => o.value)).toEqual([
      "1",
      "3",
    ]);
    expect(filterProjectOptions(options, "priv").map((o) => o.value)).toEqual(["2"]);
  });

  it("does not treat a short p as a visibility filter", () => {
    const result = filterProjectOptions(options, "p");
    // Ring Attractor is public and has no "p" in name/description.
    expect(result.map((o) => o.value)).not.toContain("3");
  });

  it("returns nothing when nothing matches", () => {
    expect(filterProjectOptions(options, "zzzz-no-such-project")).toEqual([]);
  });
});
