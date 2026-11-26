import { describe, expect, it } from "vitest";
import { keepCatalogOnRefreshFailure } from "./keepCatalogOnRefreshFailure";

describe("keepCatalogOnRefreshFailure", () => {
  it("keeps an existing list and leaves a missing list missing", () => {
    const cards = [{ id: "node-a" }];
    expect(keepCatalogOnRefreshFailure(cards)).toBe(cards);
    expect(keepCatalogOnRefreshFailure(null)).toBeNull();
  });
});
