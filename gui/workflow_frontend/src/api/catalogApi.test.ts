import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CatalogApiError,
  dandiUrl,
  fetchCatalogStatistics,
  listCatalogDatasets,
  lookupCatalog,
  parseError,
  searchCatalog,
  sourceDisplayName,
  toCatalogHits,
} from "./catalogApi";

vi.mock("./authHeaders", () => ({
  createAuthHeaders: vi.fn(async () => ({
    "Content-Type": "application/json",
  })),
}));

function jsonResponse(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

describe("parseError", () => {
  it("maps 503 catalog_unconfigured", async () => {
    await expect(
      parseError(
        jsonResponse(503, {
          status: "error",
          code: "catalog_unconfigured",
          error: "Catalog service is not configured",
        })
      )
    ).rejects.toMatchObject({
      name: "CatalogApiError",
      status: 503,
      code: "catalog_unconfigured",
      message: "Catalog service is not configured",
    });
  });

  it("maps 503 catalog_unavailable", async () => {
    await expect(
      parseError(
        jsonResponse(503, {
          status: "error",
          code: "catalog_unavailable",
          error: "Catalog service is unavailable",
        })
      )
    ).rejects.toMatchObject({
      name: "CatalogApiError",
      status: 503,
      code: "catalog_unavailable",
      message: "Catalog service is unavailable",
    });
  });

  it("maps 502 catalog_auth", async () => {
    await expect(
      parseError(
        jsonResponse(502, {
          status: "error",
          code: "catalog_auth",
          error: "Catalog authentication failed",
        })
      )
    ).rejects.toMatchObject({
      name: "CatalogApiError",
      status: 502,
      code: "catalog_auth",
      message: "Catalog authentication failed",
    });
  });

  it("maps 404 catalog_not_found", async () => {
    await expect(
      parseError(
        jsonResponse(404, {
          status: "error",
          code: "catalog_not_found",
          error: "Not found",
        })
      )
    ).rejects.toMatchObject({
      name: "CatalogApiError",
      status: 404,
      code: "catalog_not_found",
      message: "Not found",
    });
  });

  it("does not throw when the response is ok", async () => {
    await expect(parseError(jsonResponse(200, { status: "ok" }))).resolves.toBe(
      undefined
    );
  });
});

describe("sourceDisplayName", () => {
  it("maps aws to SRPBS_TS", () => {
    expect(sourceDisplayName("aws")).toBe("SRPBS_TS");
    expect(sourceDisplayName("aws", "Amazon")).toBe("SRPBS_TS");
  });

  it("prefers source_display, then the raw source", () => {
    expect(sourceDisplayName("dandi", "DANDI")).toBe("DANDI");
    expect(sourceDisplayName("cbs")).toBe("cbs");
  });
});

describe("dandiUrl", () => {
  it("builds the public DANDI archive URL", () => {
    expect(dandiUrl("000003")).toBe("https://dandiarchive.org/dandiset/000003");
  });
});

describe("toCatalogHits", () => {
  it("maps browse datasets[] onto CatalogHit with name fallback", () => {
    const hits = toCatalogHits({
      datasets: [
        {
          dataset_id: "000003",
          source: "dandi",
          description: "example",
        },
      ],
    });
    expect(hits).toHaveLength(1);
    expect(hits[0].name).toBe("000003");
    expect(hits[0].dataset_id).toBe("000003");
    expect(hits[0].source).toBe("dandi");
  });

  it("maps search results[] onto CatalogHit", () => {
    const hits = toCatalogHits({
      results: [{ dataset_id: "abc", name: "Mouse", source: "cbs" }],
    });
    expect(hits[0].name).toBe("Mouse");
    expect(hits[0].source).toBe("cbs");
  });
});

describe("catalog fetch helpers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("throws CatalogApiError from fetchCatalogStatistics on 503 unconfigured", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(503, {
          status: "error",
          code: "catalog_unconfigured",
          error: "Catalog service is not configured",
        })
      )
    );
    await expect(fetchCatalogStatistics()).rejects.toBeInstanceOf(
      CatalogApiError
    );
    await expect(fetchCatalogStatistics()).rejects.toMatchObject({
      status: 503,
      code: "catalog_unconfigured",
    });
  });

  it("POSTs keyword search to /api/catalog/search/", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse(200, {
        status: "ok",
        mode: "keyword",
        query: "mouse",
        results: [],
        count: 0,
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await searchCatalog({ query: "mouse", source: "dandi", limit: 20 });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/catalog/search/");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      query: "mouse",
      source: "dandi",
      limit: 20,
    });
  });

  it("GETs datasets and lookup through /api/catalog/", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(200, { count: 0, datasets: [], timestamp: "now" })
      )
      .mockResolvedValueOnce(
        jsonResponse(200, { record: { dataset_id: "000003" }, requested_id: "000003" })
      );
    vi.stubGlobal("fetch", fetchMock);

    await listCatalogDatasets({ source: "dandi", limit: 50 });
    await lookupCatalog({ source: "dandi", id: "000003" });

    expect(fetchMock.mock.calls[0][0]).toBe(
      "/api/catalog/datasets/?source=dandi&limit=50"
    );
    expect(String(fetchMock.mock.calls[1][0])).toBe(
      "/api/catalog/lookup/?id=000003&source=dandi"
    );
  });
});
