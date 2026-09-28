import { afterEach, describe, expect, it, vi } from "vitest";
import { postNotebookToken } from "./notebookTokenApi";

describe("postNotebookToken", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("posts the project id with the given bearer token", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200 }) as Response);
    vi.stubGlobal("fetch", fetchMock);

    await postNotebookToken("4b5023b0-8f1e-4dfc-87f0-1579c1a9bf00", "eyJ.abc");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toMatch(/\/chat\/notebook-token\/$/);
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>).Authorization).toBe(
      "Bearer eyJ.abc"
    );
    expect(JSON.parse(init.body as string)).toEqual({
      project_id: "4b5023b0-8f1e-4dfc-87f0-1579c1a9bf00",
    });
  });

  it("throws on a non-2xx response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 }) as Response));

    await expect(postNotebookToken("p", "t")).rejects.toThrow("404");
  });
});
