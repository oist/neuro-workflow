import { beforeEach, describe, expect, it, vi } from "vitest";

const listChatModels = vi.fn();
vi.mock("@/api/chatApi", () => ({ listChatModels }));

const MODELS = [
  { id: "gpt-test", provider: "openai" },
  { id: "MiniMax-M3", provider: "minimax" },
];
const KEY = "chatModelId:alice";

describe("chatModelStore.init", () => {
  let stored: Record<string, string>;

  beforeEach(() => {
    stored = {};
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => stored[k] ?? null,
      setItem: (k: string, v: string) => void (stored[k] = v),
      removeItem: (k: string) => void delete stored[k],
    });
    vi.resetModules();
    listChatModels.mockReset();
  });

  const load = async () => (await import("./chatModelStore")).useChatModelStore;

  it("restores a stored model that is still offered", async () => {
    stored[KEY] = "MiniMax-M3";
    listChatModels.mockResolvedValue(MODELS);
    const store = await load();

    await store.getState().init("alice");

    expect(store.getState().selectedModelId).toBe("MiniMax-M3");
  });

  it("drops a stored model that is no longer offered", async () => {
    stored[KEY] = "MiniMax-M2";
    listChatModels.mockResolvedValue(MODELS);
    const store = await load();

    await store.getState().init("alice");

    expect(store.getState().selectedModelId).toBeNull();
    expect(stored[KEY]).toBeUndefined();
  });

  it("uses the default model when the catalogue cannot be loaded", async () => {
    stored[KEY] = "MiniMax-M2";
    listChatModels.mockRejectedValue(new Error("503"));
    const store = await load();

    await expect(store.getState().init("alice")).rejects.toThrow("503");

    // Not sent as `model`, so the backend default applies; the stored choice
    // is kept for the next successful load.
    expect(store.getState().selectedModelId).toBeNull();
    expect(stored[KEY]).toBe("MiniMax-M2");
  });
});
