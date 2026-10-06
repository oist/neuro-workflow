import { create } from "zustand";
import { listChatModels, type ChatModel } from "@/api/chatApi";

// The selected model is remembered per user in this browser; the models on
// offer come from the backend configuration (the first one is the default).
const storageKey = (userId: string) => `chatModelId:${userId}`;

const readStoredSelection = (userId: string): string | null => {
  try {
    return localStorage.getItem(storageKey(userId));
  } catch {
    return null;
  }
};

interface ChatModelStore {
  userId: string | null;
  models: ChatModel[];
  // null: the default model.
  selectedModelId: string | null;
  // Fetch the models, then restore this user's stored selection.
  init: (userId: string) => Promise<void>;
  selectModel: (id: string | null) => void;
}

export const useChatModelStore = create<ChatModelStore>((set, get) => ({
  userId: null,
  models: [],
  selectedModelId: null,

  init: async (userId) => {
    // The stored choice applies only once the catalogue confirms it is still
    // offered. Until then, and if loading fails, messages use the default
    // model instead of being rejected for a model that was since removed.
    set({ userId, selectedModelId: null });
    const models = await listChatModels();
    set({ models });
    const stored = readStoredSelection(userId);
    get().selectModel(models.some((m) => m.id === stored) ? stored : null);
  },

  selectModel: (id) => {
    const { userId } = get();
    if (userId) {
      try {
        if (id) localStorage.setItem(storageKey(userId), id);
        else localStorage.removeItem(storageKey(userId));
      } catch {
        // localStorage unavailable: selection just won't survive a reload
      }
    }
    set({ selectedModelId: id });
  },
}));
