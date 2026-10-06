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
  // Restore the stored selection for this user, then fetch the models.
  init: (userId: string) => Promise<void>;
  selectModel: (id: string | null) => void;
}

export const useChatModelStore = create<ChatModelStore>((set, get) => ({
  userId: null,
  models: [],
  selectedModelId: null,

  init: async (userId) => {
    set({ userId, selectedModelId: readStoredSelection(userId) });
    const models = await listChatModels();
    set({ models });
    // Fall back to the default if the selected model is no longer offered.
    const { selectedModelId } = get();
    if (
      selectedModelId !== null &&
      !models.some((m) => m.id === selectedModelId)
    ) {
      get().selectModel(null);
    }
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
