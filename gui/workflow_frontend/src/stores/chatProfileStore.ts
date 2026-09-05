import { create } from "zustand";
import { listChatProfiles, type ChatProfile } from "@/api/chatProfileApi";

// The selected profile is remembered per user in this browser; the profiles
// themselves live on the backend.
const storageKey = (userId: string) => `chatProfileId:${userId}`;

const readStoredSelection = (userId: string): string | null => {
  try {
    return localStorage.getItem(storageKey(userId));
  } catch {
    return null;
  }
};

interface ChatProfileStore {
  userId: string | null;
  profiles: ChatProfile[];
  selectedProfileId: string | null;
  // Restore the stored selection for this user, then fetch profiles.
  init: (userId: string) => Promise<void>;
  loadProfiles: () => Promise<void>;
  selectProfile: (id: string | null) => void;
}

export const useChatProfileStore = create<ChatProfileStore>((set, get) => ({
  userId: null,
  profiles: [],
  selectedProfileId: null,

  init: async (userId) => {
    set({ userId, selectedProfileId: readStoredSelection(userId) });
    await get().loadProfiles();
  },

  loadProfiles: async () => {
    const profiles = await listChatProfiles();
    set({ profiles });
    // Fall back to Default if the selected profile was deleted elsewhere.
    const { selectedProfileId } = get();
    if (
      selectedProfileId !== null &&
      !profiles.some((p) => p.id === selectedProfileId)
    ) {
      get().selectProfile(null);
    }
  },

  selectProfile: (id) => {
    const { userId } = get();
    if (userId) {
      try {
        if (id) localStorage.setItem(storageKey(userId), id);
        else localStorage.removeItem(storageKey(userId));
      } catch {
        // localStorage unavailable: selection just won't survive a reload
      }
    }
    set({ selectedProfileId: id });
  },
}));

export const selectSelectedProfile = (s: ChatProfileStore): ChatProfile | null =>
  s.profiles.find((p) => p.id === s.selectedProfileId) ?? null;
