import { create } from "zustand";
import {
  fetchCanManageChatProfiles,
  listChatProfiles,
  type ChatProfile,
} from "@/api/chatProfileApi";

// The selected profile is remembered per user in this browser; the profiles
// themselves live on the backend and are managed by staff.
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
  // Django is_staff: may create/edit/delete profiles and always use Default.
  canManage: boolean;
  // Restore the stored selection for this user, then fetch profiles.
  init: (userId: string) => Promise<void>;
  loadProfiles: () => Promise<void>;
  selectProfile: (id: string | null) => void;
}

export const useChatProfileStore = create<ChatProfileStore>((set, get) => ({
  userId: null,
  profiles: [],
  selectedProfileId: null,
  canManage: false,

  init: async (userId) => {
    set({ userId, selectedProfileId: readStoredSelection(userId) });
    await get().loadProfiles();
  },

  loadProfiles: async () => {
    const [profiles, canManage] = await Promise.all([
      listChatProfiles(),
      fetchCanManageChatProfiles(),
    ]);
    set({ profiles, canManage });
    // Fall back to Default if the selected profile was deleted elsewhere.
    let { selectedProfileId } = get();
    if (
      selectedProfileId !== null &&
      !profiles.some((p) => p.id === selectedProfileId)
    ) {
      selectedProfileId = null;
      get().selectProfile(null);
    }
    // Non-staff users cannot use Default while an admin default exists; the
    // backend applies it anyway, so mirror that in the UI.
    const defaultProfile = profiles.find((p) => p.is_default);
    if (selectedProfileId === null && !canManage && defaultProfile) {
      get().selectProfile(defaultProfile.id);
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

// "Default (all tools)" is available to staff, or to everyone when no admin
// default profile is set.
export const selectCanUseNoProfile = (s: ChatProfileStore): boolean =>
  s.canManage || !s.profiles.some((p) => p.is_default);
