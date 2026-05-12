import { getKeycloak } from "./keycloak";
import { reAuthBus } from "./reAuthBus";
import { AuthResult, User } from "./types";

export type { AuthResult } from "./types";

class AuthService {
  async init(): Promise<boolean> {
    const kc = getKeycloak();
    try {
      return await kc.init({
        onLoad: "login-required",
        checkLoginIframe: false,
      });
    } catch (err) {
      console.error("Keycloak init failed:", err);
      return false;
    }
  }

  async signUp(): Promise<AuthResult> {
    getKeycloak().register();
    return { success: true };
  }

  async signIn(): Promise<AuthResult> {
    await getKeycloak().login();
    return { success: true };
  }

  async signOut(): Promise<AuthResult> {
    await getKeycloak().logout({ redirectUri: window.location.origin });
    return { success: true };
  }

  async getCurrentUser(): Promise<User | null> {
    const kc = getKeycloak();
    if (!kc.authenticated || !kc.tokenParsed) return null;
    const t = kc.tokenParsed as Record<string, any>;
    return {
      id: t.sub ?? "",
      email: t.email ?? "",
      user_metadata: {
        name: t.preferred_username,
        full_name: `${t.given_name ?? ""} ${t.family_name ?? ""}`.trim(),
      },
    };
  }

  async getAccessToken(): Promise<string | null> {
    const kc = getKeycloak();
    if (!kc.authenticated) return null;
    try {
      await kc.updateToken(30);
    } catch {
      reAuthBus.emit("refresh-failed");
      return null;
    }
    return kc.token ?? null;
  }

  onAuthStateChange(callback: (event: string, session: { user: User } | null) => void) {
    const kc = getKeycloak();
    const prev = {
      onAuthSuccess: kc.onAuthSuccess,
      onAuthLogout: kc.onAuthLogout,
      onAuthRefreshSuccess: kc.onAuthRefreshSuccess,
      onTokenExpired: kc.onTokenExpired,
    };
    kc.onAuthSuccess = async () => {
      const user = await this.getCurrentUser();
      callback("SIGNED_IN", user ? { user } : null);
    };
    kc.onAuthLogout = () => callback("SIGNED_OUT", null);
    kc.onAuthRefreshSuccess = async () => {
      const user = await this.getCurrentUser();
      callback("TOKEN_REFRESHED", user ? { user } : null);
    };
    kc.onTokenExpired = () => {
      kc.updateToken(30).catch(() => reAuthBus.emit("refresh-failed"));
    };
    return {
      data: {
        subscription: {
          unsubscribe: () => {
            kc.onAuthSuccess = prev.onAuthSuccess;
            kc.onAuthLogout = prev.onAuthLogout;
            kc.onAuthRefreshSuccess = prev.onAuthRefreshSuccess;
            kc.onTokenExpired = prev.onTokenExpired;
          },
        },
      },
    };
  }
}

export const authService = new AuthService();
