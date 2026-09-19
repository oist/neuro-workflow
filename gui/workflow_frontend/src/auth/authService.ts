import { getKeycloak, isKeycloakConfigured } from "./keycloak";
import { reAuthBus } from "./reAuthBus";
import { AuthResult, User } from "./types";

export type { AuthResult } from "./types";

class AuthService {
  private keycloakInitPromise: Promise<boolean> | null = null;
  private readonly keycloakInitTimeoutMs = 15000;

  async init(): Promise<boolean> {
    if (!isKeycloakConfigured) return false;
    if (this.keycloakInitPromise) return this.keycloakInitPromise;
    if (window.__NEURO_WORKFLOW_KEYCLOAK_INIT__) {
      this.keycloakInitPromise = window.__NEURO_WORKFLOW_KEYCLOAK_INIT__;
      return this.keycloakInitPromise;
    }

    const kc = getKeycloak();
    if ((kc as any).didInitialize) {
      return Boolean(kc.authenticated);
    }

    const initPromise = kc.init({
      // Use "check-sso" rather than "login-required" so that a failed
      // updateToken() does not auto-redirect before ReAuthGate can render.
      onLoad: "check-sso",
      checkLoginIframe: false,
    });

    let timeoutId: number | undefined;
    const timeoutPromise = new Promise<boolean>((resolve) => {
      timeoutId = window.setTimeout(() => {
        console.error(`Keycloak init timed out after ${this.keycloakInitTimeoutMs}ms`);
        resolve(false);
      }, this.keycloakInitTimeoutMs);
    });

    this.keycloakInitPromise = window.__NEURO_WORKFLOW_KEYCLOAK_INIT__ = Promise.race([
      initPromise,
      timeoutPromise,
    ])
      .finally(() => {
        if (timeoutId !== undefined) {
          window.clearTimeout(timeoutId);
        }
      })
      .catch((err) => {
        console.error("Keycloak init failed:", err);
        return false;
      });

    return this.keycloakInitPromise;
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

  async getAccessToken(minValidity = 30): Promise<string | null> {
    const kc = getKeycloak();
    if (!kc.authenticated) return null;
    try {
      await kc.updateToken(minValidity);
    } catch {
      reAuthBus.emit("refresh-failed");
      return null;
    }
    return kc.token ?? null;
  }

  async getSession() {
    const user = await this.getCurrentUser();
    return user ? { user } : null;
  }

  async updatePassword(_password: string): Promise<AuthResult> {
    return {
      success: false,
      error: { message: "Password updates are handled by the identity provider." },
    };
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
    // Refresh-failure recovery flows through reAuthBus/ReAuthGate. Do not clear
    // React auth state on keycloak-js logout events fired during failed refresh.
    kc.onAuthLogout = () => {};
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
