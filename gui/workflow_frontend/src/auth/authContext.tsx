import React, { createContext, useCallback, useContext, useEffect, useRef, useState, ReactNode } from 'react';
import { authService } from './authService';
import { reAuthBus, ReAuthReason } from './reAuthBus';
import { User, AuthResult } from './types';

export type RunController = {
  isRunning: boolean;
  abort: () => void;
};

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signUp: () => Promise<AuthResult>;
  signIn: () => Promise<AuthResult>;
  signOut: () => Promise<AuthResult>;
  // Re-auth modal coordination
  reAuthRequired: boolean;
  reAuthReason: ReAuthReason | null;
  dismissReAuth: () => void;
  hasPendingSaves: boolean;
  setHasPendingSaves: (v: boolean) => void;
  registerSaveFlusher: (fn: (() => Promise<void>) | null) => void;
  registerRunController: (ctrl: RunController | null) => void;
  flushPendingSaves: () => Promise<void>;
  runController: RunController | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const [reAuthRequired, setReAuthRequired] = useState(false);
  const [reAuthReason, setReAuthReason] = useState<ReAuthReason | null>(null);
  const [hasPendingSaves, setHasPendingSaves] = useState(false);
  const [runController, setRunController] = useState<RunController | null>(null);

  const saveFlusherRef = useRef<(() => Promise<void>) | null>(null);
  const reAuthRequiredRef = useRef(false);

  useEffect(() => {
    reAuthRequiredRef.current = reAuthRequired;
  }, [reAuthRequired]);

  useEffect(() => {
    const init = async () => {
      try {
        const authenticated = await authService.init();
        if (authenticated) {
          const u = await authService.getCurrentUser();
          if (u) setUser(u);
        }
      } catch (error) {
        console.error('Error initializing auth:', error);
      } finally {
        setLoading(false);
      }
    };

    init();

    const { data: { subscription } } = authService.onAuthStateChange(
      (event, session) => {
        if (event === 'SIGNED_IN' && session?.user) {
          setUser(session.user);
        } else if (event === 'SIGNED_OUT') {
          // No-op: authService no longer forwards Keycloak's onAuthLogout, so
          // refresh failures cannot wipe the user behind our backs. Explicit
          // signOut() triggers a full browser redirect via kc.logout(), so
          // React state cleanup is unnecessary either way.
        } else if (event === 'TOKEN_REFRESHED' && session?.user) {
          setUser(session.user);
        }
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  // Subscribe to re-auth events from authService / apiInterceptors
  useEffect(() => {
    const unsubscribe = reAuthBus.subscribe((reason) => {
      // Suppress repeated emits while the modal is already up (avoids 401 loops)
      if (reAuthRequiredRef.current) return;
      reAuthRequiredRef.current = true;
      setReAuthReason(reason);
      setReAuthRequired(true);
    });
    return unsubscribe;
  }, []);

  const dismissReAuth = useCallback(() => {
    setReAuthRequired(false);
    setReAuthReason(null);
  }, []);

  const registerSaveFlusher = useCallback((fn: (() => Promise<void>) | null) => {
    saveFlusherRef.current = fn;
  }, []);

  const registerRunController = useCallback((ctrl: RunController | null) => {
    setRunController(ctrl);
  }, []);

  const flushPendingSaves = useCallback(async () => {
    const fn = saveFlusherRef.current;
    if (fn) {
      await fn();
    }
  }, []);

  const signUp = async () => {
    setLoading(true);
    try {
      return await authService.signUp();
    } finally {
      setLoading(false);
    }
  };

  const signIn = async () => {
    setLoading(true);
    try {
      return await authService.signIn();
    } finally {
      setLoading(false);
    }
  };

  const signOut = async () => {
    setLoading(true);
    try {
      return await authService.signOut();
    } finally {
      setLoading(false);
    }
  };

  const value: AuthContextType = {
    user,
    loading,
    signUp,
    signIn,
    signOut,
    reAuthRequired,
    reAuthReason,
    dismissReAuth,
    hasPendingSaves,
    setHasPendingSaves,
    registerSaveFlusher,
    registerRunController,
    flushPendingSaves,
    runController,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
