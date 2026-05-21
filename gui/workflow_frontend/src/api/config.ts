interface ApiConfig {
  baseURL: string;
  internalSecret: string;
  timeout: number;
}

export const getApiConfig = (): ApiConfig => {
  const env = import.meta.env.MODE || "development";

  switch (env) {
    case "production":
      return {
        baseURL: import.meta.env.VITE_API_BASE_URL,
        internalSecret: import.meta.env.VITE_INTERNAL_SECRET || "",
        timeout: 10000,
      };
    case "staging":
      return {
        baseURL: import.meta.env.VITE_API_BASE_URL,
        internalSecret: import.meta.env.VITE_INTERNAL_SECRET || "",
        timeout: 15000,
      };
    default: // development
      return {
        baseURL: import.meta.env.VITE_API_BASE_URL,
        internalSecret: import.meta.env.VITE_INTERNAL_SECRET || "dev-secret",
        timeout: 30000,
      };
  }
};

export interface ApiError {
  status: number;
  message: string;
  details?: any;
}

// debug settings
export const isDebugMode = () => {
  return (
    import.meta.env.VITE_DEBUG_API === "true" ||
    import.meta.env.MODE === "development"
  );
};
