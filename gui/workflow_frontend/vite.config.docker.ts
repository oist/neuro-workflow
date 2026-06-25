import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";
import "dotenv/config";
import path from "path";
import { execSync } from "child_process";

function getGitCommitHash(): string {
  try {
    return execSync("git rev-parse --short HEAD").toString().trim();
  } catch {
    return "unknown";
  }
}

// https://vite.dev/config/

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version || "0.0.0"),
    __GIT_COMMIT_HASH__: JSON.stringify(getGitCommitHash()),
  },
  server: {
    watch: {
      usePolling: true,
    },
    proxy: {
      "/api": {
        target: process.env.VITE_PROXY_BACKEND || "http://backend:3000",
        changeOrigin: true,
        secure: false,
      },
      "/jupyter": {
        target: process.env.VITE_PROXY_JUPYTER || "http://jupyterhub:8000",
        changeOrigin: true,
        secure: false,
        ws: true,
      },
      "/mcp": {
        target: process.env.VITE_PROXY_MCP || "http://mcp:8001",
        changeOrigin: true,
        secure: false,
      },
      "/auth": {
        target: process.env.VITE_PROXY_KEYCLOAK || "http://keycloak:8080",
        // Preserve the browser's Host so Keycloak builds external (not internal
        // keycloak:8080) URLs; xfwd forwards X-Forwarded-* which compose's
        // KC_PROXY_HEADERS=xforwarded honors.
        changeOrigin: false,
        secure: false,
        xfwd: true,
      },
    },
    host: true,
    allowedHosts: ["snnbuilder.riken.jp", "localhost"],
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
