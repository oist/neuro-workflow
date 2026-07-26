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
        target: process.env.VITE_PROXY_BACKEND || "http://localhost:3000",
        changeOrigin: true,
        secure: false,
      },
      "/jupyter": {
        target: process.env.VITE_PROXY_JUPYTER || "http://localhost:8000",
        changeOrigin: true,
        secure: false,
        ws: true,
      },
      "/mcp": {
        target: process.env.VITE_PROXY_MCP || "http://localhost:8001",
        changeOrigin: true,
        secure: false,
      },
      // bm_mindsdb (mdb) catalog UI. The prefix is stripped before forwarding
      // and echoed back in X-Forwarded-Prefix, which mdb uses (via ProxyFix)
      // to build its static URLs and its fetch base path.
      "/mdb": {
        target: process.env.VITE_PROXY_MDB || "http://localhost:8004",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/mdb/, ""),
        headers: { "X-Forwarded-Prefix": "/mdb" },
      },
      "/auth": {
        target: process.env.VITE_PROXY_KEYCLOAK || "http://localhost:8080",
        changeOrigin: true,
        secure: false,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
