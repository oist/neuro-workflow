/**
 * Centralized URL configuration for external services.
 *
 * In local development these resolve to localhost:PORT.
 * Behind an nginx reverse proxy (e.g. RIKEN deployment) they become
 * relative paths like "/jupyter", "/api", etc.
 *
 * All values are controlled via VITE_* environment variables set in .env
 * or injected by Docker Compose.
 */

/** Base URL the browser uses to reach JupyterHub / JupyterLab. */
export const JUPYTER_BASE_URL: string =
  import.meta.env.VITE_JUPYTER_BASE_URL?.replace(/\/+$/, '') ||
  (() => {
    try {
      if (typeof window === 'undefined') return 'http://localhost:8000';
      const { protocol, hostname, host } = window.location;
      if (host.includes(':')) {
        return `${protocol}//${hostname}:8000`;
      }
      return `${protocol}//${host}`;
    } catch {
      return 'http://localhost:8000';
    }
  })();

/** Base URL the browser uses to reach the Django API. */
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') || '/api';

/** Base URL the browser uses to reach the MCP proxy. */
export const MCP_BASE_URL: string =
  import.meta.env.VITE_MCP_BASE_URL?.replace(/\/+$/, '') || '/mcp';

/**
 * Base URL the browser uses to reach the bm_mindsdb (mdb) catalog UI.
 *
 * Same-origin on purpose: mdb has no authentication and its port is not
 * published, so it is reached only through the Vite dev proxy / nginx under
 * this prefix. Keeping it same-origin also avoids mixed-content blocking when
 * the app is served over HTTPS.
 */
export const MDB_BASE_URL: string =
  import.meta.env.VITE_MDB_BASE_URL?.replace(/\/+$/, '') || '/mdb';
