"""HTTP client from the kernel to the Django backend (workflow MCP tools).

The kernel cannot reach the MCP server directly (different Docker network), so
workflow tool calls are proxied through the backend. The kernel authenticates
with the shared service token, its own JupyterHub token (which tells the backend
which Jupyter space it runs in) and the notebook's ``project_id``; the backend
forwards the Keycloak JWT the browser relayed for that project and space, so the
user's token never enters the kernel. A manually supplied ``user_token`` is
forwarded directly instead. (LLM calls no longer go through here: the Claude Agent SDK
reaches Anthropic via the backend's ``/api/chat/anthropic`` proxy, configured
through ``ANTHROPIC_BASE_URL``.)
"""

from __future__ import annotations

from .config import AgentConfig


class BackendError(RuntimeError):
    pass


class BackendClient:
    def __init__(self, config: AgentConfig):
        self._config = config

    def _httpx(self):
        import httpx

        return httpx

    def _auth(self) -> tuple[dict, dict]:
        """Return ``(headers, extra_params)`` for the user-scoped proxies."""
        cfg = self._config
        if cfg.user_token:
            return {"Authorization": f"Bearer {cfg.user_token}"}, {}
        headers = {"x-api-key": cfg.service_token, "x-jupyterhub-token": cfg.hub_token}
        return headers, {"project_id": cfg.project_id}

    def list_mcp_tools(self) -> list[dict]:
        """Return MCP tools in OpenAI function format (empty if no user token)."""
        if not self._config.has_mcp:
            return []
        httpx = self._httpx()
        url = f"{self._config.backend_url}/api/chat/mcp-tools/"
        headers, params = self._auth()
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                raise BackendError(f"mcp-tools {resp.status_code}: {resp.text}")
            return resp.json().get("tools", [])

    def call_mcp_tool(self, name: str, arguments: dict) -> str:
        httpx = self._httpx()
        url = f"{self._config.backend_url}/api/chat/mcp-call/"
        headers, extra = self._auth()
        payload = {"tool_name": name, "arguments": arguments, **extra}
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return f"[error] mcp-call {resp.status_code}: {resp.text}"
            return resp.json().get("result", "")
