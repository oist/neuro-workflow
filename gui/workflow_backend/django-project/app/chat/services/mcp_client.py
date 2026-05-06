import httpx
import json
import logging
import os

logger = logging.getLogger(__name__)

MCP_SERVER_URL = (
    os.environ.get("MCP_SERVER_URL")
    or os.environ.get("MCP_PROXY_URL", "http://mcp:8001")
)
MCP_ENDPOINT = f"{MCP_SERVER_URL}/mcp"


class MCPClient:
    """Client for communicating with the MCP Streamable HTTP server.

    Lifecycle: a new instance is created per request (in ``orchestrate_chat``)
    and discarded when the request finishes.  It is **not** shared across
    requests or stored in long-lived state. The ``auth_token`` carries the
    end-user's JWT and is forwarded to every MCP call so the upstream
    workflow_mcp tools can attach it when calling the Django API.
    """

    def __init__(self, auth_token: str | None = None):
        self._request_id = 0
        self._session_id: str | None = None
        self._tools_cache = None
        self._auth_token = auth_token

    def _next_id(self):
        self._request_id += 1
        return self._request_id

    async def _rpc(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request to the MCP proxy (Streamable HTTP)."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._next_id(),
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(MCP_ENDPOINT, json=payload, headers=headers)
            resp.raise_for_status()

            # Capture session ID from response header
            session_id = resp.headers.get("mcp-session-id")
            if session_id:
                self._session_id = session_id

            content_type = resp.headers.get("content-type", "")

            # Streamable HTTP may respond with SSE or JSON
            if "text/event-stream" in content_type:
                return self._parse_sse_response(resp.text)
            else:
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"MCP error: {data['error']}")
                return data.get("result", {})

    def _parse_sse_response(self, text: str) -> dict:
        """Parse SSE response from MCP Streamable HTTP and extract the JSON-RPC result.

        JSON-RPC over SSE may emit multiple ``data:`` lines (e.g. progress
        notifications).  We iterate through all of them but only the **last**
        ``result`` value is used as the return value.
        """
        result = {}
        for line in text.split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    if "error" in data:
                        raise RuntimeError(f"MCP error: {data['error']}")
                    if "result" in data:
                        result = data["result"]
                except json.JSONDecodeError:
                    continue
        return result

    async def initialize(self) -> dict:
        """Initialize the MCP session."""
        return await self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "neuro-chat", "version": "1.0.0"},
        })

    async def list_tools(self) -> list[dict]:
        """Get the list of available MCP tools."""
        if self._tools_cache is not None:
            return self._tools_cache
        result = await self._rpc("tools/list")
        self._tools_cache = result.get("tools", [])
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute an MCP tool and return the text result."""
        result = await self._rpc("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # MCP returns content as a list of content blocks
        content_blocks = result.get("content", [])
        texts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else str(result)

    def invalidate_cache(self):
        """Clear the tools cache."""
        self._tools_cache = None


def mcp_tools_to_openai_functions(mcp_tools: list[dict]) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function calling format.

    MCP inputSchema is JSON Schema compatible, so the conversion is straightforward.
    """
    functions = []
    for tool in mcp_tools:
        func = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
            },
        }
        functions.append(func)
    return functions
