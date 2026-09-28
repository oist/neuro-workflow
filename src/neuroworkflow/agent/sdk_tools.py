"""In-process MCP tools for the notebook Claude agent.

Two in-process MCP servers are handed to the Claude Agent SDK:

- ``notebook`` — a single ``run_code`` tool that executes in the live IPython
  kernel (shared namespace, inline display). It is in-process rather than the
  SDK's Bash tool so Python runs against the notebook namespace.
- ``workflow`` — one tool per backend MCP workflow tool, each wrapping the
  backend proxy (``/api/chat/mcp-call/``). The kernel cannot reach the MCP
  server directly (different Docker network), and the proxy forwards the user's
  Keycloak JWT for per-user scoping — the one the browser relayed for this
  notebook's project, or an explicit ``user_token``. Only present when the
  backend lists tools for this config.

File editing (Read/Write/Edit) uses the SDK's built-in tools.
"""

import asyncio
import logging

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import tools as nb_tools
from .client import BackendClient
from .config import AgentConfig

logger = logging.getLogger(__name__)

_MCP_PREFIXES = ("mcp__notebook__", "mcp__workflow__")

# Namespace used by run_code when there is no live IPython (non-kernel use).
_FALLBACK_NS: dict = {}


def _resolve_namespace(get_ipython) -> dict:
    ip = get_ipython()
    user_ns = getattr(ip, "user_ns", None)
    return user_ns if user_ns is not None else _FALLBACK_NS


def mcp_display_name(name: str) -> str:
    """Strip the ``mcp__<server>__`` prefix for user-facing tool announcements."""
    for prefix in _MCP_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def build_servers(client: BackendClient, config: AgentConfig, get_ipython):
    """Return ``(mcp_servers, allowed_tools)`` for ``ClaudeAgentOptions``."""

    @tool(
        "run_code",
        "Execute Python in the user's live Jupyter kernel. Variables persist "
        "across calls (shared namespace) and the output is shown to the user. "
        "Prefer this over Bash for Python; take small, verifiable steps.",
        {"code": str},
    )
    async def run_code(args):
        text = nb_tools.run_code(args.get("code", ""), _resolve_namespace(get_ipython))
        return {"content": [{"type": "text", "text": text}]}

    servers = {"notebook": create_sdk_mcp_server("notebook", tools=[run_code])}
    allowed = ["Read", "Write", "Edit", "Bash", "mcp__notebook__run_code"]

    if config.has_mcp:
        workflow_tools = []
        try:
            for spec in client.list_mcp_tools():
                fn = spec.get("function", {})
                name = fn.get("name")
                if not name:
                    continue
                workflow_tools.append(
                    _make_workflow_tool(
                        client,
                        name,
                        fn.get("description", ""),
                        fn.get("parameters") or {"type": "object", "properties": {}},
                    )
                )
        except Exception as exc:
            logger.warning(
                "Workflow MCP tools disabled — could not list backend MCP tools: %s",
                exc,
            )
            workflow_tools = []
        if workflow_tools:
            servers["workflow"] = create_sdk_mcp_server("workflow", tools=workflow_tools)
            allowed.append("mcp__workflow")

    return servers, allowed


def _make_workflow_tool(client: BackendClient, name: str, description: str, schema: dict):
    @tool(name, description, schema)
    async def _call(args):
        # call_mcp_tool is a blocking HTTP call; run it off the event loop.
        result = await asyncio.to_thread(client.call_mcp_tool, name, args)
        return {"content": [{"type": "text", "text": result}]}

    return _call
