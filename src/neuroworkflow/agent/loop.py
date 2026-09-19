"""The in-kernel agent, backed by the Claude Agent SDK.

The agent loop and its tools run inside the Jupyter kernel so notebook-native
tools (``run_code``, file edits) act on the user's live workspace. The Claude
Agent SDK drives the ``claude`` CLI; the CLI reaches Anthropic through the
backend proxy (``ANTHROPIC_BASE_URL``) so the API key stays on the backend.

``run()`` is synchronous (the magics/widget call it from a kernel cell), so it
drives the async SDK on a private event loop in a worker thread, isolated from
the kernel's own running loop.
"""

from __future__ import annotations

import asyncio
import threading

from .client import BackendClient
from .config import AgentConfig
from .sdk_tools import build_servers, mcp_display_name
from .skills import build_system_prompt

MAX_TURNS = 60

_DESTRUCTIVE_BASH = (
    "rm -rf", "rm -fr", "mkfs", "dd if=", "shutdown", "reboot", "git push",
)


def _run_in_thread(coro_factory):
    """Run an async coroutine on a dedicated loop in a worker thread.

    The Jupyter kernel already owns a running asyncio loop, and the Claude Agent
    SDK spawns a subprocess and drives it with anyio task groups. Re-entering the
    kernel's loop (e.g. via nest_asyncio) corrupts the kernel's own tasks
    ("Task was destroyed but it is pending"). Instead we give the SDK a private
    loop on its own thread and block the caller until it finishes.
    """
    box: dict = {}

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            box["value"] = loop.run_until_complete(coro_factory())
        except BaseException as e:  # propagate to the calling thread
            box["error"] = e
        finally:
            # Close the SDK's async generators before tearing down the loop so a
            # pending query() athrow isn't orphaned ("Task was destroyed").
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            asyncio.set_event_loop(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value", "")


def _make_can_use_tool(workspace_root: str):
    import os

    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    root = os.path.realpath(workspace_root)

    async def can_use_tool(tool_name, tool_input, context):
        if tool_name.startswith("mcp__"):
            return PermissionResultAllow()
        if tool_name in ("Read", "Glob", "Grep", "TodoWrite", "NotebookRead"):
            return PermissionResultAllow()
        if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            path = (
                tool_input.get("file_path")
                or tool_input.get("path")
                or tool_input.get("notebook_path")
                or ""
            )
            candidate = path if os.path.isabs(path) else os.path.join(root, path)
            resolved = os.path.realpath(candidate)
            if resolved == root or resolved.startswith(root + os.sep):
                return PermissionResultAllow()
            return PermissionResultDeny(
                message=f"Write blocked: path outside workspace ({root}): {path!r}"
            )
        if tool_name == "Bash":
            command = (tool_input.get("command") or "").lower()
            if any(token in command for token in _DESTRUCTIVE_BASH):
                return PermissionResultDeny(
                    message="Bash blocked: potentially destructive command"
                )
            return PermissionResultAllow()
        return PermissionResultAllow()

    return can_use_tool


async def _prompt_stream(user_message: str):
    # can_use_tool requires streaming-mode input (an async iterable, not a str).
    yield {
        "type": "user",
        "message": {"role": "user", "content": user_message},
        "parent_tool_use_id": None,
    }


class Agent:
    def __init__(self, config: AgentConfig, ipython=None):
        self._config = config
        self._client = BackendClient(config)
        self._ipython = ipython
        self._session_id: str | None = None
        self._servers, self._allowed = build_servers(
            self._client, config, self._get_ipython
        )
        # Advertise workflow tools only when the backend actually listed them
        # (the browser may not have relayed a token yet).
        self._append_prompt = build_system_prompt(
            config.skills_dir,
            with_mcp="workflow" in self._servers,
            project_id=config.project_id,
        )

    def _get_ipython(self):
        if self._ipython is not None:
            return self._ipython
        from IPython import get_ipython

        return get_ipython()

    def run(self, user_message: str, on_text=None, on_tool=None) -> str:
        """Run one user turn to completion; return the final assistant text.

        ``on_text(delta)`` is called for streamed text; ``on_tool(name, args)``
        before each tool runs. Conversation context carries across calls via the
        SDK session (``resume``).
        """
        return _run_in_thread(
            lambda: self._arun(user_message, on_text, on_tool)
        )

    async def _arun(self, user_message, on_text, on_tool) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            StreamEvent,
            TextBlock,
            ToolUseBlock,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": self._append_prompt,
            },
            mcp_servers=self._servers,
            allowed_tools=self._allowed,
            can_use_tool=_make_can_use_tool(self._config.workspace_root),
            permission_mode="default",
            cwd=self._config.workspace_root,
            setting_sources=[],
            model=self._config.anthropic_model,
            max_turns=MAX_TURNS,
            include_partial_messages=True,
            resume=self._session_id,
            env=self._config.cli_env(),
        )

        final_text = ""
        saw_text_stream = False

        async for message in query(
            prompt=_prompt_stream(user_message), options=options
        ):
            if isinstance(message, StreamEvent):
                event = message.event or {}
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        saw_text_stream = True
                        if on_text:
                            on_text(delta["text"])
            elif isinstance(message, AssistantMessage):
                if message.session_id:
                    self._session_id = message.session_id
                for block in message.content:
                    if isinstance(block, TextBlock):
                        final_text += block.text
                        if not saw_text_stream and on_text and block.text:
                            on_text(block.text)
                    elif isinstance(block, ToolUseBlock) and on_tool:
                        on_tool(mcp_display_name(block.name), block.input)
            elif isinstance(message, ResultMessage):
                if message.session_id:
                    self._session_id = message.session_id
                # Do NOT break here: let query()'s async generator finish on its
                # own. Breaking early triggers a messy aclose() inside the SDK
                # ("aclose(): asynchronous generator is already running").

        return final_text
