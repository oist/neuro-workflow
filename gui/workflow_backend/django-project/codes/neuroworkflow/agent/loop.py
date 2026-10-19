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
import contextvars
import sys
import threading

from .client import BackendClient
from .config import AgentConfig
from .record import TurnRecorder, redact
from .sdk_tools import build_servers, mcp_display_name
from .skills import build_system_prompt

MAX_TURNS = 90

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

    # Run in a copy of the caller's context so the worker keeps the caller's
    # ipykernel parent request. Without it the thread falls back to whichever
    # message the kernel handled last (e.g. a widget's outputs sync), and
    # streamed text is routed to that parent instead of the ChatPanel/cell.
    ctx = contextvars.copy_context()
    thread = threading.Thread(target=ctx.run, args=(worker,), daemon=True)
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


def _cli_stderr(line: str):
    """Relay the CLI's stderr, minus its per-turn warning about a non-Claude model."""
    if not line.lstrip().startswith("[claude-code:unrecognized_model]"):
        print(line, file=sys.stderr)


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
        # Backend id of this agent's saved conversation (set by the first turn).
        self._conversation_id: str | None = None
        self._record_warned = False
        self._build_tools()

    def _build_tools(self):
        self._servers = build_servers(
            self._client, self._config, self._get_ipython
        )
        # Advertise workflow tools only when the backend actually listed them
        # (the browser may not have relayed a token yet).
        self._append_prompt = build_system_prompt(
            self._config.skills_dir,
            with_mcp="workflow" in self._servers,
            project_id=self._config.project_id,
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
        if self._config.has_mcp and "workflow" not in self._servers:
            # The browser may have relayed its token since the last attempt.
            self._build_tools()
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
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": self._append_prompt,
            },
            mcp_servers=self._servers,
            # No allowed_tools: an allow entry auto-approves the tool before
            # can_use_tool runs, which would bypass the workspace/Bash guards.
            can_use_tool=_make_can_use_tool(self._config.workspace_root),
            permission_mode="default",
            cwd=self._config.workspace_root,
            setting_sources=[],
            model=self._config.model,
            max_turns=MAX_TURNS,
            include_partial_messages=True,
            resume=self._session_id,
            env=self._config.cli_env(),
            stderr=_cli_stderr if self._config.provider_model else None,
        )

        final_text = ""
        saw_text_stream = False
        # Messages of a subagent (parent_tool_use_id set) are not recorded; the
        # Task tool's own call and result are.
        rec = TurnRecorder(user_message)
        error = None

        try:
            async for message in query(
                prompt=_prompt_stream(user_message), options=options
            ):
                top_level = not getattr(message, "parent_tool_use_id", None)
                if isinstance(message, StreamEvent):
                    event = message.event or {}
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            saw_text_stream = True
                            if top_level:
                                rec.add_text_delta(delta["text"])
                            if on_text:
                                on_text(delta["text"])
                elif isinstance(message, AssistantMessage):
                    if message.session_id:
                        self._session_id = message.session_id
                    if top_level:
                        rec.model = rec.model or getattr(message, "model", None)
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            final_text += block.text
                            if top_level:
                                rec.add_text(block.text)
                            if not saw_text_stream and on_text and block.text:
                                on_text(block.text)
                        elif isinstance(block, ToolUseBlock):
                            name = mcp_display_name(block.name)
                            if top_level:
                                rec.add_tool_use(block.id, name, block.input)
                            if on_tool:
                                on_tool(name, block.input)
                elif isinstance(message, UserMessage):
                    # Tool results come back as user messages with content blocks.
                    if top_level and isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                rec.add_tool_result(
                                    block.tool_use_id, block.content, block.is_error
                                )
                elif isinstance(message, ResultMessage):
                    if message.session_id:
                        self._session_id = message.session_id
                    # Do NOT break here: let query()'s async generator finish on its
                    # own. Breaking early triggers a messy aclose() inside the SDK
                    # ("aclose(): asynchronous generator is already running").
        except BaseException as e:
            error = e
            raise
        finally:
            self._record(rec.finish(error), rec.model)

        return final_text

    def _record(self, messages: list[dict], model: str | None = None):
        """Save the turn on the backend; a failure never breaks the chat."""
        cfg = self._config
        if not cfg.project_id:
            # Saved conversations are read per project; nowhere to file this one.
            return
        payload = redact(
            {
                "project_id": cfg.project_id,
                "conversation_id": self._conversation_id,
                "model": cfg.model or model,
                "session_id": self._session_id,
                "messages": messages,
            },
            [cfg.service_token, cfg.hub_token, cfg.user_token],
        )
        try:
            self._conversation_id = self._client.record_turn(payload)
        except Exception as e:
            if not self._record_warned:
                self._record_warned = True
                print(
                    f"[neuroworkflow.agent] conversation not saved: {e}",
                    file=sys.stderr,
                )
