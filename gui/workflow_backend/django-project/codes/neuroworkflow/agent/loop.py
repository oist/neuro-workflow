"""The in-kernel agent, backed by the Claude Agent SDK.

The agent loop and its tools run inside the Jupyter kernel so notebook-native
tools (``run_code``, file edits) act on the user's live workspace. The Claude
Agent SDK drives the ``claude`` CLI; the CLI reaches Anthropic through the
backend proxy (``ANTHROPIC_BASE_URL``) so the API key stays on the backend.

The SDK runs on a private event loop in a worker thread, isolated from the
kernel's own running loop. Everything that touches the notebook (streamed
text, tool announcements, approval prompts, ``run_code``) is sent back to the
kernel's main thread through a ``dispatch`` function, so it never runs at the
same time as a user's cell:

- ``start()`` returns at once (``ChatPanel``: dispatch is the kernel's IO loop,
  so the panel's buttons keep working while the agent runs);
- ``run()`` blocks the calling cell and services the dispatched work itself
  (``%chat``, ``chat()``).

Every tool call passes the permission gate (``permissions.py``): allow, deny,
or ask the user through ``approve``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import logging
import os
import queue
import sys
import threading
import time

from .client import BackendClient
from .config import AgentConfig
from .permissions import (
    PermissionDecision,
    PermissionRequest,
    Permissions,
    Policy,
    describe,
    tool_key,
)
from .record import TurnRecorder, redact
from .sdk_tools import build_servers, mcp_display_name
from .skills import build_system_prompt

logger = logging.getLogger(__name__)

MAX_TURNS = 90

# How long an approval prompt in the ChatPanel waits before the call is denied.
APPROVAL_TIMEOUT_S = 600

_DESTRUCTIVE_BASH = (
    "rm -rf", "rm -fr", "mkfs", "dd if=", "shutdown", "reboot", "git push",
)


class RunHandle:
    """A running agent turn.

    ``done`` resolves to the turn's final text, or is cancelled when the run
    was stopped with ``cancel()``.
    """

    def __init__(self):
        self.done: concurrent.futures.Future = concurrent.futures.Future()
        self._lock = threading.Lock()
        self._loop = None
        self._task = None
        self._cancelled = False

    def _bind(self, loop, task):
        with self._lock:
            self._loop, self._task = loop, task
            if self._cancelled:
                task.cancel()

    def _unbind(self):
        with self._lock:
            self._loop = self._task = None

    def cancel(self):
        with self._lock:
            self._cancelled = True
            if self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)


class _Run:
    """One turn's bridge from the SDK worker thread to the main thread."""

    def __init__(
        self, dispatch, handle, on_text, on_tool, on_output, approve, timeout
    ):
        self._dispatch = dispatch
        self.handle = handle
        self.on_text = on_text
        self.on_tool = on_tool
        self.on_output = on_output
        self.approve = approve
        self.approval_timeout = timeout

    def call_main(self, fn, *args) -> concurrent.futures.Future:
        """Run ``fn(*args)`` on the main thread; the future gets its result."""
        future: concurrent.futures.Future = concurrent.futures.Future()

        def job():
            if future.cancelled():
                return
            try:
                result = fn(*args)
            except KeyboardInterrupt:
                # e.g. Ctrl-C at an input() prompt: let the caller stop the turn.
                future.cancel()
                raise
            except BaseException as e:
                result, error = None, e
            else:
                error = None
            try:
                if error is None:
                    future.set_result(result)
                else:
                    future.set_exception(error)
            except concurrent.futures.InvalidStateError:
                pass  # cancelled meanwhile (approval timed out)

        self._dispatch(job)
        return future

    def notify(self, fn, *args):
        """Call a frontend callback on the main thread, ignoring its result."""
        if fn is None:
            return

        def job():
            try:
                fn(*args)
            except Exception:
                logger.exception("notebook agent callback failed")

        self._dispatch(job)


def _pump(jobs: queue.Queue, done: concurrent.futures.Future, timeout=None):
    """Run the jobs a worker dispatches until ``done``, then the rest."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while not done.done():
        if deadline is not None and time.monotonic() > deadline:
            return
        try:
            job = jobs.get(timeout=0.05)
        except queue.Empty:
            continue
        job()
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        job()


def _hard_deny(root: str, tool: str, tool_input: dict) -> str | None:
    """Limits that apply whatever the mode: writes stay inside the workspace,
    and obviously destructive shell commands are refused."""
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        path = (
            tool_input.get("file_path")
            or tool_input.get("path")
            or tool_input.get("notebook_path")
            or ""
        )
        candidate = path if os.path.isabs(path) else os.path.join(root, path)
        resolved = os.path.realpath(candidate)
        if resolved == root or resolved.startswith(root + os.sep):
            return None
        return f"Write blocked: path outside workspace ({root}): {path!r}"
    if tool == "Bash":
        command = (tool_input.get("command") or "").lower()
        if any(token in command for token in _DESTRUCTIVE_BASH):
            return "Bash blocked: potentially destructive command"
    return None


async def _ask(run: _Run, request: PermissionRequest) -> PermissionDecision:
    """Ask the user on the main thread; no answer in time means deny.

    ``approve(request)`` returns the decision (``%chat``: it prompts and
    blocks) or a future the UI resolves later (``ChatPanel``).
    """
    if run.approve is None:
        return PermissionDecision(allow=False, source="none")

    async def answer():
        decision = await asyncio.wrap_future(run.call_main(run.approve, request))
        if isinstance(decision, concurrent.futures.Future):
            decision = await asyncio.wrap_future(decision)
        return decision

    try:
        # On timeout this cancels the UI's future, which disables its prompt.
        return await asyncio.wait_for(answer(), run.approval_timeout)
    except asyncio.TimeoutError:
        return PermissionDecision(allow=False, source="timeout")


def _deny_message(request, decision, timeout) -> str:
    what = f"{request.tool} ({request.category})"
    if decision.source == "hard":
        return decision.reason
    if decision.source == "policy":
        return (
            f"Blocked by the notebook tool policy: {what} is set to deny. "
            "Continue without it."
        )
    if decision.source == "timeout":
        return (
            f"No approval for {what} within {timeout:.0f} s, so it was not run. "
            "Ask the user before trying again."
        )
    if decision.source == "none":
        return (
            f"{what} needs the user's approval and nobody can approve it here, "
            "so it was not run."
        )
    reason = f": {decision.reason}" if decision.reason else ""
    return (
        f"The user declined this {request.tool} call{reason}. Do not retry it; "
        "continue without it or ask the user."
    )


def _make_can_use_tool(
    workspace_root: str, permissions: Permissions, run: _Run, on_decision=None
):
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    root = os.path.realpath(workspace_root)

    async def can_use_tool(tool_name, tool_input, context):
        tool = tool_key(tool_name)
        request = PermissionRequest(
            tool=tool,
            category=permissions.category(tool, tool_name),
            input=tool_input,
            summary=describe(tool, tool_input),
            can_always_allow=permissions.can_always_allow(tool, tool_name),
        )
        hard = _hard_deny(root, tool, tool_input)
        mode = permissions.mode(tool, tool_name)
        if hard:
            decision = PermissionDecision(allow=False, reason=hard, source="hard")
        elif mode == "allow":
            decision = PermissionDecision(allow=True, source="policy")
        elif mode == "deny":
            decision = PermissionDecision(allow=False, source="policy")
        else:
            decision = await _ask(run, request)
            if decision.allow and decision.always:
                permissions.allow_for_conversation(tool, tool_name)
        if on_decision is not None:
            on_decision(request, decision)
        if decision.allow:
            return PermissionResultAllow()
        message = _deny_message(request, decision, run.approval_timeout)
        return PermissionResultDeny(message=message)

    return can_use_tool


async def _ask_every_tool(input_data, tool_use_id, context):
    """PreToolUse hook: send every tool call to ``can_use_tool``.

    Without it the CLI approves reads inside the working directory and the
    shell commands it considers read-only (``ls``, ``grep``, …) on its own,
    so their ask/deny settings would be ignored.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": "notebook tool policy",
        }
    }


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
    def __init__(self, config: AgentConfig, ipython=None, permissions=None):
        self._config = config
        self._client = BackendClient(config)
        self._ipython = ipython
        self._session_id: str | None = None
        # Backend id of this agent's saved conversation (set by the first turn).
        self._conversation_id: str | None = None
        self._record_warned = False
        self.permissions: Permissions = permissions or Permissions()
        # Called on the worker thread as on_decision(request, decision) for
        # every tool call.
        self.on_decision = None
        self._busy = threading.Lock()
        self._run: _Run | None = None
        # Fetched now so the user's settings are checked against the real
        # limits before the first turn, not the fallback ones.
        self._refresh_policy()
        self._build_tools()

    def _build_tools(self):
        self._servers, workflow_tools = build_servers(
            self._client, self._config, self._get_ipython, lambda: self._run
        )
        self.permissions.workflow_tools = frozenset(workflow_tools)
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

    def _refresh_policy(self) -> bool:
        """Fetch the admin ceilings (keeping the last known ones on failure);
        return whether they changed."""
        try:
            policy = Policy.from_json(self._client.get_notebook_policy())
        except Exception as e:
            logger.warning("Notebook tool policy not refreshed: %s", e)
            return False
        return self.permissions.refresh(policy)

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def start(
        self,
        user_message: str,
        *,
        dispatch,
        on_text=None,
        on_tool=None,
        on_output=None,
        approve=None,
        approval_timeout=APPROVAL_TIMEOUT_S,
    ) -> RunHandle:
        """Start one user turn and return at once.

        ``dispatch(fn)`` must call ``fn()`` soon on the main thread; the
        callbacks, ``approve`` and ``run_code`` run through it. ``on_text(delta)``
        gets streamed text, ``on_tool(name, args)`` each tool call,
        ``on_output(captured)`` what ``run_code`` displayed (IPython's
        ``CapturedIO``; default: shown in the current cell), and
        ``approve(PermissionRequest)`` returns a ``PermissionDecision`` or a
        future of one. Without ``approve`` every "ask" is denied. Conversation
        context carries across turns via the SDK session (``resume``).
        """
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("The agent is still answering; wait or stop it.")
        try:
            policy_changed = self._refresh_policy()
            # Rebuild the workflow tools when the browser may have relayed its
            # token since the last attempt, or when the policy changed: the
            # backend leaves out the tools the admin denies.
            if self._config.has_mcp and (
                policy_changed or "workflow" not in self._servers
            ):
                self._build_tools()
        except BaseException:
            self._busy.release()
            raise
        handle = RunHandle()
        run = _Run(
            dispatch, handle, on_text, on_tool, on_output, approve, approval_timeout
        )
        self._run = run

        def worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            outcome = None
            try:
                task = loop.create_task(self._arun(user_message, run))
                handle._bind(loop, task)
                outcome = ("result", loop.run_until_complete(task))
            except asyncio.CancelledError:
                outcome = ("cancelled", None)
            except BaseException as e:  # handed to the caller through ``done``
                outcome = ("error", e)
            finally:
                handle._unbind()
                # Close the SDK's async generators before tearing down the loop
                # so a pending query() athrow isn't orphaned ("Task was
                # destroyed").
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()
                asyncio.set_event_loop(None)
                self._run = None
                # Release before resolving ``done``: its callbacks may start
                # the next turn.
                self._busy.release()
            kind, value = outcome
            if kind == "result":
                handle.done.set_result(value)
            elif kind == "cancelled":
                handle.done.cancel()
            else:
                handle.done.set_exception(value)

        # Run in a copy of the caller's context so the worker keeps the caller's
        # ipykernel parent request: anything it prints (warnings) lands in the
        # cell or panel that started the turn, not wherever the kernel was last.
        ctx = contextvars.copy_context()
        threading.Thread(target=ctx.run, args=(worker,), daemon=True).start()
        return handle

    def run(
        self, user_message: str, on_text=None, on_tool=None, on_output=None,
        approve=None,
    ) -> str:
        """Run one user turn to completion; return the final assistant text.

        The calling thread services the callbacks while it waits, so
        ``approve`` may block (e.g. on ``input()``); there is no approval
        timeout. Ctrl-C / kernel interrupt stops the turn.
        """
        jobs: queue.Queue = queue.Queue()
        handle = self.start(
            user_message,
            dispatch=jobs.put,
            on_text=on_text,
            on_tool=on_tool,
            on_output=on_output,
            approve=approve,
            approval_timeout=None,
        )
        try:
            _pump(jobs, handle.done)
        except KeyboardInterrupt:
            handle.cancel()
            _pump(jobs, handle.done, timeout=10)  # let the CLI shut down
            print("\n[interrupted]")
            return ""
        if handle.done.cancelled():
            return ""
        return handle.done.result()

    async def _arun(self, user_message: str, run: _Run) -> str:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
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
            # can_use_tool runs, which would bypass the permission gate.
            can_use_tool=_make_can_use_tool(
                self._config.workspace_root, self.permissions, run, self.on_decision
            ),
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_ask_every_tool])]},
            # Hide what is set to deny so the model does not try it.
            disallowed_tools=self.permissions.denied_builtins(),
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
        rec = TurnRecorder(user_message, self._secrets())
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
                            run.notify(run.on_text, delta["text"])
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
                            if not saw_text_stream and block.text:
                                run.notify(run.on_text, block.text)
                        elif isinstance(block, ToolUseBlock):
                            name = mcp_display_name(block.name)
                            if top_level:
                                rec.add_tool_use(block.id, name, block.input)
                            run.notify(run.on_tool, name, block.input)
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

    def _secrets(self) -> list:
        cfg = self._config
        return [cfg.service_token, cfg.hub_token, cfg.user_token]

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
            self._secrets(),
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
