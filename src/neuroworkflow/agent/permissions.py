"""Tool permissions of the notebook agent: allow / ask / deny (Issue #123).

Every tool belongs to a category; each category (or a single tool) is set to
``allow`` (run it), ``ask`` (the user approves each call) or ``deny``. Users
change the modes for their session; staff set ceilings on the backend
(``/api/chat/notebook-policy/``), the loosest mode a user may choose.

All of this is decided inside the kernel, which the user controls, so it
protects users from their own agent, not the platform from users. Once
``run_code`` or ``Bash`` is allowed, the agent can lift these limits itself;
only a deny ceiling on a workflow tool is also enforced by the backend.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MODES = ("allow", "ask", "deny")  # loosest to strictest

CATEGORIES = (
    "read",
    "workflow-read",
    "workflow-edit",
    "file-write",
    "code-exec",
    "web",
    "other",
)

DEFAULTS = {
    "read": "allow",
    "workflow-read": "allow",
    "workflow-edit": "ask",
    "file-write": "ask",
    "code-exec": "ask",
    "web": "ask",
    "other": "ask",
}

# SDK built-ins and in-process notebook tools; anything else is "other".
# ToolSearch only loads tool definitions for the CLI.
BUILTIN_CATEGORIES = {
    "Read": "read",
    "Glob": "read",
    "Grep": "read",
    "NotebookRead": "read",
    "TodoWrite": "read",
    "ToolSearch": "read",
    "Write": "file-write",
    "Edit": "file-write",
    "MultiEdit": "file-write",
    "NotebookEdit": "file-write",
    "Bash": "code-exec",
    "BashOutput": "code-exec",
    "KillBash": "code-exec",
    "KillShell": "code-exec",
    "run_code": "code-exec",
    "WebFetch": "web",
    "WebSearch": "web",
}

_WORKFLOW_PREFIX = "mcp__workflow__"
_NOTEBOOK_PREFIX = "mcp__notebook__"

SUMMARY_LIMIT = 4000


def stricter(a: str, b: str) -> str:
    return a if MODES.index(a) >= MODES.index(b) else b


def tool_key(sdk_name: str) -> str:
    """Name used by users and the policy: ``mcp__workflow__x`` -> ``x``."""
    for prefix in (_WORKFLOW_PREFIX, _NOTEBOOK_PREFIX):
        if sdk_name.startswith(prefix):
            return sdk_name[len(prefix) :]
    return sdk_name


@dataclass(frozen=True)
class Policy:
    """Admin ceilings (from the backend) and the read-only workflow tools."""

    category_ceilings: dict = field(default_factory=dict)
    tool_ceilings: dict = field(default_factory=dict)
    workflow_read_tools: frozenset = frozenset()

    @classmethod
    def from_json(cls, data: dict) -> "Policy":
        def ceilings(name, valid_key):
            out = {}
            for key, mode in (data.get(name) or {}).items():
                if valid_key(key) and mode in MODES:
                    out[key] = mode
                else:
                    logger.warning(
                        "Ignoring notebook policy %s[%r]=%r", name, key, mode
                    )
            return out

        return cls(
            category_ceilings=ceilings("category_ceilings", CATEGORIES.__contains__),
            tool_ceilings=ceilings("tool_ceilings", bool),
            workflow_read_tools=frozenset(data.get("workflow_read_tools") or ()),
        )

    @classmethod
    def fallback(cls) -> "Policy":
        """Used while the backend's policy is unknown: users may tighten the
        defaults but not loosen them."""
        return cls(category_ceilings=dict(DEFAULTS))


@dataclass
class PermissionRequest:
    tool: str  # tool_key()
    category: str
    input: dict
    summary: str  # describe()
    can_always_allow: bool


@dataclass
class PermissionDecision:
    allow: bool
    reason: str = ""
    always: bool = False
    # user | policy | hard | timeout | none (no approver)
    source: str = "user"


class Permissions:
    """The effective mode of each tool: user choice, capped by the admin ceiling."""

    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy.fallback()
        self._overrides: dict[str, str] = {}
        # Tools the user allowed for the rest of this conversation.
        self._always: set[str] = set()

    def category(self, tool: str, sdk_name: str | None = None) -> str:
        if tool in BUILTIN_CATEGORIES:
            return BUILTIN_CATEGORIES[tool]
        if sdk_name is not None and not sdk_name.startswith(_WORKFLOW_PREFIX):
            return "other"
        # A workflow (MCP) tool; one the backend does not list as read-only
        # counts as an edit.
        if tool in self.policy.workflow_read_tools:
            return "workflow-read"
        return "workflow-edit"

    def ceiling(self, tool: str, category: str) -> str:
        if tool in self.policy.tool_ceilings:
            return self.policy.tool_ceilings[tool]
        return self.policy.category_ceilings.get(category, "allow")

    def mode(self, tool: str, sdk_name: str | None = None) -> str:
        category = self.category(tool, sdk_name)
        base = self._overrides.get(tool) or self._overrides.get(category)
        base = base or DEFAULTS[category]
        if tool in self._always:
            base = "allow"
        return stricter(base, self.ceiling(tool, category))

    def set(self, key: str, mode: str):
        """Set a category or a single tool to ``mode`` for this session."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        if key in CATEGORIES:
            ceiling = self.policy.category_ceilings.get(key, "allow")
        else:
            ceiling = self.ceiling(key, self.category(key))
        if stricter(mode, ceiling) != mode:
            raise ValueError(f"{key}: the administrator's limit is {ceiling!r}")
        self._overrides[key] = mode

    def reset(self):
        """Back to the defaults (the admin ceilings still apply)."""
        self._overrides.clear()
        self._always.clear()

    def can_always_allow(self, tool: str, sdk_name: str | None = None) -> bool:
        return self.ceiling(tool, self.category(tool, sdk_name)) == "allow"

    def allow_for_conversation(self, tool: str, sdk_name: str | None = None):
        if self.can_always_allow(tool, sdk_name):
            self._always.add(tool)

    def new_conversation(self):
        """Keep the user's settings; "always allow" lasts one conversation."""
        self._always.clear()

    def refresh(self, policy: Policy):
        self.policy = policy

    def denied_builtins(self) -> list[str]:
        """SDK tool names currently set to deny (hidden from the model)."""
        names = [
            name
            for name in BUILTIN_CATEGORIES
            if name != "run_code" and self.mode(name, name) == "deny"
        ]
        if self.mode("run_code", _NOTEBOOK_PREFIX + "run_code") == "deny":
            names.append(_NOTEBOOK_PREFIX + "run_code")
        return names

    def table(self) -> str:
        """Category modes, user overrides and admin limits, for display."""
        rows = [f"{'category':<14} {'mode':<6} admin limit"]
        for category in CATEGORIES:
            base = self._overrides.get(category) or DEFAULTS[category]
            ceiling = self.policy.category_ceilings.get(category, "allow")
            rows.append(f"{category:<14} {stricter(base, ceiling):<6} {ceiling}")
        tools = sorted(set(self._overrides) - set(CATEGORIES))
        tools += sorted(set(self.policy.tool_ceilings) - set(tools))
        for tool in tools:
            ceiling = self.ceiling(tool, self.category(tool))
            rows.append(f"{tool:<14} {self.mode(tool):<6} {ceiling}")
        if self._always:
            rows.append(
                "always allowed in this conversation: "
                + ", ".join(sorted(self._always))
            )
        return "\n".join(rows)

    def __repr__(self):
        return self.table()


def _clip(text: str, limit: int = SUMMARY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…({len(text) - limit} chars truncated)"


def describe(tool: str, tool_input: dict) -> str:
    """What the user is asked to approve: the code, command or change."""
    i = tool_input or {}
    if tool == "run_code":
        text = i.get("code", "")
    elif tool == "Bash":
        text = i.get("command", "")
        if i.get("description"):
            text = f"# {i['description']}\n{text}"
    elif tool == "Write":
        lines = (i.get("content") or "").splitlines()
        more = f"\n…({len(lines) - 40} more lines)" if len(lines) > 40 else ""
        text = f"{i.get('file_path', '')}\n" + "\n".join(lines[:40]) + more
    elif tool == "Edit":
        text = (
            f"{i.get('file_path', '')}\n--- old\n{i.get('old_string', '')}"
            f"\n+++ new\n{i.get('new_string', '')}"
        )
    elif tool == "MultiEdit":
        parts = [i.get("file_path", "")]
        for edit in i.get("edits") or []:
            parts.append(
                f"--- old\n{edit.get('old_string', '')}\n+++ new\n"
                f"{edit.get('new_string', '')}"
            )
        text = "\n".join(parts)
    elif tool == "NotebookEdit":
        text = (
            f"{i.get('notebook_path', '')} cell={i.get('cell_id', '')} "
            f"mode={i.get('edit_mode', 'replace')}\n{i.get('new_source', '')}"
        )
    elif tool == "WebFetch":
        text = f"{i.get('url', '')}\n{i.get('prompt', '')}"
    elif tool == "WebSearch":
        text = i.get("query", "")
    else:
        text = json.dumps(i, indent=1, ensure_ascii=False, default=str)
    return _clip(text)


def parse_answer(text: str, can_always: bool) -> PermissionDecision:
    """``y`` approves, ``a`` approves for the conversation, anything else
    denies (non-empty text other than ``n`` is the reason given to the agent)."""
    answer = (text or "").strip()
    lowered = answer.lower()
    if lowered in ("y", "yes"):
        return PermissionDecision(allow=True)
    if lowered in ("a", "always"):
        # Without "always" (admin limit is ask) it approves this call only.
        return PermissionDecision(allow=True, always=can_always)
    if lowered in ("", "n", "no"):
        return PermissionDecision(allow=False)
    return PermissionDecision(allow=False, reason=answer)
