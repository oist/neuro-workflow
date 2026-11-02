"""Recording of agent turns, saved on the backend so people can read them later.

Each turn is sent as OpenAI-shaped messages (the shape of the backend's chat
``Message`` rows) to ``/api/chat/notebook-conversations/``. Tool arguments and
results are clipped, and the kernel's tokens are redacted before sending; in
tool data that happens before clipping, so no fragment of a token survives a cut.
"""

from __future__ import annotations

import json

TOOL_TEXT_LIMIT = 4000
REDACTED = "[REDACTED]"


def clip(text: str, limit: int = TOOL_TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…({len(text) - limit} chars truncated)"


def redact(value, secrets):
    """Replace every secret in the strings of ``value`` (nested dicts/lists)."""
    secrets = [s for s in secrets if s and len(s) >= 8]
    if not secrets:
        return value

    def walk(v):
        if isinstance(v, str):
            for secret in secrets:
                v = v.replace(secret, REDACTED)
            return v
        if isinstance(v, dict):
            return {k: walk(item) for k, item in v.items()}
        if isinstance(v, list):
            return [walk(item) for item in v]
        return v

    return walk(value)


def tool_result_text(content) -> str:
    """Text of a ``ToolResultBlock.content`` (a str, content blocks or None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            kind = item.get("type") if isinstance(item, dict) else None
            parts.append(f"[{kind or 'content'}]")
    return "\n".join(parts)


class TurnRecorder:
    """Collect one user turn as messages for the backend."""

    def __init__(self, user_message: str, secrets=()):
        self.messages: list[dict] = [{"role": "user", "content": user_message}]
        self._secrets = list(secrets)
        self.model: str | None = None
        self._tool_names: dict[str, str] = {}
        # Text streamed since the last complete assistant message.
        self._streamed = ""

    def add_text_delta(self, text: str):
        self._streamed += text

    def _clip(self, text: str) -> str:
        # Redact first: a token cut in half by clip() would no longer match.
        return clip(redact(text, self._secrets))

    def _assistant(self) -> dict:
        # Consecutive text/tool blocks belong to one assistant message.
        if self.messages[-1]["role"] != "assistant":
            self.messages.append({"role": "assistant", "content": ""})
        return self.messages[-1]

    def add_text(self, text: str):
        self._streamed = ""
        self._assistant()["content"] += text

    def add_tool_use(self, tool_id: str, name: str, args):
        self._streamed = ""
        self._tool_names[tool_id] = name
        arguments = self._clip(json.dumps(args, ensure_ascii=False, default=str))
        self._assistant().setdefault("tool_calls", []).append(
            {
                "id": tool_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    def add_tool_result(self, tool_use_id: str, content, is_error):
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "tool_name": self._tool_names.get(tool_use_id, ""),
                "content": self._clip(tool_result_text(content)),
                "is_error": bool(is_error),
            }
        )

    def finish(self, error: BaseException | None = None) -> list[dict]:
        """Return the turn's messages, marking an interrupted or failed turn."""
        if self._streamed:
            # The text streamed, but its complete message never arrived.
            self._assistant()["content"] += self._streamed + " …(interrupted)"
            self._streamed = ""
        if error is not None:
            self._assistant()["content"] += f"\n[error] {type(error).__name__}: {error}"
        return self.messages
