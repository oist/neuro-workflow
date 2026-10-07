"""``%chat`` / ``%%chat`` magics for talking to the agent inline."""

from __future__ import annotations

import sys


def _stream_text(delta: str):
    sys.stdout.write(delta)
    sys.stdout.flush()


def _announce_tool(name: str, args: dict):
    preview = ", ".join(f"{k}={v!r}"[:60] for k, v in args.items())
    print(f"\n\033[2m[tool] {name}({preview})\033[0m")


def _split_model(line: str) -> tuple[str | None, str]:
    """Split a leading ``--model <id>`` off the line."""
    parts = line.split(None, 2)
    if len(parts) >= 2 and parts[0] == "--model":
        return parts[1], parts[2] if len(parts) > 2 else ""
    return None, line


def chat_magic(line: str, cell: str | None = None):
    """Send a message to the notebook agent.

    Line form:  ``%chat how do I build a SONATA network?``
    Cell form:  ``%%chat`` followed by a multi-line prompt.

    ``--model <id>`` first (``%chat --model MiniMax-M3 ...``) switches the model
    for this and later messages and starts a new conversation; ``--model
    default`` returns to Claude.
    """
    from . import get_agent, list_models

    model, line = _split_model(line)
    message = (cell if cell is not None else line).strip()
    if not message:
        print("Usage: %chat [--model <id>] <message>  or  %%chat (cell)")
        return
    if model == "default":
        model = ""
    elif model is not None:
        models = list_models()
        if model not in models:
            choices = ", ".join(["default", *models[1:]])
            print(f"Unknown model {model!r}. Available: {choices}")
            return
    agent = get_agent(model=model)
    agent.run(message, on_text=_stream_text, on_tool=_announce_tool)
    print()


def register(ipython):
    ipython.register_magic_function(chat_magic, magic_kind="line_cell", magic_name="chat")
