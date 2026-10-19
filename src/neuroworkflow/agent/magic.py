"""``%chat`` / ``%%chat`` magics for talking to the agent inline."""

from __future__ import annotations

import sys

from .permissions import CATEGORIES, MODES, PermissionRequest, parse_answer


def _stream_text(delta: str):
    sys.stdout.write(delta)
    sys.stdout.flush()


def _announce_tool(name: str, args: dict):
    preview = ", ".join(f"{k}={v!r}"[:60] for k, v in args.items())
    print(f"\n\033[2m[tool] {name}({preview})\033[0m")


def console_approve(request: PermissionRequest):
    """Ask in the cell (``input()``) whether the agent may run a tool call."""
    print(f"\n[approval] {request.tool} ({request.category})")
    print(request.summary)
    choices = "[y]es / [n]o"
    if request.can_always_allow:
        choices += " / [a]lways this conversation"
    answer = input(f"Allow? {choices} / or type a reason to deny: ")
    return parse_answer(answer, request.can_always_allow)


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
    default`` returns to Claude. Tool calls that need approval are asked here.
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
    agent.run(
        message,
        on_text=_stream_text,
        on_tool=_announce_tool,
        approve=console_approve,
    )
    print()


def chat_permissions_magic(line: str):
    """Show or change the agent's tool permissions for this session.

    ``%chat_permissions`` shows them; ``%chat_permissions code-exec allow`` sets
    a category (or a single tool, e.g. ``Bash``) to allow / ask / deny within
    the administrator's limits; ``%chat_permissions reset`` restores the
    defaults.
    """
    from . import get_agent

    permissions = get_agent().permissions
    parts = line.split()
    if parts == ["reset"]:
        permissions.reset()
    elif len(parts) == 2:
        try:
            permissions.set(parts[0], parts[1])
        except ValueError as e:
            print(e)
            return
    elif parts:
        print(
            "Usage: %chat_permissions [<category or tool> <mode> | reset]\n"
            f"  categories: {', '.join(CATEGORIES)}\n  modes: {', '.join(MODES)}"
        )
        return
    print(permissions.table())


def register(ipython):
    ipython.register_magic_function(chat_magic, magic_kind="line_cell", magic_name="chat")
    ipython.register_magic_function(
        chat_permissions_magic, magic_kind="line", magic_name="chat_permissions"
    )
