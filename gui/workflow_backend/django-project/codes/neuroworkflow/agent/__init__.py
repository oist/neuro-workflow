"""In-notebook chat agent for NeuroWorkflow (Issue #52).

Usage inside a Jupyter notebook::

    %load_ext neuroworkflow.agent
    %chat how do I build a SONATA network?

or a persistent panel::

    from neuroworkflow.agent import ChatPanel
    ChatPanel()   # workflow tools via the app's token relay (see docs)

``ChatPanel`` shows a model picker and ``%chat --model <id> ...`` switches the
model when the backend offers other models next to Claude.

Tool calls follow the session's permissions (allow / ask / deny per category,
within the administrator's limits; ``%chat_permissions`` or
``get_agent().permissions``). Calls set to ask wait for the user's approval
in the panel or the cell.

The agent loop runs in the kernel (Claude Agent SDK); the Anthropic key and MCP
tools stay on the backend, reached over HTTP.
"""

from __future__ import annotations

from .client import BackendClient
from .config import get_config
from .loop import Agent

_agent: Agent | None = None


def list_models() -> list[str]:
    """Model ids to pick from: ``""`` (the default Claude model), then the others."""
    return ["", *BackendClient(get_config()).list_provider_models()]


def get_agent(*, user_token=None, project_id=None, model=None) -> Agent:
    """Return the shared agent, creating it on first use.

    Passing ``user_token``/``project_id``/``model`` (or changing them) rebuilds
    the agent; whatever is not passed is kept from the current one. ``model`` is
    an id from ``list_models()``; switching it starts a new conversation,
    because a session cannot move between providers. The user's permission
    settings are kept.
    """
    global _agent
    needs_new = (
        _agent is None
        or (user_token is not None and _agent._config.user_token != user_token)
        or (project_id is not None and _agent._config.project_id != project_id)
        or (model is not None and (_agent._config.provider_model or "") != model)
    )
    if needs_new:
        if _agent is not None:
            # e.g. switching the model must not drop a manually supplied token.
            current = _agent._config
            if user_token is None:
                user_token = current.user_token
            if project_id is None:
                project_id = current.project_id
            if model is None:
                model = current.provider_model
        config = get_config(
            user_token=user_token, project_id=project_id, model=model
        )
        permissions = None
        if _agent is not None:
            permissions = _agent.permissions
            permissions.new_conversation()
        _agent = Agent(config, permissions=permissions)
    return _agent


def reset_agent():
    """Drop the current agent (clears history and permission settings, and
    re-reads config on next use)."""
    global _agent
    _agent = None


def chat(message: str):
    """Send one message and stream the reply to stdout (approvals asked here)."""
    from .magic import _stream_text, console_approve

    get_agent().run(message, on_text=_stream_text, approve=console_approve)
    print()


def ChatPanel(**kwargs):
    from .widget import ChatPanel as _ChatPanel

    return _ChatPanel(**kwargs)


def load_ipython_extension(ipython):
    from .magic import register

    register(ipython)
