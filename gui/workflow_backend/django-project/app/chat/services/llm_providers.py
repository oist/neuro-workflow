"""Catalogue of the LLM models a user may pick for the chats.

Driven by the backend environment: a provider is offered only when its API key
is set. The environment is read on every call (not at import) so the catalogue
follows the configuration without a module reload.
"""

import os


def minimax_api_key() -> str:
    return os.environ.get("MINIMAX_API_KEY", "")


def minimax_openai_base_url() -> str:
    return (
        os.environ.get("MINIMAX_OPENAI_BASE_URL") or "https://api.minimax.io/v1"
    ).rstrip("/")


def minimax_models() -> list[str]:
    """MiniMax model ids on offer (``MINIMAX_MODELS``); empty without a key."""
    if not minimax_api_key():
        return []
    raw = os.environ.get("MINIMAX_MODELS") or "MiniMax-M3"
    return [m.strip() for m in raw.split(",") if m.strip()]


def chat_models() -> list[dict]:
    """Models selectable in the browser chat; the first one is the default."""
    models = []
    if os.environ.get("OPENAI_API_KEY"):
        models.append(
            {"id": os.environ.get("OPENAI_MODEL", "gpt-4o"), "provider": "openai"}
        )
    models += [{"id": m, "provider": "minimax"} for m in minimax_models()]
    return models
