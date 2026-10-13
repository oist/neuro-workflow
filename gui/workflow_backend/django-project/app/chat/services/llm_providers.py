"""Catalogue of the LLM models a user may pick for the chats.

Besides OpenAI, any OpenAI-compatible provider can be offered. Providers are
listed in a JSON file (``LLM_PROVIDERS_FILE``, default
``workflow_backend/llm_providers.json``)::

    {"providers": [{
        "name": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat"],
        "anthropic_base_url": "https://api.deepseek.com/anthropic",
        "extra_body": {}
    }]}

The file names the environment variable holding the key, never the key itself.
A provider is offered only when that key is set; ``anthropic_base_url``
(optional) also offers its models to the notebook agent; ``extra_body``
(optional) is merged into each Chat Completions request. MiniMax can still be
configured with the ``MINIMAX_*`` variables alone.

The file and the environment are read on every call (not at import) so the
catalogue follows the configuration without a module reload.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# workflow_backend/, which the backend container mounts at /django-app.
_DEFAULT_PROVIDERS_FILE = Path(__file__).resolve().parents[4] / "llm_providers.json"


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    models: list[str]
    anthropic_base_url: str | None = None
    extra_body: dict = field(default_factory=dict)


def _strip(url: str | None) -> str | None:
    return url.rstrip("/") if url else None


def _check_entry(entry: dict):
    """Raise ValueError unless ``entry`` follows the documented schema."""
    for key in ("name", "base_url", "api_key_env"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ValueError(f"{key} must be a non-empty string: {entry[key]!r}")
    models = entry["models"]
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise ValueError(f"models must be a list of strings: {models!r}")
    if not isinstance(entry.get("anthropic_base_url") or "", str):
        raise ValueError(
            f"anthropic_base_url must be a string: {entry['anthropic_base_url']!r}"
        )
    if not isinstance(entry.get("extra_body") or {}, dict):
        raise ValueError(f"extra_body must be an object: {entry['extra_body']!r}")


def _file_providers() -> list[Provider]:
    path = Path(os.environ.get("LLM_PROVIDERS_FILE") or _DEFAULT_PROVIDERS_FILE)
    if not path.is_file():
        return []
    try:
        entries = json.loads(path.read_text())["providers"]
        providers = []
        for entry in entries:
            _check_entry(entry)
            providers.append(
                Provider(
                    name=entry["name"],
                    base_url=_strip(entry["base_url"]),
                    api_key=os.environ.get(entry["api_key_env"], ""),
                    models=[m.strip() for m in entry["models"] if m.strip()],
                    anthropic_base_url=_strip(entry.get("anthropic_base_url")),
                    extra_body=entry.get("extra_body") or {},
                )
            )
        return providers
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        logger.warning("Ignoring %s: %s", path, e)
        return []


def _minimax_provider() -> Provider:
    """MiniMax from the ``MINIMAX_*`` variables (configured before the file)."""
    raw = os.environ.get("MINIMAX_MODELS") or "MiniMax-M3"
    return Provider(
        name="minimax",
        base_url=_strip(
            os.environ.get("MINIMAX_OPENAI_BASE_URL") or "https://api.minimax.io/v1"
        ),
        api_key=os.environ.get("MINIMAX_API_KEY", ""),
        models=[m.strip() for m in raw.split(",") if m.strip()],
        anthropic_base_url=_strip(
            os.environ.get("MINIMAX_ANTHROPIC_BASE_URL")
            or "https://api.minimax.io/anthropic"
        ),
        # Keep the model's thinking out of the answer text.
        extra_body={"reasoning_split": True},
    )


def providers() -> list[Provider]:
    """OpenAI-compatible providers on offer, each with its own model ids.

    Providers without a key are left out. A model id is offered once: when the
    OpenAI model or an earlier provider has it, a later provider loses it.
    """
    candidates = _file_providers()
    if not any(p.name == "minimax" for p in candidates):
        candidates.append(_minimax_provider())

    seen = set()
    if os.environ.get("OPENAI_API_KEY"):
        seen.add(os.environ.get("OPENAI_MODEL", "gpt-4o"))
    offered = []
    for provider in candidates:
        if not provider.api_key:
            continue
        models = []
        for model in provider.models:
            if model in seen:
                logger.warning(
                    "Model %r of provider %r is already offered; skipped",
                    model,
                    provider.name,
                )
                continue
            seen.add(model)
            models.append(model)
        provider.models = models
        offered.append(provider)
    return offered


def provider_for_model(model: str | None) -> Provider | None:
    """The OpenAI-compatible provider offering ``model``; None for OpenAI."""
    for provider in providers():
        if model in provider.models:
            return provider
    return None


def chat_models() -> list[dict]:
    """Models selectable in the chats; the first one is the default.

    ``notebook`` marks the models the notebook agent can use too (their
    provider has an Anthropic-compatible API).
    """
    models = []
    if os.environ.get("OPENAI_API_KEY"):
        models.append(
            {
                "id": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "provider": "openai",
                "notebook": False,
            }
        )
    for provider in providers():
        models += [
            {
                "id": m,
                "provider": provider.name,
                "notebook": bool(provider.anthropic_base_url),
            }
            for m in provider.models
        ]
    return models
