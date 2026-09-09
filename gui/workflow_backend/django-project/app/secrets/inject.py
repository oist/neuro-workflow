"""Collect named secret refs from flow JSON and fail closed before a run."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .redaction import is_secret_ref, secret_ref_name
from .services import materialize_named_secrets


class MissingRuntimeSecrets(Exception):
    def __init__(self, names: list[str]):
        self.names = list(names)
        super().__init__(
            "Missing secrets: "
            + ", ".join(self.names)
            + ". Bind your own secret in Settings → Secrets."
        )


def _walk_value(value: Any, names: set[str]) -> None:
    if is_secret_ref(value):
        name = secret_ref_name(value)
        if name:
            names.add(name)
        return
    if isinstance(value, dict):
        for item in value.values():
            _walk_value(item, names)
    elif isinstance(value, list):
        for item in value:
            _walk_value(item, names)


def collect_secret_names_from_data(data: Any) -> list[str]:
    names: set[str] = set()
    _walk_value(data, names)
    return sorted(names)


def collect_secret_names_from_nodes(nodes: Iterable[Any]) -> list[str]:
    names: set[str] = set()
    for node in nodes:
        data = getattr(node, "data", node)
        names.update(collect_secret_names_from_data(data))
    return sorted(names)


def collect_secret_names_for_project(project) -> list[str]:
    return collect_secret_names_from_nodes(project.nodes.all())


def require_owner_runtime_secrets(owner, names: list[str], *, actor=None, ip=None) -> dict[str, str]:
    wanted = [n for n in names if n]
    if not wanted:
        return {}
    mapping = materialize_named_secrets(owner, wanted, actor=actor, ip=ip)
    missing = [n for n in wanted if n not in mapping]
    if missing:
        raise MissingRuntimeSecrets(missing)
    return mapping


def wrap_jupyter_code(user_code: str, mapping: dict[str, str] | None) -> str:
    """Prefix ephemeral kernel code with in-process secret install. Do not log the result."""
    if not mapping:
        return user_code
    payload = json.dumps(mapping, ensure_ascii=False)
    return (
        "from neuroworkflow.core.secrets import install_runtime_secrets\n"
        f"install_runtime_secrets({payload})\n"
        + user_code
    )


def redact_with_values(text: Any, values: Iterable[str] | None) -> Any:
    if not text or not values:
        return text
    if not isinstance(text, str):
        return text
    out = text
    for value in sorted((v for v in values if v), key=len, reverse=True):
        out = out.replace(value, "••••")
    return out


def redact_event(event: dict, values: Iterable[str] | None) -> dict:
    if not event or not values:
        return event
    data = event.get("data")
    if not isinstance(data, dict):
        return event
    if "content" in data:
        data["content"] = redact_with_values(data["content"], values)
    if "evalue" in data:
        data["evalue"] = redact_with_values(data["evalue"], values)
    tb = data.get("traceback")
    if isinstance(tb, list):
        data["traceback"] = [redact_with_values(item, values) for item in tb]
    return event
