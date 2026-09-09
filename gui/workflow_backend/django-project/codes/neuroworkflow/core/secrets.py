"""Runtime secret references. Django-free — used by generated Jupyter/Slurm code.

Generated workflows call ``load_runtime_secrets()`` (file or ``NW_SECRET_*`` env)
then ``configure(password=SecretRef("ASPERA_PASSWORD"))``. Node code should
``resolve()`` secret parameters before using them as credentials.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

SECRET_REF_KEY = "__nw_secret"
_SECRETS_FILE_ENV = "NW_SECRETS_FILE"
_SECRETS_ENV_PREFIX = "NW_SECRET_"

_installed: dict[str, str] = {}


class MissingSecretError(ValueError):
    """A SecretRef could not be resolved from the runtime mapping."""


@dataclass(frozen=True)
class SecretRef:
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {SECRET_REF_KEY: {"id": "", "name": self.name}}


class SecretStr:
    """Holds a secret in memory; repr/str never echo the value."""

    def __init__(self, value: str):
        self._value = value if value is not None else ""

    def get_secret(self) -> str:
        return self._value

    def __str__(self) -> str:
        return "••••" if self._value else ""

    def __repr__(self) -> str:
        return "SecretStr(••••)" if self._value else "SecretStr('')"

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return self._value == other._value
        return False


def install_runtime_secrets(mapping: Mapping[str, str] | None) -> None:
    global _installed
    _installed = {str(k): str(v) for k, v in (mapping or {}).items() if k}


def clear_runtime_secrets() -> None:
    global _installed
    _installed = {}


def load_runtime_secrets() -> dict[str, str]:
    """Load from ``NW_SECRETS_FILE`` JSON and/or ``NW_SECRET_<NAME>`` env vars."""
    mapping: dict[str, str] = dict(_installed)
    path = os.environ.get(_SECRETS_FILE_ENV, "").strip()
    if path:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            mapping.update({str(k): str(v) for k, v in data.items()})
    for key, value in os.environ.items():
        if key.startswith(_SECRETS_ENV_PREFIX) and key != _SECRETS_FILE_ENV:
            name = key[len(_SECRETS_ENV_PREFIX) :]
            if name:
                mapping[name] = value
    install_runtime_secrets(mapping)
    return dict(_installed)


def is_secret_ref(value: Any) -> bool:
    if isinstance(value, SecretRef):
        return True
    return isinstance(value, dict) and SECRET_REF_KEY in value


def secret_ref_name(value: Any) -> str | None:
    if isinstance(value, SecretRef):
        return value.name
    if isinstance(value, dict) and SECRET_REF_KEY in value:
        inner = value.get(SECRET_REF_KEY) or {}
        if isinstance(inner, dict):
            return inner.get("name") or None
    return None


def resolve(value: Any, *, required_name: str | None = None) -> Any:
    """Return the plaintext for a SecretRef/dict ref, or passthrough."""
    if isinstance(value, SecretStr):
        return value.get_secret()
    if not is_secret_ref(value):
        return value
    name = secret_ref_name(value) or required_name
    if not name:
        raise MissingSecretError("Secret reference is missing a name")
    if name not in _installed:
        load_runtime_secrets()
    if name not in _installed:
        raise MissingSecretError(f"Secret '{name}' is not available in this run")
    return _installed[name]


def wrap_resolved(value: Any, *, secret: bool) -> Any:
    if not secret:
        return value
    if isinstance(value, SecretStr):
        return value
    plaintext = resolve(value) if is_secret_ref(value) else (value if isinstance(value, str) else "")
    if is_secret_ref(value) or (isinstance(value, str) and value):
        if is_secret_ref(value):
            plaintext = resolve(value)
        return SecretStr(str(plaintext or ""))
    return SecretStr("")
