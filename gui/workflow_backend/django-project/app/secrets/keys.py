"""Load versioned KEKs from the operator master key."""

from __future__ import annotations

import os
import sys

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .crypto import CURRENT_KEY_VERSION, derive_kek


def _running_tests() -> bool:
    return "pytest" in sys.modules


def _master_bytes(raw: str) -> bytes:
    return raw.encode("utf-8")


def require_production_master_key() -> None:
    """Refuse to boot production without an explicit SECRETS_MASTER_KEY."""
    if settings.DEBUG or _running_tests():
        return
    if not os.getenv("SECRETS_MASTER_KEY", "").strip():
        raise ImproperlyConfigured(
            "SECRETS_MASTER_KEY is required when DJANGO_DEBUG is false. "
            "Generate one with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )


def _current_master() -> tuple[bytes, int]:
    explicit = os.getenv("SECRETS_MASTER_KEY", "").strip()
    if explicit:
        return _master_bytes(explicit), CURRENT_KEY_VERSION
    if settings.DEBUG or _running_tests():
        secret = settings.SECRET_KEY
        if not secret:
            raise ImproperlyConfigured(
                "DJANGO_SECRET_KEY is required to derive a development secrets KEK."
            )
        return _master_bytes(str(secret)), CURRENT_KEY_VERSION
    raise ImproperlyConfigured("SECRETS_MASTER_KEY is not set")


def get_kek(version: int | None = None) -> bytes:
    master, current_version = _current_master()
    use_version = current_version if version is None else version
    if use_version == current_version:
        return derive_kek(master, use_version)
    previous = os.getenv("SECRETS_MASTER_KEY_PREVIOUS", "").strip()
    if previous and use_version == current_version - 1:
        return derive_kek(_master_bytes(previous), use_version)
    # Dev derivation is version CURRENT_KEY_VERSION only.
    if (settings.DEBUG or _running_tests()) and use_version == CURRENT_KEY_VERSION:
        return derive_kek(master, use_version)
    raise ImproperlyConfigured(f"No KEK material for key_version={use_version}")


def decrypt_kek_for_version(version: int) -> bytes:
    """KEK used to unwrap a stored blob of this version."""
    master, current_version = _current_master()
    if version == current_version:
        return derive_kek(master, version)
    previous = os.getenv("SECRETS_MASTER_KEY_PREVIOUS", "").strip()
    if previous:
        return derive_kek(_master_bytes(previous), version)
    if version == current_version:
        return derive_kek(master, version)
    # Same master can still unwrap old versions if only the version integer changed.
    return derive_kek(master, version)
