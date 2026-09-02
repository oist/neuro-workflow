"""Load versioned KEKs from the operator master key."""

from __future__ import annotations

import os

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .crypto import CURRENT_KEY_VERSION, CryptoError, EncryptedBlob, derive_kek, envelope_decrypt


def _running_tests() -> bool:
    return bool(os.environ.get("NW_TESTING") or os.environ.get("PYTEST_CURRENT_TEST"))


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
    """KEK for wrapping new ciphertext. Always the current master."""
    master, current_version = _current_master()
    use_version = current_version if version is None else version
    if use_version == current_version:
        return derive_kek(master, use_version)
    raise ImproperlyConfigured(
        f"Encrypt uses the current KEK only (key_version={current_version}), "
        f"not key_version={use_version}."
    )


def _keks_for_decrypt(stored_version: int) -> list[bytes]:
    """Current master first, then SECRETS_MASTER_KEY_PREVIOUS. Same AAD/version salt."""
    master, current_version = _current_master()
    previous = os.getenv("SECRETS_MASTER_KEY_PREVIOUS", "").strip()
    keks: list[bytes] = []
    seen: set[bytes] = set()

    def add(kek: bytes) -> None:
        if kek not in seen:
            seen.add(kek)
            keks.append(kek)

    add(derive_kek(master, stored_version))
    if stored_version != current_version:
        add(derive_kek(master, current_version))
    if previous:
        prev = _master_bytes(previous)
        add(derive_kek(prev, stored_version))
        if stored_version != current_version:
            add(derive_kek(prev, current_version))
    return keks


def decrypt_kek_for_version(version: int) -> bytes:
    """First KEK to try for a stored blob (current master). Prefer decrypt_blob()."""
    return _keks_for_decrypt(version)[0]


def decrypt_blob(blob: EncryptedBlob, aad: bytes) -> bytes:
    """Unwrap with current KEK, then PREVIOUS. AAD tamper still fails all candidates."""
    last_error: CryptoError | None = None
    for kek in _keks_for_decrypt(blob.key_version):
        try:
            return envelope_decrypt(blob, kek, aad)
        except CryptoError as exc:
            last_error = exc
    raise last_error or CryptoError("decrypt failed")
