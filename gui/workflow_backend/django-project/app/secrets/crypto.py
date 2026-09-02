"""AES-256-GCM envelope encryption. No Django imports — unit-testable alone.

Each payload is encrypted with a random 32-byte DEK. The DEK is wrapped with
a versioned KEK (HKDF-SHA256 of the operator master key). AAD binds the
ciphertext to an application identity so rows cannot be swapped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

CURRENT_KEY_VERSION = 1
_NONCE_LEN = 12
_DEK_LEN = 32
_HKDF_SALT_PREFIX = b"neuroworkflow-secrets-kek-v"
_HKDF_INFO = b"user-secret-store"


class CryptoError(Exception):
    """Envelope encrypt/decrypt failed."""


@dataclass(frozen=True)
class EncryptedBlob:
    wrapped_dek: bytes
    ciphertext: bytes
    nonce: bytes
    key_version: int


def derive_kek(master_key: bytes, version: int) -> bytes:
    if not master_key:
        raise CryptoError("master key is empty")
    if version < 0:
        raise CryptoError("invalid key version")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_HKDF_SALT_PREFIX + str(version).encode("ascii"),
        info=_HKDF_INFO,
    ).derive(master_key)


def envelope_encrypt(plaintext: bytes, kek: bytes, aad: bytes, *, key_version: int = CURRENT_KEY_VERSION) -> EncryptedBlob:
    if not isinstance(plaintext, (bytes, bytearray)):
        raise CryptoError("plaintext must be bytes")
    dek = os.urandom(_DEK_LEN)
    wrap_nonce = os.urandom(_NONCE_LEN)
    wrapped_body = AESGCM(kek).encrypt(wrap_nonce, dek, aad)
    data_nonce = os.urandom(_NONCE_LEN)
    ciphertext = AESGCM(dek).encrypt(data_nonce, bytes(plaintext), aad)
    return EncryptedBlob(
        wrapped_dek=wrap_nonce + wrapped_body,
        ciphertext=bytes(ciphertext),
        nonce=data_nonce,
        key_version=key_version,
    )


def envelope_decrypt(blob: EncryptedBlob, kek: bytes, aad: bytes) -> bytes:
    if len(blob.wrapped_dek) < _NONCE_LEN + 16:
        raise CryptoError("wrapped DEK is truncated")
    if len(blob.nonce) != _NONCE_LEN:
        raise CryptoError("data nonce must be 12 bytes")
    wrap_nonce = blob.wrapped_dek[:_NONCE_LEN]
    wrapped_body = blob.wrapped_dek[_NONCE_LEN:]
    try:
        dek = AESGCM(kek).decrypt(wrap_nonce, wrapped_body, aad)
        return AESGCM(dek).decrypt(blob.nonce, blob.ciphertext, aad)
    except Exception as exc:
        raise CryptoError("decrypt failed") from exc


def aad_for_user_secret(owner_id: int | str, secret_id: str) -> bytes:
    return f"user-secret:{owner_id}:{secret_id}".encode("utf-8")


def aad_for_custom_db(owner_id: int | str | None, database_id: str) -> bytes:
    owner = owner_id if owner_id is not None else "0"
    return f"custom-db:{owner}:{database_id}:api_key".encode("utf-8")
