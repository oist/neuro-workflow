"""Envelope encryption tests — no Django DB required for the crypto module."""

import pytest

from app.secrets.crypto import (
    CryptoError,
    EncryptedBlob,
    aad_for_user_secret,
    derive_kek,
    envelope_decrypt,
    envelope_encrypt,
)


def test_round_trip():
    kek = derive_kek(b"test-master-key", 1)
    aad = aad_for_user_secret(7, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    blob = envelope_encrypt(b"super-secret", kek, aad)
    assert envelope_decrypt(blob, kek, aad) == b"super-secret"
    assert blob.nonce != blob.wrapped_dek[:12]


def test_aad_tamper_fails():
    kek = derive_kek(b"test-master-key", 1)
    aad = aad_for_user_secret(7, "sid-1")
    blob = envelope_encrypt(b"super-secret", kek, aad)
    with pytest.raises(CryptoError):
        envelope_decrypt(blob, kek, aad_for_user_secret(8, "sid-1"))


def test_wrong_kek_fails():
    kek = derive_kek(b"test-master-key", 1)
    other = derive_kek(b"other-master-key", 1)
    aad = aad_for_user_secret(1, "sid")
    blob = envelope_encrypt(b"x", kek, aad)
    with pytest.raises(CryptoError):
        envelope_decrypt(blob, other, aad)


def test_decrypt_blob_uses_previous_master(monkeypatch):
    from app.secrets.crypto import envelope_encrypt
    from app.secrets.keys import decrypt_blob, get_kek

    monkeypatch.setenv("SECRETS_MASTER_KEY", "old-master-key-material")
    aad = aad_for_user_secret(7, "sid-rot")
    blob = envelope_encrypt(b"fixture-secret", get_kek(), aad)
    monkeypatch.setenv("SECRETS_MASTER_KEY", "new-master-key-material")
    monkeypatch.setenv("SECRETS_MASTER_KEY_PREVIOUS", "old-master-key-material")
    assert decrypt_blob(blob, aad) == b"fixture-secret"
    with pytest.raises(CryptoError):
        decrypt_blob(blob, aad_for_user_secret(8, "sid-rot"))


def test_truncated_wrapped_dek_fails():
    kek = derive_kek(b"test-master-key", 1)
    aad = b"aad"
    blob = envelope_encrypt(b"x", kek, aad)
    bad = EncryptedBlob(
        wrapped_dek=blob.wrapped_dek[:4],
        ciphertext=blob.ciphertext,
        nonce=blob.nonce,
        key_version=blob.key_version,
    )
    with pytest.raises(CryptoError):
        envelope_decrypt(bad, kek, aad)
