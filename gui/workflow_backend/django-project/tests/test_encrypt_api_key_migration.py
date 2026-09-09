"""0002 encrypt helper encrypts then drops plaintext api_key."""

import importlib.util
from pathlib import Path

from app.secrets.crypto import EncryptedBlob, aad_for_custom_db, envelope_decrypt
from app.secrets.keys import get_kek

PLAIN = "plain-legacy-key"
_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "metadata"
    / "migrations"
    / "0002_encrypt_api_key.py"
)


def _load_encrypt_helper():
    spec = importlib.util.spec_from_file_location("nw_encrypt_api_key_migration", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.encrypt_existing_api_keys


class _Row:
    def __init__(self):
        self.id = "11111111-1111-1111-1111-111111111111"
        self.created_by_id = 1
        self.api_key = PLAIN
        self.api_key_wrapped_dek = None
        self.api_key_ciphertext = None
        self.api_key_nonce = None
        self.api_key_key_version = 1
        self.saved_fields = None

    def save(self, update_fields=None):
        self.saved_fields = update_fields


def test_encrypt_existing_api_keys_drops_plaintext():
    encrypt_existing_api_keys = _load_encrypt_helper()
    row = _Row()

    class _Manager:
        def all(self):
            return [row]

    class _Model:
        objects = _Manager()

    class _Apps:
        def get_model(self, app, name):
            assert app == "metadata"
            return _Model

    encrypt_existing_api_keys(_Apps(), None)
    assert row.api_key is None
    assert row.api_key_ciphertext
    blob = EncryptedBlob(
        wrapped_dek=bytes(row.api_key_wrapped_dek),
        ciphertext=bytes(row.api_key_ciphertext),
        nonce=bytes(row.api_key_nonce),
        key_version=row.api_key_key_version,
    )
    assert envelope_decrypt(blob, get_kek(), aad_for_custom_db(1, str(row.id))) == PLAIN.encode()
    assert "api_key" in (row.saved_fields or [])
