"""UserSecret persist/decrypt round trip."""

import pytest
from django.core.exceptions import ValidationError

from app.secrets.crypto import aad_for_user_secret, envelope_decrypt, EncryptedBlob
from app.secrets.keys import get_kek
from app.secrets.models import UserSecret

pytestmark = pytest.mark.django_db


def test_user_secret_round_trip(user_alice):
    secret = UserSecret(owner=user_alice, name="ASPERA_PASSWORD", description="aspera")
    secret.set_plaintext("not-a-real-password")
    secret.save()
    loaded = UserSecret.objects.get(pk=secret.pk)
    assert loaded.decrypt_plaintext() == "not-a-real-password"
    assert "not-a-real-password" not in str(bytes(loaded.ciphertext))


def test_user_secret_aad_binds_owner(user_alice, user_bob):
    secret = UserSecret(owner=user_alice, name="OPENAI_API_KEY")
    secret.set_plaintext("sk-test")
    secret.save()
    swapped_aad = aad_for_user_secret(user_bob.id, str(secret.id))
    blob = EncryptedBlob(
        wrapped_dek=bytes(secret.wrapped_dek),
        ciphertext=bytes(secret.ciphertext),
        nonce=bytes(secret.nonce),
        key_version=secret.key_version,
    )
    with pytest.raises(Exception):
        envelope_decrypt(blob, get_kek(), swapped_aad)


def test_empty_secret_rejected(user_alice):
    secret = UserSecret(owner=user_alice, name="EMPTY_SECRET")
    with pytest.raises(ValidationError):
        secret.set_plaintext("")


def test_custom_database_api_key_encrypted(user_alice):
    from app.metadata.models import CustomDatabase

    db = CustomDatabase.objects.create(
        name="test-db",
        base_url="https://example.com",
        created_by=user_alice,
    )
    db.set_api_key("db-secret-key")
    db.save()
    loaded = CustomDatabase.objects.get(pk=db.pk)
    assert loaded.get_api_key() == "db-secret-key"
    field_names = {f.name for f in loaded._meta.local_fields}
    assert "api_key" not in field_names
    assert "api_key_ciphertext" in field_names
