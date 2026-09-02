"""CustomDatabase vault ref is resolved server-side and never appears as plaintext on GET."""

import pytest
from django.urls import reverse

from app.metadata.models import CustomDatabase
from app.secrets.redaction import make_secret_ref
from app.secrets.services import create_user_secret

pytestmark = pytest.mark.django_db

FIXTURE_KEY = "custom-db-fixture-key"


def test_customdb_vault_ref_not_in_get(auth_client, user_alice):
    secret = create_user_secret(user_alice, name="DB_API_KEY", value=FIXTURE_KEY)
    db = CustomDatabase.objects.create(
        name="demo",
        base_url="https://example.invalid",
        created_by=user_alice,
        config={"api_key_secret": make_secret_ref(secret.id, secret.name)},
    )
    client = auth_client(user_alice)
    resp = client.get(f"/api/metadata/custom-databases/{db.id}/")
    assert resp.status_code == 200
    body = resp.json()
    dumped = str(body)
    assert FIXTURE_KEY not in dumped
    assert body.get("api_key") in (None, "", "••••") or "api_key" not in body or body["api_key"] is None
    assert body["config"]["api_key_secret"]["__nw_secret"]["name"] == "DB_API_KEY"
    assert db.resolve_api_key() == FIXTURE_KEY


def test_to_adapter_config_does_not_keep_raw_config_key(user_alice):
    db = CustomDatabase.objects.create(
        name="demo2",
        base_url="https://example.invalid",
        created_by=user_alice,
        config={"api_key": "raw-config-key", "timeout": 5},
    )
    db.set_api_key(FIXTURE_KEY)
    db.save()
    cfg = db.to_adapter_config()
    assert cfg["api_key"] == FIXTURE_KEY
    assert "raw-config-key" not in str(cfg)


def test_suggestions_queryset_excludes_other_owners(user_alice, user_bob):
    from app.metadata.views import _custom_databases_for_suggestions

    CustomDatabase.objects.create(
        name="bob-db",
        base_url="https://bob.example.invalid",
        created_by=user_bob,
        is_active=True,
        is_verified=True,
    )
    alice_db = CustomDatabase.objects.create(
        name="alice-db",
        base_url="https://alice.example.invalid",
        created_by=user_alice,
        is_active=True,
        is_verified=True,
    )
    qs = _custom_databases_for_suggestions(user_alice)
    assert list(qs) == [alice_db]


def test_get_drops_config_api_key(auth_client, user_alice):
    db = CustomDatabase.objects.create(
        name="leaky",
        base_url="https://example.invalid",
        created_by=user_alice,
        config={"api_key": FIXTURE_KEY, "timeout": 1},
    )
    db.set_api_key(FIXTURE_KEY)
    db.save()
    client = auth_client(user_alice)
    resp = client.get(f"/api/metadata/custom-databases/{db.id}/")
    assert resp.status_code == 200
    dumped = str(resp.json())
    assert FIXTURE_KEY not in dumped
    assert "api_key" not in (resp.json().get("config") or {})
