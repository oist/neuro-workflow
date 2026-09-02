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
