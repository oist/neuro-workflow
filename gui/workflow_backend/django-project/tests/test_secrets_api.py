"""Owner-only /api/secrets/ — values never appear in responses."""

import pytest
from django.urls import reverse

from app.secrets.models import SecretAuditEvent, UserSecret
from app.secrets.services import create_user_secret

pytestmark = pytest.mark.django_db


def test_create_list_never_returns_value(auth_client, user_alice):
    client = auth_client(user_alice)
    url = reverse("secrets:secret-list-create")
    resp = client.post(
        url,
        {"name": "ASPERA_PASSWORD", "value": "plain-secret-value", "description": "aspera"},
        format="json",
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "ASPERA_PASSWORD"
    assert body["is_set"] is True
    assert "value" not in body or body.get("value") in (None, "", "••••")
    dumped = str(body)
    assert "plain-secret-value" not in dumped

    listed = client.get(url)
    assert listed.status_code == 200
    assert "plain-secret-value" not in listed.content.decode()
    assert listed.json()[0]["name"] == "ASPERA_PASSWORD"

    audit = SecretAuditEvent.objects.get(secret_name="ASPERA_PASSWORD", action="create")
    assert "plain-secret-value" not in str(audit.__dict__)


def test_owner_isolation(auth_client, user_alice, user_bob):
    secret = create_user_secret(user_alice, name="OPENAI_API_KEY", value="sk-alice")
    bob = auth_client(user_bob)
    detail = reverse("secrets:secret-detail", kwargs={"secret_id": secret.id})
    assert bob.get(detail).status_code == 404
    assert bob.patch(detail, {"value": "sk-bob"}, format="json").status_code == 404
    assert bob.delete(detail).status_code == 404
    listed = bob.get(reverse("secrets:secret-list-create"))
    assert listed.json() == []


def test_alice_can_rotate_and_revoke(auth_client, user_alice):
    client = auth_client(user_alice)
    created = client.post(
        reverse("secrets:secret-list-create"),
        {"name": "DB_PASSWORD", "value": "first"},
        format="json",
    ).json()
    detail = reverse("secrets:secret-detail", kwargs={"secret_id": created["id"]})
    rotated = client.patch(detail, {"value": "second"}, format="json")
    assert rotated.status_code == 200
    assert "second" not in rotated.content.decode()
    deleted = client.delete(detail)
    assert deleted.status_code == 204
    assert UserSecret.objects.get(id=created["id"]).revoked_at is not None
    assert client.get(detail).status_code == 404


def test_recreate_same_name_after_revoke(auth_client, user_alice):
    client = auth_client(user_alice)
    url = reverse("secrets:secret-list-create")
    first = client.post(url, {"name": "ASPERA_PASSWORD", "value": "first"}, format="json")
    assert first.status_code == 201
    detail = reverse("secrets:secret-detail", kwargs={"secret_id": first.json()["id"]})
    assert client.delete(detail).status_code == 204
    second = client.post(url, {"name": "ASPERA_PASSWORD", "value": "second"}, format="json")
    assert second.status_code == 201
    assert second.json()["name"] == "ASPERA_PASSWORD"
    assert "second" not in second.content.decode()
    dup = client.post(url, {"name": "ASPERA_PASSWORD", "value": "third"}, format="json")
    assert dup.status_code == 400
