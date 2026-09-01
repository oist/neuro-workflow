"""Tests for the kernel-plane harvested records endpoint (service token).

``GET /api/harvest/records/?identifiers=...`` serves the local record store to
the workflow kernels (OAIPMHRecordsNode and the download node's fallback).
"""

import pytest
from app.harvest.models import HarvestedRecord
from app.harvest.views import MAX_RECORDS_PER_REQUEST
from django.urls import reverse

pytestmark = pytest.mark.django_db


def _url():
    return reverse("harvest-records")


@pytest.fixture
def service_env(monkeypatch):
    monkeypatch.setenv("JUPYTERHUB_API_TOKEN", "svc-token")


def _row(identifier, deleted=False):
    return HarvestedRecord.objects.create(
        oai_identifier=identifier,
        datestamp="2026-01-02T00:00:00Z",
        set_specs=["public", "dataset"],
        deleted=deleted,
        metadata=({} if deleted else {"name": "Alpha", "path": "/Alpha", "size": 1}),
        files=[] if deleted else [{"id": "f-1", "name": "a.txt"}],
        search_text="",
    )


def test_requires_service_token(client, service_env):
    _row("oai:repo:1")
    assert client.get(_url(), {"identifiers": "oai:repo:1"}).status_code == 401
    assert (
        client.get(
            _url(), {"identifiers": "oai:repo:1"}, HTTP_X_API_KEY="nope"
        ).status_code
        == 401
    )


def test_returns_records_in_request_order_with_missing(client, service_env):
    _row("oai:repo:b")
    _row("oai:repo:a")
    resp = client.get(
        _url(),
        {"identifiers": "oai:repo:a, oai:repo:b ,oai:repo:missing"},
        HTTP_X_API_KEY="svc-token",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert [r["identifier"] for r in data["records"]] == [
        "oai:repo:a",
        "oai:repo:b",
    ]
    assert data["count"] == 2
    assert data["missing"] == ["oai:repo:missing"]
    record = data["records"][0]
    assert record["metadata"]["name"] == "Alpha"
    assert record["files"] == [{"id": "f-1", "name": "a.txt"}]
    assert record["metadata_prefix"] == "mdrs"
    assert record["set_specs"] == ["public", "dataset"]
    assert record["datestamp"] == "2026-01-02T00:00:00Z"
    assert record["deleted"] is False


def test_deleted_records_are_returned_flagged(client, service_env):
    _row("oai:repo:gone", deleted=True)
    data = client.get(
        _url(), {"identifiers": "oai:repo:gone"}, HTTP_X_API_KEY="svc-token"
    ).json()
    assert data["records"][0]["deleted"] is True
    assert data["records"][0]["metadata"] is None
    assert data["missing"] == []


def test_rejects_empty_identifiers(client, service_env):
    resp = client.get(_url(), {"identifiers": " , "}, HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 400
    assert client.get(_url(), HTTP_X_API_KEY="svc-token").status_code == 400


def test_rejects_more_identifiers_than_the_cap(client, service_env):
    identifiers = ",".join(f"oai:repo:{i}" for i in range(MAX_RECORDS_PER_REQUEST + 1))
    resp = client.get(_url(), {"identifiers": identifiers}, HTTP_X_API_KEY="svc-token")
    assert resp.status_code == 400
