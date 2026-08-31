"""Tests for the NeuroWorkflow catalog proxy (mdb-mindsdb).

The proxy authenticates the Django user, then calls mdb with a dedicated
search token. User JWTs must never be forwarded; only keyword search is
allowed; mdb errors are mapped to generic catalog codes.
"""
import httpx
import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db

MDB_BASE = "http://mdb-mindsdb:8004"
MDB_TOKEN = "mdb-search-token"
USER_JWT = "user-jwt-must-not-leak"


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload

    def json(self):
        return self._payload


class FakeHttpxClient:
    """Records the outgoing mdb request and returns a canned response."""

    captured = None
    called = False
    response = None
    error = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @classmethod
    def reset(cls, response=None, error=None):
        cls.captured = None
        cls.called = False
        cls.response = response if response is not None else FakeResponse()
        cls.error = error

    def request(self, method, url, headers=None, params=None, json=None):
        type(self).called = True
        type(self).captured = {
            "method": method,
            "url": url,
            "headers": dict(headers or {}),
            "params": params,
            "json": json,
        }
        if type(self).error is not None:
            raise type(self).error
        return type(self).response

    def close(self):
        pass


@pytest.fixture
def mdb_configured(monkeypatch):
    monkeypatch.setattr(
        "app.catalog.client.get_mdb_config",
        lambda: (MDB_BASE, MDB_TOKEN, 15),
    )


@pytest.fixture
def fake_httpx(monkeypatch):
    FakeHttpxClient.reset()
    monkeypatch.setattr("app.catalog.client.httpx.Client", FakeHttpxClient)
    return FakeHttpxClient


def _error(resp):
    body = resp.json()
    assert body["status"] == "error"
    return body


def test_statistics_requires_auth(auth_client):
    resp = auth_client().get(reverse("catalog:catalog-statistics"))
    assert resp.status_code == 401


def test_statistics_unconfigured(auth_client, user_alice, monkeypatch, fake_httpx):
    monkeypatch.setattr(
        "app.catalog.client.get_mdb_config",
        lambda: ("", "", 15),
    )
    resp = auth_client(user_alice).get(reverse("catalog:catalog-statistics"))
    assert resp.status_code == 503
    assert _error(resp)["code"] == "catalog_unconfigured"
    assert fake_httpx.called is False


def test_search_rejects_agent_mode_before_httpx(
    auth_client, user_alice, mdb_configured, fake_httpx
):
    resp = auth_client(user_alice).post(
        reverse("catalog:catalog-search"),
        {"query": "mouse", "mode": "agent", "limit": 20},
        format="json",
    )
    assert resp.status_code == 400
    assert _error(resp)["code"] == "invalid_mode"
    assert fake_httpx.called is False


def test_search_sends_keyword_mode_clamped_limit_and_search_token(
    auth_client, user_alice, mdb_configured, fake_httpx
):
    fake_httpx.reset(response=FakeResponse(200, {"hits": []}))
    client = auth_client(user_alice)
    # Token is request.auth only — sending Bearer would hit Keycloak JWKS.
    client.force_authenticate(user=user_alice, token=USER_JWT)
    resp = client.post(
        reverse("catalog:catalog-search"),
        {"query": "mouse", "limit": 999},
        format="json",
    )
    assert resp.status_code == 200
    captured = fake_httpx.captured
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/catalog_search")
    assert captured["json"]["mode"] == "keyword"
    assert captured["json"]["limit"] == 200
    assert captured["json"]["query"] == "mouse"
    headers = captured["headers"]
    assert headers["Authorization"] == f"Bearer {MDB_TOKEN}"
    assert headers["Content-Type"] == "application/json"
    assert USER_JWT not in headers.values()
    assert all(USER_JWT not in str(value) for value in headers.values())


@pytest.mark.parametrize("table", ["local_catalog_datasets", "metadata_entries"])
def test_lookup_rejects_forbidden_tables(
    auth_client, user_alice, mdb_configured, fake_httpx, table
):
    resp = auth_client(user_alice).get(
        reverse("catalog:catalog-lookup"),
        {"id": "123", "table": table},
    )
    assert resp.status_code == 400
    assert _error(resp)["code"] == "invalid_table"
    assert fake_httpx.called is False


def test_lookup_missing_id(auth_client, user_alice, mdb_configured, fake_httpx):
    resp = auth_client(user_alice).get(reverse("catalog:catalog-lookup"))
    assert resp.status_code == 400
    assert _error(resp)["code"] == "invalid_id"
    assert fake_httpx.called is False


def test_search_rejects_unknown_source(
    auth_client, user_alice, mdb_configured, fake_httpx
):
    resp = auth_client(user_alice).post(
        reverse("catalog:catalog-search"),
        {"query": "mouse", "source": "not-a-source"},
        format="json",
    )
    assert resp.status_code == 400
    assert _error(resp)["code"] == "invalid_source"
    assert fake_httpx.called is False


def test_datasets_rejects_unknown_source(
    auth_client, user_alice, mdb_configured, fake_httpx
):
    resp = auth_client(user_alice).get(
        reverse("catalog:catalog-datasets"),
        {"source": "not-a-source"},
    )
    assert resp.status_code == 400
    assert _error(resp)["code"] == "invalid_source"
    assert fake_httpx.called is False


def test_lookup_maps_mdb_404(auth_client, user_alice, mdb_configured, fake_httpx):
    fake_httpx.reset(
        response=FakeResponse(
            404, {"status": "error", "error": "not found"}
        )
    )
    resp = auth_client(user_alice).get(
        reverse("catalog:catalog-lookup"),
        {"id": "missing-id"},
    )
    assert resp.status_code == 404
    assert _error(resp)["code"] == "catalog_not_found"


def test_maps_mdb_503(auth_client, user_alice, mdb_configured, fake_httpx):
    fake_httpx.reset(
        response=FakeResponse(503, {"status": "error", "error": "down"})
    )
    resp = auth_client(user_alice).get(reverse("catalog:catalog-statistics"))
    assert resp.status_code == 503
    assert _error(resp)["code"] == "catalog_unavailable"


@pytest.mark.parametrize(
    "exc",
    [httpx.TimeoutException("timed out"), httpx.ConnectError("refused")],
)
def test_maps_httpx_network_errors(
    auth_client, user_alice, mdb_configured, fake_httpx, exc
):
    fake_httpx.reset(error=exc)
    resp = auth_client(user_alice).get(reverse("catalog:catalog-statistics"))
    assert resp.status_code == 503
    assert _error(resp)["code"] == "catalog_unavailable"


def test_maps_mdb_401_without_leaking_token(
    auth_client, user_alice, mdb_configured, fake_httpx
):
    leak = "invalid token secret-mdb-token-abc"
    fake_httpx.reset(response=FakeResponse(401, {"error": leak}))
    resp = auth_client(user_alice).get(reverse("catalog:catalog-statistics"))
    assert resp.status_code == 502
    body = _error(resp)
    assert body["code"] == "catalog_auth"
    assert body["error"] == "Catalog authentication failed"
    assert leak not in body["error"]
    assert "secret-mdb-token-abc" not in resp.content.decode()


def test_statistics_success_forwards_json(
    auth_client, user_alice, mdb_configured, fake_httpx
):
    payload = {"status": "ok", "counts": {"dandi": 12}, "extra": [1, 2]}
    fake_httpx.reset(response=FakeResponse(200, payload))
    resp = auth_client(user_alice).get(reverse("catalog:catalog-statistics"))
    assert resp.status_code == 200
    assert resp.json() == payload


def test_datasets_get_forwards_query_params_and_token(
    auth_client, user_alice, mdb_configured, fake_httpx
):
    fake_httpx.reset(response=FakeResponse(200, {"datasets": []}))
    resp = auth_client(user_alice).get(
        reverse("catalog:catalog-datasets"),
        {"source": "dandi", "limit": 20},
    )
    assert resp.status_code == 200
    captured = fake_httpx.captured
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/api/api_datasets")
    assert captured["params"] == {"source": "dandi", "limit": 20}
    assert captured["headers"]["Authorization"] == f"Bearer {MDB_TOKEN}"
