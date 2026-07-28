"""Tests for the bm_mindsdb (mdb) catalog proxy.

mdb has no authentication of its own and exposes an arbitrary-SQL endpoint, so
this proxy is the security boundary in front of it. These tests pin the two
properties that matter: only allow-listed routes and parameters reach mdb, and
every caller must present a Keycloak identity.
"""

import httpx
import pytest
from app.catalog import views as catalog_views
from django.urls import reverse
from rest_framework.test import APIClient


class _FakeResponse:
    def __init__(self, payload, status_code=200, json_ok=True):
        self._payload = payload
        self.status_code = status_code
        self._json_ok = json_ok

    def json(self):
        if not self._json_ok:
            raise ValueError("not json")
        return self._payload


class _StubUser:
    """Minimal authenticated principal.

    These tests exercise routing and the allow-list, not user data, so they use
    a stub rather than a real ORM user and stay database-free.
    """

    is_active = True
    is_authenticated = True
    username = "alice-sub-uuid"


@pytest.fixture
def stub_user():
    return _StubUser()


@pytest.fixture
def mdb_configured(monkeypatch):
    monkeypatch.setenv("MDB_BASE_URL", "http://mdb:8004")
    return "http://mdb:8004"


@pytest.fixture
def captured(monkeypatch):
    """Replace httpx.request with a recorder returning a canned mdb response."""
    calls = []

    def _fake_request(method, url, params=None, timeout=None):
        calls.append(
            {"method": method, "url": url, "params": params, "timeout": timeout}
        )
        return _FakeResponse({"datasets": [], "count": 0})

    monkeypatch.setattr(catalog_views.httpx, "request", _fake_request)
    return calls


# --------------------------------------------------------------------------
# authentication
# --------------------------------------------------------------------------


def test_requires_authentication(mdb_configured, captured):
    resp = APIClient().get(reverse("catalog:statistics"))
    assert resp.status_code in (401, 403)
    assert captured == [], "an unauthenticated request must never reach mdb"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def test_unconfigured_returns_503(monkeypatch, auth_client, stub_user, captured):
    monkeypatch.delenv("MDB_BASE_URL", raising=False)
    resp = auth_client(stub_user).get(reverse("catalog:statistics"))
    assert resp.status_code == 503
    assert resp.json()["available"] is False
    assert captured == []


def test_blank_base_url_treated_as_unconfigured(
    monkeypatch, auth_client, stub_user, captured
):
    monkeypatch.setenv("MDB_BASE_URL", "")
    resp = auth_client(stub_user).get(reverse("catalog:statistics"))
    assert resp.status_code == 503
    assert captured == []


# --------------------------------------------------------------------------
# route mapping
# --------------------------------------------------------------------------


def test_statistics_maps_to_mdb_path(mdb_configured, auth_client, stub_user, captured):
    resp = auth_client(stub_user).get(reverse("catalog:statistics"))
    assert resp.status_code == 200
    assert captured[0]["method"] == "GET"
    assert captured[0]["url"] == "http://mdb:8004/api/api_statistics"


def test_search_forwards_allowed_params_only(
    mdb_configured, auth_client, stub_user, captured
):
    resp = auth_client(stub_user).get(
        reverse("catalog:search"),
        {"q": "mouse", "source": "cbs", "table": "sqlite_master", "limit": "5"},
    )
    assert resp.status_code == 200
    assert captured[0]["url"] == "http://mdb:8004/api/search_api_datasets"
    # `table` and `limit` are not part of mdb's search contract and must be dropped.
    assert captured[0]["params"] == {"q": "mouse", "source": "cbs"}


def test_datasets_forwards_source_and_limit(
    mdb_configured, auth_client, stub_user, captured
):
    auth_client(stub_user).get(
        reverse("catalog:datasets"), {"source": "cbs", "limit": "10", "q": "x"}
    )
    assert captured[0]["url"] == "http://mdb:8004/api/api_datasets"
    assert captured[0]["params"] == {"source": "cbs", "limit": "10"}


def test_lookup_passes_through_mdb_404(
    mdb_configured, auth_client, stub_user, monkeypatch
):
    def _fake_request(method, url, params=None, timeout=None):
        return _FakeResponse({"status": "error", "error": "not found"}, status_code=404)

    monkeypatch.setattr(catalog_views.httpx, "request", _fake_request)
    resp = auth_client(stub_user).get(reverse("catalog:lookup"), {"id": "999999"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "not found"


# --------------------------------------------------------------------------
# local BIDS catalog
# --------------------------------------------------------------------------


def test_local_index_route(mdb_configured, auth_client, stub_user, captured):
    auth_client(stub_user).get(reverse("catalog:local-index", args=["aws", "srpbs-ts"]))
    assert captured[0]["url"] == "http://mdb:8004/api/local_catalog/aws/srpbs-ts"


def test_local_sessions_filters_forwarded(
    mdb_configured, auth_client, stub_user, captured
):
    auth_client(stub_user).get(
        reverse("catalog:local-view", args=["aws", "srpbs-ts", "sessions"]),
        {"site_code": "ATT", "participant_id": "sub-01", "evil": "1"},
    )
    assert (
        captured[0]["url"] == "http://mdb:8004/api/local_catalog/aws/srpbs-ts/sessions"
    )
    assert captured[0]["params"] == {"site_code": "ATT", "participant_id": "sub-01"}


def test_local_rejects_unknown_view(mdb_configured, auth_client, stub_user, captured):
    resp = auth_client(stub_user).get(
        reverse("catalog:local-view", args=["aws", "srpbs-ts", "execute_sql"])
    )
    assert resp.status_code == 404
    assert captured == [], "an unknown view must be rejected before reaching mdb"


# --------------------------------------------------------------------------
# the deliberately excluded surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/catalog/execute_sql/",
        "/api/catalog/database_stats/",
        "/api/catalog/mindsdb_agent/chat/",
        "/api/catalog/start_mindsdb_server/",
    ],
)
def test_dangerous_mdb_routes_are_not_proxied(
    path, mdb_configured, auth_client, stub_user, captured
):
    resp = auth_client(stub_user).post(path, {}, format="json")
    assert resp.status_code == 404
    assert captured == []


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


def test_sync_posts_with_long_timeout(mdb_configured, auth_client, stub_user, captured):
    resp = auth_client(stub_user).post(reverse("catalog:sync"), {}, format="json")
    assert resp.status_code == 200
    assert captured[0]["method"] == "POST"
    assert captured[0]["url"] == "http://mdb:8004/api/sync_apis"
    # A catalog read timeout would abort a real sync partway through.
    assert captured[0]["timeout"] == 600.0


def test_sync_requires_authentication(mdb_configured, captured):
    resp = APIClient().post(reverse("catalog:sync"), {}, format="json")
    assert resp.status_code in (401, 403)
    assert captured == []


# --------------------------------------------------------------------------
# upstream failures
# --------------------------------------------------------------------------


def test_unreachable_mdb_returns_502(
    mdb_configured, auth_client, stub_user, monkeypatch
):
    def _boom(method, url, params=None, timeout=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(catalog_views.httpx, "request", _boom)
    resp = auth_client(stub_user).get(reverse("catalog:statistics"))
    assert resp.status_code == 502
    assert resp.json()["available"] is False


def test_non_json_response_returns_502(
    mdb_configured, auth_client, stub_user, monkeypatch
):
    def _html(method, url, params=None, timeout=None):
        return _FakeResponse(None, status_code=200, json_ok=False)

    monkeypatch.setattr(catalog_views.httpx, "request", _html)
    resp = auth_client(stub_user).get(reverse("catalog:statistics"))
    assert resp.status_code == 502
