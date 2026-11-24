import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient


@pytest.fixture(autouse=True)
def _no_local_llm_providers_file(monkeypatch, tmp_path):
    """Keep a developer's own llm_providers.json out of the tests."""
    monkeypatch.setenv("LLM_PROVIDERS_FILE", str(tmp_path / "no-llm-providers.json"))


class _FakeHubResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


# Tokens the fake JupyterHub recognises -> the identity /hub/api/user returns.
# The first two are per-server tokens of the project and community spaces; the
# last is the hub's own backend service token, which kernels must not use.
HUB_TOKENS = {
    "hub-internal": {"kind": "user", "name": "internal"},
    "hub-hackathon": {"kind": "user", "name": "hackathon"},
    "svc-token": {"kind": "service", "name": "backend"},
}


@pytest.fixture
def fake_hub(monkeypatch):
    """Answer JupyterHub's ``/hub/api/user`` for the tokens in ``HUB_TOKENS``."""
    from django.core.cache import cache

    def fake_hub_get(url, headers=None, timeout=None):
        assert url.endswith("/hub/api/user")
        token = (headers or {}).get("Authorization", "").removeprefix("token ")
        if token in HUB_TOKENS:
            return _FakeHubResponse(200, HUB_TOKENS[token])
        return _FakeHubResponse(403, {"message": "Forbidden"})

    cache.clear()
    monkeypatch.setattr("app.chat.views.httpx.get", fake_hub_get)
    yield
    cache.clear()


@pytest.fixture
def user_alice(db):
    return get_user_model().objects.create_user(
        username="alice-sub-uuid", email="alice@example.com"
    )


@pytest.fixture
def user_bob(db):
    return get_user_model().objects.create_user(
        username="bob-sub-uuid", email="bob@example.com"
    )


@pytest.fixture
def user_admin(db):
    return get_user_model().objects.create_user(
        username="admin-sub-uuid", email="admin@example.com", is_staff=True
    )


@pytest.fixture
def auth_client():
    """Return a factory that yields an APIClient force-authenticated as a user.

    We bypass the real JWT decoder (KeycloakAuthentication) and rely on
    DRF's force_authenticate, which exercises permission_classes correctly.
    """

    def _make(user=None):
        client = APIClient()
        if user is not None:
            client.force_authenticate(user=user)
        return client

    return _make
