import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient


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
def auth_client():
    """Return a factory that yields an APIClient force-authenticated as a user.

    We bypass the real JWT decoders (CombinedJWTAuthentication) and rely on
    DRF's force_authenticate, which exercises permission_classes correctly.
    """

    def _make(user=None):
        client = APIClient()
        if user is not None:
            client.force_authenticate(user=user)
        return client

    return _make
