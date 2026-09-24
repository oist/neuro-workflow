"""Short-lived signed tokens that identify the Keycloak user to a shared Lab."""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.core import signing

from app.tenants import (
    get_user_tenant,
    hub_username_for_tenant,
    normalize_tenant,
)

VIEWER_SALT = "nw-jupyter-viewer"
VIEWER_MAX_AGE_SECONDS = 8 * 60 * 60


class ViewerTokenError(Exception):
    pass


def mint_viewer_token(user, *, tenant: str | None = None) -> str:
    tenant = normalize_tenant(tenant or get_user_tenant(user))
    signer = signing.TimestampSigner(salt=VIEWER_SALT)
    return signer.sign_object(
        {
            "uid": user.id,
            "tenant": tenant,
            "hub": hub_username_for_tenant(tenant),
        }
    )


def unsign_viewer_token(token: str, *, max_age: int = VIEWER_MAX_AGE_SECONDS) -> dict:
    signer = signing.TimestampSigner(salt=VIEWER_SALT)
    try:
        payload = signer.unsign_object(token, max_age=max_age)
    except signing.BadSignature as exc:
        raise ViewerTokenError("Invalid or expired viewer token") from exc
    if not isinstance(payload, dict) or "uid" not in payload:
        raise ViewerTokenError("Invalid viewer token payload")
    payload["tenant"] = normalize_tenant(payload.get("tenant"))
    payload["hub"] = payload.get("hub") or hub_username_for_tenant(payload["tenant"])
    return payload


def user_from_viewer_token(token: str):
    payload = unsign_viewer_token(token)
    User = get_user_model()
    try:
        user = User.objects.get(pk=payload["uid"])
    except User.DoesNotExist as exc:
        raise ViewerTokenError("Viewer token user no longer exists") from exc
    return user, payload
