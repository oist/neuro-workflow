"""A user's own Claude or Codex key, used by the lab proxy."""

from __future__ import annotations

import base64
import hashlib
import logging
from uuid import UUID

from app.auth.authentication import KeycloakAuthentication
from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import NotebookToken, PersonalAgentCredential

logger = logging.getLogger(__name__)


def _fernet() -> Fernet:
    digest = hashlib.sha256(settings.SECRET_KEY.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_key(value: str) -> str:
    """Return Fernet ciphertext, or an empty string when the key is unset."""
    if not value:
        return ""
    return _fernet().encrypt(value.encode()).decode("ascii")


def decrypt_key(stored: str) -> str | None:
    """Return the saved key, or None when it is unset or unreadable."""
    if not stored:
        return None
    try:
        return _fernet().decrypt(stored.encode("ascii")).decode()
    except (InvalidToken, ValueError, TypeError, UnicodeError):
        logger.warning("A personal agent key could not be read")
        return None


def _clean(value) -> str | None:
    """A string to store, or None when the field should be left unchanged."""
    if not isinstance(value, str):
        return None
    return value.strip()


def credential_flags(credential: PersonalAgentCredential | None) -> dict:
    if credential is None:
        return {"claude": False, "codex": False}
    return {
        "claude": decrypt_key(credential.anthropic_key) is not None,
        "codex": decrypt_key(credential.openai_key) is not None,
    }


def personal_key_for_open_notebook(
    project_id: str, hub_user: str, provider: str
) -> str | None:
    """Key for the user whose notebook tab is open in this Jupyter space.

    The shared lab key remains the fallback when this returns None.
    """
    if provider not in {"anthropic", "openai"}:
        return None
    try:
        UUID(str(project_id))
    except (ValueError, TypeError, AttributeError):
        return None
    token = (
        NotebookToken.objects.filter(
            project_id=project_id,
            hub_user=hub_user,
            expires_at__gt=timezone.now(),
        )
        .select_related("user")
        .first()
    )
    if token is None:
        return None
    credential = PersonalAgentCredential.objects.filter(user=token.user).first()
    if credential is None:
        return None
    stored = (
        credential.anthropic_key if provider == "anthropic" else credential.openai_key
    )
    return decrypt_key(stored)


class PersonalAgentCredentialView(APIView):
    """Save or clear the signed-in user's agent keys. Keys are write-only."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        credential = PersonalAgentCredential.objects.filter(user=request.user).first()
        return Response(credential_flags(credential))

    def put(self, request):
        credential, _created = PersonalAgentCredential.objects.get_or_create(
            user=request.user
        )
        if "claude_key" in request.data:
            claude_key = _clean(request.data.get("claude_key"))
            if claude_key is not None:
                credential.anthropic_key = encrypt_key(claude_key)
        if "codex_key" in request.data:
            codex_key = _clean(request.data.get("codex_key"))
            if codex_key is not None:
                credential.openai_key = encrypt_key(codex_key)
        credential.save()
        return Response(credential_flags(credential))

    def delete(self, request):
        PersonalAgentCredential.objects.filter(user=request.user).delete()
        return Response({"claude": False, "codex": False})
