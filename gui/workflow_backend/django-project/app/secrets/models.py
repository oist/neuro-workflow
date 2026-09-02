"""Encrypted owner secrets and an audit trail that never stores plaintext."""

from __future__ import annotations

import re
import uuid

from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone

from .crypto import (
    EncryptedBlob,
    aad_for_user_secret,
    envelope_decrypt,
    envelope_encrypt,
)
from .keys import decrypt_kek_for_version, get_kek

SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def validate_secret_name(value: str) -> None:
    if not SECRET_NAME_RE.match(value or ""):
        raise ValidationError(
            "Secret name must match ^[A-Z][A-Z0-9_]{1,63}$ (e.g. ASPERA_PASSWORD)."
        )


class UserSecret(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="user_secrets",
    )
    name = models.CharField(max_length=64, validators=[validate_secret_name])
    description = models.TextField(blank=True, default="")
    wrapped_dek = models.BinaryField()
    ciphertext = models.BinaryField()
    nonce = models.BinaryField()
    key_version = models.PositiveSmallIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["owner", "name"],
                name="uniq_user_secret_owner_name",
            )
        ]
        ordering = ["name"]
        indexes = [
            models.Index(fields=["owner", "revoked_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.owner_id})"

    @property
    def is_set(self) -> bool:
        return bool(self.ciphertext) and self.revoked_at is None

    def set_plaintext(self, value: str) -> None:
        if not value:
            raise ValidationError("Secret value cannot be empty.")
        if not self.id:
            self.id = uuid.uuid4()
        aad = aad_for_user_secret(self.owner_id, str(self.id))
        kek = get_kek()
        blob = envelope_encrypt(value.encode("utf-8"), kek, aad)
        self.wrapped_dek = blob.wrapped_dek
        self.ciphertext = blob.ciphertext
        self.nonce = blob.nonce
        self.key_version = blob.key_version
        self.revoked_at = None

    def decrypt_plaintext(self) -> str:
        if self.revoked_at is not None:
            raise ValidationError("Secret has been revoked.")
        aad = aad_for_user_secret(self.owner_id, str(self.id))
        kek = decrypt_kek_for_version(self.key_version)
        blob = EncryptedBlob(
            wrapped_dek=bytes(self.wrapped_dek),
            ciphertext=bytes(self.ciphertext),
            nonce=bytes(self.nonce),
            key_version=self.key_version,
        )
        return envelope_decrypt(blob, kek, aad).decode("utf-8")

    def mark_used(self) -> None:
        self.last_used_at = timezone.now()
        self.save(update_fields=["last_used_at"])

    def revoke(self) -> None:
        self.revoked_at = timezone.now()
        self.save(update_fields=["revoked_at"])


class SecretAuditEvent(models.Model):
    class Action(models.TextChoices):
        CREATE = "create", "Create"
        ROTATE = "rotate", "Rotate"
        DELETE = "delete", "Delete"
        INJECT = "inject", "Inject"
        DENIED = "denied", "Denied"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="secret_audit_events",
    )
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="secret_audit_actions",
    )
    secret = models.ForeignKey(
        UserSecret,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_events",
    )
    secret_id_snapshot = models.UUIDField(null=True, blank=True)
    secret_name = models.CharField(max_length=64)
    action = models.CharField(max_length=16, choices=Action.choices)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.action} {self.secret_name}"
