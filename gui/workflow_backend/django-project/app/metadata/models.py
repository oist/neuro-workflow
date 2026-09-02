"""
Metadata app models.
"""
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class CustomDatabase(models.Model):
    """User-defined custom database source for parameter suggestions."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    base_url = models.URLField()
    api_key_wrapped_dek = models.BinaryField(null=True, blank=True)
    api_key_ciphertext = models.BinaryField(null=True, blank=True)
    api_key_nonce = models.BinaryField(null=True, blank=True)
    api_key_key_version = models.PositiveSmallIntegerField(default=1)
    config = models.JSONField(default=dict, blank=True, help_text="Additional configuration (headers, query params, auth type, etc.)")
    adapter_type = models.CharField(max_length=50, default="rest_api")
    is_active = models.BooleanField(default=True)
    is_verified = models.BooleanField(default=False)
    last_tested = models.DateTimeField(null=True, blank=True)
    test_result = models.TextField(blank=True, null=True)
    test_error = models.TextField(blank=True, null=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="custom_databases",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Custom database"
        verbose_name_plural = "Custom databases"

    def __str__(self):
        return self.name

    @property
    def api_key_is_set(self) -> bool:
        if bool(self.api_key_ciphertext):
            return True
        cfg = self.get_config_dict()
        ref = cfg.get("api_key_secret")
        return isinstance(ref, dict) and "__nw_secret" in ref

    def get_api_key(self) -> str:
        if not self.api_key_ciphertext:
            return ""
        from app.secrets.crypto import EncryptedBlob, aad_for_custom_db
        from app.secrets.keys import decrypt_blob

        aad = aad_for_custom_db(self.created_by_id, str(self.id))
        blob = EncryptedBlob(
            wrapped_dek=bytes(self.api_key_wrapped_dek or b""),
            ciphertext=bytes(self.api_key_ciphertext),
            nonce=bytes(self.api_key_nonce or b""),
            key_version=self.api_key_key_version,
        )
        return decrypt_blob(blob, aad).decode("utf-8")

    def resolve_api_key(self) -> str:
        """Decrypt the stored key or materialize a vault SecretRef from config."""
        cfg = self.get_config_dict()
        ref = cfg.get("api_key_secret")
        if isinstance(ref, dict) and "__nw_secret" in ref:
            inner = ref.get("__nw_secret") or {}
            name = inner.get("name") if isinstance(inner, dict) else None
            if name and self.created_by_id:
                from app.secrets.services import materialize_named_secrets

                mapping = materialize_named_secrets(self.created_by, [name], audit=False)
                if name not in mapping:
                    raise ValueError(f"Secret '{name}' is not available")
                return mapping[name]
        return self.get_api_key()

    def set_api_key(self, value: str | None) -> None:
        if not value:
            self.api_key_wrapped_dek = None
            self.api_key_ciphertext = None
            self.api_key_nonce = None
            return
        from app.secrets.crypto import aad_for_custom_db, envelope_encrypt
        from app.secrets.keys import get_kek

        if not self.id:
            self.id = uuid.uuid4()
        aad = aad_for_custom_db(self.created_by_id, str(self.id))
        blob = envelope_encrypt(value.encode("utf-8"), get_kek(), aad)
        self.api_key_wrapped_dek = blob.wrapped_dek
        self.api_key_ciphertext = blob.ciphertext
        self.api_key_nonce = blob.nonce
        self.api_key_key_version = blob.key_version

    # Back-compat for callers that still read database.api_key
    @property
    def api_key(self) -> str:
        return self.get_api_key()

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        self.set_api_key(value)

    def get_config_dict(self):
        """Return config as dict for adapter initialization."""
        return self.config if isinstance(self.config, dict) else {}

    def to_adapter_config(self, openai_client=None):
        """Build config dict for GenericDatabaseAdapter."""
        extra = dict(self.get_config_dict())
        extra.pop("api_key", None)
        extra.pop("api_key_secret", None)
        cfg = {
            "base_url": self.base_url.rstrip("/"),
            "source_name": self.name,
            "enabled": self.is_active,
            "openai_client": openai_client,
            **extra,
            "api_key": self.resolve_api_key(),
        }
        cfg.setdefault("adapter_type", self.adapter_type)
        return cfg
