"""Audit helper and owner-secret CRUD (values never leave the vault API)."""

from __future__ import annotations

from django.core.exceptions import ValidationError

from .models import SECRET_NAME_RE, SecretAuditEvent, UserSecret, validate_secret_name


def client_ip(request) -> str | None:
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def record_audit(*, owner, actor, secret: UserSecret | None, action: str, ip=None, name: str | None = None) -> None:
    SecretAuditEvent.objects.create(
        owner=owner,
        actor=actor,
        secret=secret,
        secret_id_snapshot=getattr(secret, "id", None),
        secret_name=name or (secret.name if secret else ""),
        action=action,
        ip_address=ip,
    )


def owner_secrets_qs(user):
    return UserSecret.objects.filter(owner=user, revoked_at__isnull=True)


def get_owned_secret(user, secret_id) -> UserSecret | None:
    return UserSecret.objects.filter(owner=user, id=secret_id).first()


def create_user_secret(owner, *, name: str, value: str, description: str = "", actor=None, ip=None) -> UserSecret:
    validate_secret_name(name)
    if UserSecret.objects.filter(owner=owner, name=name, revoked_at__isnull=True).exists():
        raise ValidationError({"name": "A secret with this name already exists."})
    secret = UserSecret(owner=owner, name=name, description=description or "")
    secret.set_plaintext(value)
    secret.save()
    record_audit(owner=owner, actor=actor or owner, secret=secret, action=SecretAuditEvent.Action.CREATE, ip=ip)
    return secret


def rotate_user_secret(secret: UserSecret, *, value: str | None = None, description: str | None = None, actor=None, ip=None) -> UserSecret:
    if secret.revoked_at is not None:
        raise ValidationError("Secret has been revoked.")
    if description is not None:
        secret.description = description
    if value:
        secret.set_plaintext(value)
    secret.save()
    record_audit(
        owner=secret.owner,
        actor=actor or secret.owner,
        secret=secret,
        action=SecretAuditEvent.Action.ROTATE,
        ip=ip,
    )
    return secret


def revoke_user_secret(secret: UserSecret, *, actor=None, ip=None) -> None:
    secret.revoke()
    record_audit(
        owner=secret.owner,
        actor=actor or secret.owner,
        secret=secret,
        action=SecretAuditEvent.Action.DELETE,
        ip=ip,
    )


def materialize_named_secrets(owner, names: list[str], *, actor=None, ip=None, audit: bool = True) -> dict[str, str]:
    """Decrypt owner secrets by name. Missing names are omitted (caller fails closed)."""
    wanted = {n for n in names if n}
    if not wanted:
        return {}
    mapping: dict[str, str] = {}
    qs = owner_secrets_qs(owner).filter(name__in=wanted)
    found = set()
    for secret in qs:
        mapping[secret.name] = secret.decrypt_plaintext()
        secret.mark_used()
        found.add(secret.name)
        if audit:
            record_audit(
                owner=owner,
                actor=actor or owner,
                secret=secret,
                action=SecretAuditEvent.Action.INJECT,
                ip=ip,
            )
    missing = wanted - found
    if missing and audit:
        record_audit(
            owner=owner,
            actor=actor or owner,
            secret=None,
            action=SecretAuditEvent.Action.DENIED,
            ip=ip,
            name=",".join(sorted(missing)),
        )
    return mapping


def suggest_secret_name(owner, base: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in (base or "SECRET").upper())
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "S_" + cleaned
    cleaned = cleaned[:64]
    if SECRET_NAME_RE.match(cleaned) and not owner_secrets_qs(owner).filter(name=cleaned).exists():
        return cleaned
    for i in range(2, 100):
        candidate = f"{cleaned[:60]}_{i}"
        if not owner_secrets_qs(owner).filter(name=candidate).exists():
            return candidate
    return cleaned[:50] + "_X"
