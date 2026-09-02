"""Reject plaintext secret parameters on flow/node writes. Persist refs only."""

from __future__ import annotations

from typing import Any

from django.core.exceptions import ValidationError

from .redaction import is_secret_ref, make_secret_ref, param_is_secret, secret_ref_id, secret_ref_name
from .services import get_owned_secret, owner_secrets_qs


def _require_ref_or_empty(owner, key: str, value: Any) -> Any:
    if value in (None, ""):
        return value
    if is_secret_ref(value):
        sid = secret_ref_id(value)
        name = secret_ref_name(value)
        owned = None
        if sid:
            owned = get_owned_secret(owner, sid)
        if owned is None and name:
            owned = owner_secrets_qs(owner).filter(name=name).first()
        if owned is None or owned.revoked_at is not None:
            raise ValidationError(f"Secret parameter '{key}' is not an owned vault reference.")
        return make_secret_ref(owned.id, owned.name)
    raise ValidationError(
        f"Secret parameter '{key}' must be a vault reference, not a literal value."
    )


def _secretish_param(param: Any, value: Any) -> bool:
    return param_is_secret(param) or is_secret_ref(value)


def sanitize_node_data(owner, data: Any) -> Any:
    """Fail closed: secret-schema params must be a valid owner SecretRef or empty."""
    if not isinstance(data, dict):
        return data
    schema = data.get("schema") or {}
    params = schema.get("parameters") or {}
    if isinstance(params, dict):
        for key, param in params.items():
            if not isinstance(param, dict):
                continue
            value = param.get("default_value")
            if not _secretish_param(param, value):
                continue
            if "default_value" in param:
                param["default_value"] = _require_ref_or_empty(owner, key, value)
    mods = data.get("parameter_modifications")
    if isinstance(mods, dict):
        for key, info in mods.items():
            if not isinstance(info, dict):
                continue
            param = params.get(key) if isinstance(params, dict) else {}
            secret_mod = param_is_secret(param) or is_secret_ref(info.get("current_value")) or is_secret_ref(
                info.get("original_value")
            )
            field_mods = info.get("field_modifications")
            if isinstance(field_mods, dict) and any(is_secret_ref(v) for v in field_mods.values()):
                secret_mod = True
            if not secret_mod:
                continue
            for field in ("original_value", "current_value"):
                if field in info:
                    info[field] = _require_ref_or_empty(owner, key, info.get(field))
            if isinstance(field_mods, dict):
                for fk, fv in list(field_mods.items()):
                    if param_is_secret(param) or is_secret_ref(fv):
                        field_mods[fk] = _require_ref_or_empty(owner, f"{key}.{fk}", fv)
    return data
