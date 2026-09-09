"""Redact secret material from API payloads, flow JSON, and generated code views."""

from __future__ import annotations

import copy
import re
from typing import Any

SECRET_REF_KEY = "__nw_secret"
REDACTED = "••••"

_CONTEXT_BLOCKED = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "aspera_pass",
    "authorization",
)


def is_secret_ref(value: Any) -> bool:
    return isinstance(value, dict) and SECRET_REF_KEY in value and isinstance(value[SECRET_REF_KEY], dict)


def make_secret_ref(secret_id: str | None, name: str) -> dict[str, Any]:
    return {SECRET_REF_KEY: {"id": str(secret_id or ""), "name": name}}


def secret_ref_name(value: Any) -> str | None:
    if not is_secret_ref(value):
        return None
    return value[SECRET_REF_KEY].get("name") or None


def secret_ref_id(value: Any) -> str | None:
    if not is_secret_ref(value):
        return None
    sid = value[SECRET_REF_KEY].get("id") or None
    return str(sid) if sid else None


def param_is_secret(param: Any) -> bool:
    if not isinstance(param, dict):
        return False
    if param.get("secret") is True:
        return True
    return is_secret_ref(param.get("default_value"))


def redact_param_value(param: Any, value: Any) -> Any:
    if is_secret_ref(value):
        ref = value[SECRET_REF_KEY]
        return make_secret_ref(ref.get("id"), ref.get("name") or "")
    if param_is_secret(param) and value not in (None, ""):
        name = secret_ref_name(param.get("default_value")) or ""
        return make_secret_ref(secret_ref_id(param.get("default_value")), name or "REDACTED")
    return value


def _redact_modifications(params: dict, modifications: Any) -> Any:
    if not isinstance(modifications, dict):
        return modifications
    out = copy.deepcopy(modifications)
    for key, info in out.items():
        if not isinstance(info, dict):
            continue
        param = params.get(key) if isinstance(params, dict) else {}
        if not param_is_secret(param) and not is_secret_ref(info.get("current_value")) and not is_secret_ref(info.get("original_value")):
            field_mods = info.get("field_modifications") or {}
            if not any(
                is_secret_ref(field_mods.get(k))
                for k in ("default_value", "default_value_original")
            ):
                continue
        for field in ("original_value", "current_value"):
            if field in info:
                info[field] = redact_param_value(param, info.get(field))
        field_mods = info.get("field_modifications")
        if isinstance(field_mods, dict):
            for fk, fv in list(field_mods.items()):
                field_mods[fk] = redact_param_value(param, fv)
    return out


def redact_node_data(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    redacted = copy.deepcopy(data)
    schema = redacted.get("schema") or {}
    params = schema.get("parameters") or {}
    if isinstance(params, dict):
        for key, param in params.items():
            if not isinstance(param, dict):
                continue
            if "default_value" in param:
                param["default_value"] = redact_param_value(param, param.get("default_value"))
    if "parameter_modifications" in redacted:
        redacted["parameter_modifications"] = _redact_modifications(
            params if isinstance(params, dict) else {},
            redacted.get("parameter_modifications"),
        )
    return redacted


def redact_flow_payload(flow: Any) -> Any:
    if not isinstance(flow, dict):
        return flow
    out = copy.deepcopy(flow)
    nodes = out.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and "data" in node:
                node["data"] = redact_node_data(node.get("data"))
    return out


def redact_generated_code(source: str | None) -> str | None:
    """Keep SecretRef(...) lines; never required to know plaintext.

    Also masks obvious password= / api_key= string literals as defense in depth.
    """
    if source is None:
        return None
    patterns = [
        re.compile(r"(password\s*=\s*)(['\"])([^'\"]+)\2", re.IGNORECASE),
        re.compile(r"(api_key\s*=\s*)(['\"])([^'\"]+)\2", re.IGNORECASE),
        re.compile(r"(aspera_pass\s*[:=]\s*)(['\"])([^'\"]+)\2", re.IGNORECASE),
    ]
    out = source
    for pat in patterns:
        out = pat.sub(rf"\1\2{REDACTED}\2", out)
    return out


def blocked_context_keys(context: Any) -> list[str]:
    if not isinstance(context, dict):
        return []
    hits = []
    for key in context.keys():
        lowered = str(key).lower().replace("-", "_")
        if any(token in lowered for token in _CONTEXT_BLOCKED):
            hits.append(str(key))
    return hits


class WorkflowContextBlockedError(ValueError):
    def __init__(self, keys: list[str]):
        self.keys = keys
        super().__init__(
            "workflow_context must not contain "
            + ", ".join(keys)
            + "; store credentials in Settings → Secrets."
        )


def assert_context_has_no_secrets(context: Any) -> None:
    hits = blocked_context_keys(context)
    if hits:
        raise WorkflowContextBlockedError(hits)
