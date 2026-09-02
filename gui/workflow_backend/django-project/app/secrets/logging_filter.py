"""Contextvar-backed logging filter that never writes known secret values."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar

_secret_values: ContextVar[tuple[str, ...]] = ContextVar("nw_secret_values", default=())


def register_secret_values(*values: str) -> None:
    current = list(_secret_values.get())
    for value in values:
        if value and value not in current:
            current.append(value)
    _secret_values.set(tuple(current))


def clear_secret_values() -> None:
    _secret_values.set(())


@contextmanager
def secret_value_scope(*values: str):
    token = _secret_values.set(tuple(v for v in values if v))
    try:
        yield
    finally:
        _secret_values.reset(token)


def redact_text(text: str, extra_values: tuple[str, ...] | None = None) -> str:
    if not text:
        return text
    values = list(_secret_values.get())
    if extra_values:
        values.extend(extra_values)
    out = text
    for value in sorted((v for v in values if v), key=len, reverse=True):
        out = out.replace(value, "••••")
    return out


class SecretValueFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = redact_text(msg)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True
