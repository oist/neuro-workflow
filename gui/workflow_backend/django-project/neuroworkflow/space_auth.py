"""Per-space JupyterHub passwords.

The project Lab and the community Lab do not share a password. Values come
from the environment. This module does not read or store them.
"""

from __future__ import annotations

import hmac


def authenticate_space_user(
    username: str | None,
    password: str | None,
    *,
    project_user: str,
    community_user: str,
    project_password: str,
    community_password: str,
) -> str | None:
    """Return the Hub user for a matching password, or None.

    ``user1`` is the legacy project login and is accepted only with the
    project password. ``hackathon`` is not a login name.
    """
    name = (username or "").strip()
    secret = password if isinstance(password, str) else ""
    if name == "user1":
        name = project_user
    if name == project_user:
        expected = project_password or ""
    elif name == community_user:
        expected = community_password or ""
    else:
        return None
    if not expected or not secret or len(secret) != len(expected):
        return None
    if not hmac.compare_digest(secret, expected):
        return None
    return name
