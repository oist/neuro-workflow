"""Per-space JupyterHub passwords.

The project Lab and the community Lab do not share a password. Values come
from the environment. This module does not read or store them.
"""

from __future__ import annotations

import hashlib
import hmac
import os


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
    if not expected or not secret:
        return None
    # compare_digest on str rejects non-ASCII and can reveal a length
    # mismatch; fixed-size digests avoid both.
    if not hmac.compare_digest(_digest(secret), _digest(expected)):
        return None
    return name


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).digest()


def resolve_community_host_path(
    host_project_path: str,
    community_env: str | None,
    hackathon_env: str | None,
) -> str:
    """Choose the host directory mounted into the community Lab.

    The Hub runs in a container that cannot see host folders, so it does not
    detect a community folder by itself. ``HOST_COMMUNITY_PATH`` (or its alias
    ``HOST_HACKATHON_PATH``) selects one; otherwise ``codes-hackathon`` is used.
    """
    explicit = (community_env or "").strip() or (hackathon_env or "").strip()
    if explicit:
        return explicit
    return os.path.join(host_project_path, "codes-hackathon")
