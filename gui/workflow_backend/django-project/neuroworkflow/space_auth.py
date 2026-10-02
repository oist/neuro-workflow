"""Per-space JupyterHub passwords.

The project Lab and the community Lab do not share a password. Values come
from the environment. This module does not read or store them.
"""

from __future__ import annotations

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
    if not expected or not secret or len(secret) != len(expected):
        return None
    if not hmac.compare_digest(secret, expected):
        return None
    return name


def directory_has_files(path: str) -> bool:
    """True when nodes/ or projects/ under path contains a file."""
    for sub in ("nodes", "projects"):
        base = os.path.join(path, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            if filenames:
                return True
    return False


def resolve_community_host_path(
    host_project_path: str,
    community_env: str | None,
    hackathon_env: str | None,
    is_dir,
    has_files,
) -> str:
    """Choose the host directory mounted into the community Lab.

    An explicit environment path wins. A visible codes-community tree is used
    only when it contains files. Otherwise use codes-hackathon, including when
    the Hub process cannot see the host directories at all.
    """
    explicit = (community_env or "").strip() or (hackathon_env or "").strip()
    if explicit:
        return explicit
    community = os.path.join(host_project_path, "codes-community")
    legacy = os.path.join(host_project_path, "codes-hackathon")
    if is_dir(community) and has_files(community):
        return community
    return legacy
