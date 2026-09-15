"""App tenants: project vs community.

Canonical slugs are ``project`` / ``community``. Live Keycloak groups
``nw-internal`` / ``nw-hackathon``, Hub users ``internal`` / ``hackathon`` /
``user1``, and tenant slugs ``internal`` / ``hackathon`` stay accepted aliases.
Existing users with no group are treated as project (and assigned that group
on first login with no tenant claim).
"""

from __future__ import annotations

import os

from django.contrib.auth.models import Group

TENANT_PROJECT = "project"
TENANT_COMMUNITY = "community"
TENANT_CHOICES = (
    (TENANT_PROJECT, "Project"),
    (TENANT_COMMUNITY, "Community"),
)

TENANT_PROJECT_ALIASES = frozenset({"project", "internal"})
TENANT_COMMUNITY_ALIASES = frozenset({"community", "hackathon"})

GROUP_PROJECT = "nw-project"
GROUP_COMMUNITY = "nw-community"
GROUP_PROJECT_LEGACY = "nw-internal"
GROUP_COMMUNITY_LEGACY = "nw-hackathon"
GROUP_NODE_REVIEWERS = "node-reviewers"

PROJECT_GROUPS = frozenset({GROUP_PROJECT, GROUP_PROJECT_LEGACY})
COMMUNITY_GROUPS = frozenset({GROUP_COMMUNITY, GROUP_COMMUNITY_LEGACY})
TENANT_GROUPS = (GROUP_PROJECT, GROUP_COMMUNITY)

HUB_USER_PROJECT = os.environ.get("JUPYTERHUB_PROJECT_USER", "internal")
HUB_USER_COMMUNITY = os.environ.get("JUPYTERHUB_COMMUNITY_USER", "hackathon")
HUB_USER_LEGACY = "user1"

JUPYTER_HONESTY_NOTICE = (
    "Jupyter file browser hides other private projects in this space. "
    "The kernel and terminal can still see every path mounted in this Lab. "
    "Isolation between project and community spaces is filesystem-level."
)


def normalize_tenant(value: str | None) -> str:
    key = (value or "").strip().lower()
    if key in TENANT_COMMUNITY_ALIASES:
        return TENANT_COMMUNITY
    if key in TENANT_PROJECT_ALIASES:
        return TENANT_PROJECT
    return TENANT_PROJECT


def tenant_query_values(value: str | None) -> tuple[str, ...]:
    """Canonical tenant plus live aliases, for queryset filters during cutover."""
    tenant = normalize_tenant(value)
    if tenant == TENANT_COMMUNITY:
        return (TENANT_COMMUNITY, "hackathon")
    return (TENANT_PROJECT, "internal")


def hub_username_for_tenant(tenant: str | None) -> str:
    """Hub username for a tenant. Defaults stay live-safe (internal/hackathon)."""
    if normalize_tenant(tenant) == TENANT_COMMUNITY:
        return HUB_USER_COMMUNITY
    return HUB_USER_PROJECT


def ensure_tenant_groups() -> dict[str, Group]:
    names = (
        GROUP_PROJECT,
        GROUP_COMMUNITY,
        GROUP_NODE_REVIEWERS,
        GROUP_PROJECT_LEGACY,
        GROUP_COMMUNITY_LEGACY,
    )
    return {name: Group.objects.get_or_create(name=name)[0] for name in names}


def get_user_tenant(user) -> str:
    if user is None or not getattr(user, "is_authenticated", False):
        return TENANT_PROJECT
    names = set(user.groups.values_list("name", flat=True))
    if names & PROJECT_GROUPS:
        return TENANT_PROJECT
    if names & COMMUNITY_GROUPS:
        return TENANT_COMMUNITY
    return TENANT_PROJECT


def set_user_tenant(user, tenant: str) -> str:
    tenant = normalize_tenant(tenant)
    names = set(user.groups.values_list("name", flat=True))
    if tenant == TENANT_COMMUNITY:
        needs = GROUP_COMMUNITY not in names or bool(names & PROJECT_GROUPS)
    else:
        needs = GROUP_PROJECT not in names or bool(names & COMMUNITY_GROUPS)
    if not needs:
        return tenant
    groups = ensure_tenant_groups()
    if tenant == TENANT_COMMUNITY:
        user.groups.remove(groups[GROUP_PROJECT])
        user.groups.remove(groups[GROUP_PROJECT_LEGACY])
        user.groups.add(groups[GROUP_COMMUNITY])
    else:
        user.groups.remove(groups[GROUP_COMMUNITY])
        user.groups.remove(groups[GROUP_COMMUNITY_LEGACY])
        user.groups.add(groups[GROUP_PROJECT])
    return tenant


def is_node_reviewer(user) -> bool:
    if user is None or not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_staff", False) or getattr(user, "is_superuser", False):
        return True
    return user.groups.filter(name=GROUP_NODE_REVIEWERS).exists()


def same_tenant(user, obj) -> bool:
    obj_tenant = getattr(obj, "tenant", None)
    if obj_tenant is None:
        return True
    return normalize_tenant(obj_tenant) == get_user_tenant(user)


def _claim_strings(payload: dict) -> list[str]:
    values: list[str] = []
    groups = payload.get("groups") or []
    if isinstance(groups, str):
        groups = [groups]
    values.extend(str(g) for g in groups)
    realm = payload.get("realm_access") or {}
    roles = realm.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    values.extend(str(r) for r in roles)
    return values


def _claim_name_set(payload: dict) -> set[str]:
    """Exact group/role names (last path segment), not substring matches."""
    names: set[str] = set()
    for raw in _claim_strings(payload):
        text = str(raw).strip().strip("/")
        if not text:
            continue
        names.add(text.lower())
        names.add(text.rsplit("/", 1)[-1].lower())
    return names


def tenant_from_claims(payload: dict | None) -> str | None:
    """Return a tenant if the token names one; otherwise None (leave as-is)."""
    if not payload:
        return None
    names = _claim_name_set(payload)
    has_project = bool(names & {g.lower() for g in PROJECT_GROUPS})
    has_community = bool(names & {g.lower() for g in COMMUNITY_GROUPS})
    if has_project:
        return TENANT_PROJECT
    if has_community:
        return TENANT_COMMUNITY
    return None


def sync_user_tenant_from_payload(user, payload: dict | None) -> str:
    claimed = tenant_from_claims(payload)
    if claimed:
        return set_user_tenant(user, claimed)
    names = set(user.groups.values_list("name", flat=True))
    if not (names & PROJECT_GROUPS) and not (names & COMMUNITY_GROUPS):
        return set_user_tenant(user, TENANT_PROJECT)
    return get_user_tenant(user)
