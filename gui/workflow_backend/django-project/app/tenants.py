"""App tenants: internal vs hackathon.

Keycloak groups ``nw-internal`` / ``nw-hackathon`` are synced onto Django
``Group`` membership at login. Existing users with no group are treated as
internal (and assigned that group on first login with no tenant claim).
"""

from __future__ import annotations

from django.contrib.auth.models import Group

TENANT_INTERNAL = "internal"
TENANT_HACKATHON = "hackathon"
TENANT_CHOICES = (
    (TENANT_INTERNAL, "Internal"),
    (TENANT_HACKATHON, "Hackathon"),
)

GROUP_INTERNAL = "nw-internal"
GROUP_HACKATHON = "nw-hackathon"
GROUP_NODE_REVIEWERS = "node-reviewers"

TENANT_GROUPS = (GROUP_INTERNAL, GROUP_HACKATHON)

HUB_USER_INTERNAL = "internal"
HUB_USER_HACKATHON = "hackathon"
HUB_USER_LEGACY = "user1"

JUPYTER_HONESTY_NOTICE = (
    "Jupyter file browser hides other private projects in this space. "
    "The kernel and terminal can still see every path mounted in this Lab. "
    "Isolation between internal and hackathon spaces is filesystem-level."
)


def normalize_tenant(value: str | None) -> str:
    if (value or "").strip().lower() == TENANT_HACKATHON:
        return TENANT_HACKATHON
    return TENANT_INTERNAL


def hub_username_for_tenant(tenant: str | None) -> str:
    if normalize_tenant(tenant) == TENANT_HACKATHON:
        return HUB_USER_HACKATHON
    return HUB_USER_INTERNAL


def ensure_tenant_groups() -> dict[str, Group]:
    names = (GROUP_INTERNAL, GROUP_HACKATHON, GROUP_NODE_REVIEWERS)
    return {name: Group.objects.get_or_create(name=name)[0] for name in names}


def get_user_tenant(user) -> str:
    if user is None or not getattr(user, "is_authenticated", False):
        return TENANT_INTERNAL
    names = set(user.groups.values_list("name", flat=True))
    if GROUP_INTERNAL in names:
        return TENANT_INTERNAL
    if GROUP_HACKATHON in names:
        return TENANT_HACKATHON
    return TENANT_INTERNAL


def set_user_tenant(user, tenant: str) -> str:
    tenant = normalize_tenant(tenant)
    groups = ensure_tenant_groups()
    if tenant == TENANT_HACKATHON:
        user.groups.remove(groups[GROUP_INTERNAL])
        user.groups.add(groups[GROUP_HACKATHON])
    else:
        user.groups.remove(groups[GROUP_HACKATHON])
        user.groups.add(groups[GROUP_INTERNAL])
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


def tenant_from_claims(payload: dict | None) -> str | None:
    """Return a tenant if the token names one; otherwise None (leave as-is)."""
    if not payload:
        return None
    tokens = " ".join(_claim_strings(payload)).lower()
    has_internal = "nw-internal" in tokens
    has_hackathon = "nw-hackathon" in tokens
    if has_internal:
        return TENANT_INTERNAL
    if has_hackathon:
        return TENANT_HACKATHON
    return None


def sync_user_tenant_from_payload(user, payload: dict | None) -> str:
    claimed = tenant_from_claims(payload)
    if claimed:
        return set_user_tenant(user, claimed)
    names = set(user.groups.values_list("name", flat=True))
    if GROUP_INTERNAL not in names and GROUP_HACKATHON not in names:
        return set_user_tenant(user, TENANT_INTERNAL)
    return get_user_tenant(user)
