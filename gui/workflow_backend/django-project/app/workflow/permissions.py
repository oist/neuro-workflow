"""Permission rules for FlowProject visibility and tenant isolation.

Current product rule:
- Private projects are visible and editable only to the owner.
- Public projects are visible and editable by any authenticated user in the
  same tenant.
- A project-space user can also read a public community project. That read
  does not grant edits, deletes, or visibility changes.
- A community user does not see project-space projects.
- DELETE and visibility changes are owner-only, even for public projects.
"""

from rest_framework import exceptions, permissions

from app.tenants import (
    TENANT_COMMUNITY,
    TENANT_PROJECT,
    get_user_tenant,
    normalize_tenant,
    same_tenant,
    tenant_query_values,
)

from .models import FlowProject


SAFE_METHODS = permissions.SAFE_METHODS


def _is_visibility_change(request) -> bool:
    return request.method in ("PATCH", "PUT") and "visibility" in (request.data or {})


def _project_of(obj):
    project = obj if isinstance(obj, FlowProject) else getattr(obj, "project", None)
    if project is None:
        project = getattr(obj, "workflow", None)
    return project


def project_user_may_read_community(user, project) -> bool:
    """Project-space users may open a public community project, read-only."""
    if project is None or not getattr(user, "is_authenticated", False):
        return False
    return (
        get_user_tenant(user) == TENANT_PROJECT
        and normalize_tenant(getattr(project, "tenant", None)) == TENANT_COMMUNITY
        and project.visibility == FlowProject.Visibility.PUBLIC
    )


class IsAuthenticatedAndProjectVisible(permissions.BasePermission):
    """Allow access when the user owns the project or it is public in-tenant."""

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated)

    def has_object_permission(self, request, view, obj):
        project = _project_of(obj)
        if project is None:
            return False
        if project.owner_id == request.user.id:
            return True
        if not same_tenant(request.user, project):
            return (
                request.method in SAFE_METHODS
                and project_user_may_read_community(request.user, project)
            )
        return project.visibility == FlowProject.Visibility.PUBLIC


class IsOwnerForDestructive(permissions.BasePermission):
    """DELETE and visibility changes require ownership.

    Non-destructive writes on public projects are intentionally allowed for
    authenticated non-owners in the same tenant.
    """

    def has_permission(self, request, view):
        return True

    def has_object_permission(self, request, view, obj):
        project = obj if isinstance(obj, FlowProject) else getattr(obj, "project", None)
        if project is None:
            project = getattr(obj, "workflow", None)
        if project is None:
            return False
        if not same_tenant(request.user, project):
            return request.method in SAFE_METHODS and project_user_may_read_community(
                request.user, project
            )
        if request.method == "DELETE":
            return project.owner_id == request.user.id
        if _is_visibility_change(request):
            return project.owner_id == request.user.id
        return True


def get_accessible_project(request, project_id, *, write: bool = False) -> FlowProject:
    """Resolve a FlowProject for a child resource and enforce visibility/owner rules.

    Notes:
        write=True intentionally allows writes by non-owners when the parent
        project is public. Owner-only constraints for DELETE and visibility
        updates are enforced by IsOwnerForDestructive at the view level.

    Raises:
        NotFound: if the project does not exist or the user cannot see it.
        PermissionDenied: if write access is required but disallowed.
    """
    if not request.user or not request.user.is_authenticated:
        raise exceptions.NotAuthenticated()

    try:
        project = FlowProject.objects.get(id=project_id, is_active=True)
    except FlowProject.DoesNotExist:
        raise exceptions.NotFound("Project not found.")

    in_tenant = same_tenant(request.user, project)
    cross_read = project_user_may_read_community(request.user, project)
    if not in_tenant and not cross_read:
        raise exceptions.NotFound("Project not found.")

    is_owner = project.owner_id == request.user.id
    is_public = project.visibility == FlowProject.Visibility.PUBLIC
    if not (is_owner or is_public):
        raise exceptions.NotFound("Project not found.")

    if write and not in_tenant:
        raise exceptions.PermissionDenied("Not allowed to modify this project.")
    if write and not (is_owner or is_public):
        raise exceptions.PermissionDenied("Not allowed to modify this project.")

    return project


def tenant_queryset(user):
    """Active projects the user may list in their tenant."""
    from django.db.models import Q

    user_tenant = get_user_tenant(user)
    visible = Q(owner=user) | (
        Q(tenant__in=tenant_query_values(user_tenant))
        & Q(visibility=FlowProject.Visibility.PUBLIC)
    )
    if user_tenant == TENANT_PROJECT:
        visible |= Q(tenant__in=tenant_query_values(TENANT_COMMUNITY)) & Q(
            visibility=FlowProject.Visibility.PUBLIC
        )
    return FlowProject.objects.filter(is_active=True).filter(visible)
