"""Permission rules for FlowProject visibility and tenant isolation.

Current product rule:
- Projects are visible only inside the caller's tenant.
- Private projects are visible/editable only to the owner.
- Public projects are visible and editable by any authenticated user in the
  same tenant.
- DELETE and visibility changes are owner-only, even for public projects.
"""

from rest_framework import exceptions, permissions

from app.tenants import get_user_tenant, same_tenant

from .models import FlowProject


SAFE_METHODS = permissions.SAFE_METHODS


def _is_visibility_change(request) -> bool:
    return request.method in ("PATCH", "PUT") and "visibility" in (request.data or {})


class IsAuthenticatedAndProjectVisible(permissions.BasePermission):
    """Allow access when the user owns the project or it is public in-tenant."""

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated)

    def has_object_permission(self, request, view, obj):
        project = obj if isinstance(obj, FlowProject) else getattr(obj, "project", None)
        if project is None:
            project = getattr(obj, "workflow", None)
        if project is None:
            return False
        if not same_tenant(request.user, project):
            return False
        if project.owner_id == request.user.id:
            return True
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
            return False
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

    if not same_tenant(request.user, project):
        raise exceptions.NotFound("Project not found.")

    is_owner = project.owner_id == request.user.id
    is_public = project.visibility == FlowProject.Visibility.PUBLIC
    if not (is_owner or is_public):
        raise exceptions.NotFound("Project not found.")

    if write and not (is_owner or is_public):
        raise exceptions.PermissionDenied("Not allowed to modify this project.")

    return project


def tenant_queryset(user):
    """Active projects the user may list in their tenant."""
    from django.db.models import Q

    return FlowProject.objects.filter(is_active=True).filter(
        Q(owner=user)
        | (
            Q(tenant=get_user_tenant(user))
            & Q(visibility=FlowProject.Visibility.PUBLIC)
        )
    )
