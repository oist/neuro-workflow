"""Jupyter session + visible-path APIs used by the Lab contents filter."""

from __future__ import annotations

from django.db.models import Q
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.auth.authentication import KeycloakAuthentication
from app.tenants import (
    JUPYTER_HONESTY_NOTICE,
    get_user_tenant,
    hub_username_for_tenant,
    is_node_reviewer,
    tenant_query_values,
)
from app.workflow.models import FlowProject
from app.workflow.path_utils import legacy_folder_name
from app.workflow.viewer_tokens import mint_viewer_token

from .jupyter_auth import JupyterViewerTokenAuthentication


def visible_projects_for_user(user):
    tenant = get_user_tenant(user)
    return FlowProject.objects.filter(is_active=True).filter(
        Q(owner=user)
        | (Q(tenant__in=tenant_query_values(tenant)) & Q(visibility=FlowProject.Visibility.PUBLIC))
    )


def _exclusive_legacy_names(tenant: str) -> set[str]:
    """Legacy folder names that belong to exactly one active project in the tenant."""
    counts: dict[str, int] = {}
    for project in FlowProject.objects.filter(
        is_active=True,
        tenant__in=tenant_query_values(tenant),
    ).only("id", "name"):
        folder = legacy_folder_name(project)
        counts[folder] = counts.get(folder, 0) + 1
    return {folder for folder, count in counts.items() if count == 1}


def visible_paths_payload(user) -> dict:
    tenant = get_user_tenant(user)
    projects = list(visible_projects_for_user(user).only("id", "name", "visibility"))
    project_ids = [str(p.id) for p in projects]
    exclusive = _exclusive_legacy_names(tenant)
    legacy_names = []
    for project in projects:
        folder = legacy_folder_name(project)
        if folder in exclusive:
            legacy_names.append(folder)
    return {
        "tenant": tenant,
        "hub_user": hub_username_for_tenant(tenant),
        "project_ids": project_ids,
        "legacy_names": sorted(set(legacy_names)),
        "hide_unlisted_projects": True,
        "notice": JUPYTER_HONESTY_NOTICE,
    }


class JupyterSessionView(APIView):
    """Mint a viewer token and tell the GUI which Hub user to open."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        tenant = get_user_tenant(request.user)
        hub_user = hub_username_for_tenant(tenant)
        token = mint_viewer_token(request.user, tenant=tenant)
        return Response(
            {
                "tenant": tenant,
                "hub_user": hub_user,
                "jupyter_path": f"/user/{hub_user}/",
                "viewer_token": token,
                "is_node_reviewer": is_node_reviewer(request.user),
                "notice": JUPYTER_HONESTY_NOTICE,
            }
        )


class JupyterVisiblePathsView(APIView):
    """Allow-list of project dirs the Lab file browser may show."""

    authentication_classes = [
        JupyterViewerTokenAuthentication,
        KeycloakAuthentication,
    ]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(visible_paths_payload(request.user))
