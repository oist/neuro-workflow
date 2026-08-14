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
)
from app.workflow.models import FlowProject
from app.workflow.viewer_tokens import mint_viewer_token

from .jupyter_auth import JupyterViewerTokenAuthentication


def visible_projects_for_user(user):
    tenant = get_user_tenant(user)
    return FlowProject.objects.filter(is_active=True, tenant=tenant).filter(
        Q(owner=user) | Q(visibility=FlowProject.Visibility.PUBLIC)
    )


def visible_paths_payload(user) -> dict:
    tenant = get_user_tenant(user)
    projects = list(visible_projects_for_user(user).only("id", "name", "visibility"))
    project_ids = [str(p.id) for p in projects]
    legacy_names = []
    for project in projects:
        name = (project.name or "").replace(" ", "")
        if name:
            capitalized = name[:1].upper() + name[1:] if name else name
            legacy_names.append(capitalized)
            legacy_names.append(name)
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
