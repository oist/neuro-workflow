"""Zip one project's Jupyter folder for download."""

from __future__ import annotations

import re
import tempfile
import zipfile

from app.auth.authentication import KeycloakAuthentication
from app.workflow.path_utils import existing_project_dir
from app.workflow.permissions import get_accessible_project
from django.http import FileResponse
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

_SKIP_PARTS = {".ipynb_checkpoints", "__pycache__"}
_SPOOL_MAX_BYTES = 8 * 1024 * 1024


def archive_download_name(project) -> str:
    """ASCII download name from the project name, or the project id."""
    raw = (getattr(project, "name", None) or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    if not cleaned:
        cleaned = str(project.id)
    return f"{cleaned[:80]}.zip"


class ProjectArchiveView(APIView):
    """Download the files Jupyter already has for this project."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, workflow_id):
        project = get_accessible_project(request, workflow_id, write=False)
        root = existing_project_dir(project, create=False)
        if not root.is_dir():
            return Response(
                {"error": "Project folder not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        spool = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_BYTES)
        try:
            with zipfile.ZipFile(
                spool, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for path in sorted(root.rglob("*")):
                    if path.is_symlink() or not path.is_file():
                        continue
                    if _SKIP_PARTS.intersection(path.parts):
                        continue
                    archive.write(path, arcname=path.relative_to(root).as_posix())
        except Exception:
            spool.close()
            raise
        spool.seek(0)
        return FileResponse(
            spool,
            as_attachment=True,
            filename=archive_download_name(project),
            content_type="application/zip",
        )
