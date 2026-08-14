"""Node governance: private → submitted → approved → public."""

from __future__ import annotations

from django.db.models import Q
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied, ValidationError

from app.tenants import get_user_tenant, is_node_reviewer
from app.box.models import NodeAuditLog, PythonFile


def visible_python_files(user):
    tenant = get_user_tenant(user)
    qs = PythonFile.objects.filter(is_active=True, tenant=tenant)
    own = Q(uploaded_by=user)
    public = Q(status=PythonFile.Status.PUBLIC) | Q(uploaded_by__isnull=True)
    if is_node_reviewer(user):
        review = Q(status=PythonFile.Status.SUBMITTED)
        return qs.filter(own | public | review)
    return qs.filter(own | public)


def log_node_event(python_file, *, actor, action, from_status="", to_status="", comment=""):
    NodeAuditLog.objects.create(
        python_file=python_file,
        actor=actor,
        action=action,
        from_status=from_status or "",
        to_status=to_status or "",
        comment=comment or "",
        tenant=python_file.tenant,
    )


def submit_node(python_file, user):
    if python_file.uploaded_by_id != user.id:
        raise PermissionDenied("Only the owner can submit this node.")
    if python_file.status != PythonFile.Status.PRIVATE:
        raise ValidationError("Only private nodes can be submitted.")
    previous = python_file.status
    python_file.status = PythonFile.Status.SUBMITTED
    python_file.submitted_at = timezone.now()
    python_file.review_comment = ""
    python_file.save(
        update_fields=["status", "submitted_at", "review_comment", "updated_at"]
    )
    log_node_event(
        python_file,
        actor=user,
        action="submitted",
        from_status=previous,
        to_status=python_file.status,
    )
    return python_file


def approve_node(python_file, user, *, make_public: bool = False, comment: str = ""):
    if not is_node_reviewer(user):
        raise PermissionDenied("Node reviewers only.")
    if python_file.status != PythonFile.Status.SUBMITTED:
        raise ValidationError("Only submitted nodes can be approved.")
    previous = python_file.status
    python_file.status = (
        PythonFile.Status.PUBLIC if make_public else PythonFile.Status.APPROVED
    )
    python_file.reviewed_at = timezone.now()
    python_file.reviewed_by = user
    python_file.review_comment = comment or ""
    python_file.save(
        update_fields=[
            "status",
            "reviewed_at",
            "reviewed_by",
            "review_comment",
            "updated_at",
        ]
    )
    log_node_event(
        python_file,
        actor=user,
        action="published" if make_public else "approved",
        from_status=previous,
        to_status=python_file.status,
        comment=comment,
    )
    return python_file


def publish_node(python_file, user, *, comment: str = ""):
    if not is_node_reviewer(user):
        raise PermissionDenied("Node reviewers only.")
    if python_file.status not in (
        PythonFile.Status.APPROVED,
        PythonFile.Status.SUBMITTED,
    ):
        raise ValidationError("Only approved or submitted nodes can be published.")
    previous = python_file.status
    python_file.status = PythonFile.Status.PUBLIC
    python_file.reviewed_at = timezone.now()
    python_file.reviewed_by = user
    if comment:
        python_file.review_comment = comment
    python_file.save(
        update_fields=["status", "reviewed_at", "reviewed_by", "review_comment", "updated_at"]
    )
    log_node_event(
        python_file,
        actor=user,
        action="published",
        from_status=previous,
        to_status=python_file.status,
        comment=comment,
    )
    return python_file


def reject_node(python_file, user, *, comment: str = ""):
    if not is_node_reviewer(user):
        raise PermissionDenied("Node reviewers only.")
    if python_file.status != PythonFile.Status.SUBMITTED:
        raise ValidationError("Only submitted nodes can be rejected.")
    previous = python_file.status
    python_file.status = PythonFile.Status.PRIVATE
    python_file.reviewed_at = timezone.now()
    python_file.reviewed_by = user
    python_file.review_comment = comment or ""
    python_file.save(
        update_fields=[
            "status",
            "reviewed_at",
            "reviewed_by",
            "review_comment",
            "updated_at",
        ]
    )
    log_node_event(
        python_file,
        actor=user,
        action="rejected",
        from_status=previous,
        to_status=python_file.status,
        comment=comment,
    )
    return python_file
