"""Node governance: owner opening + optional review labels.

Opening (community-visible) is ``status=public``. Review is
``review_status`` (unreviewed | in_review | reviewed). Submit / approve /
reject stay; ``NODE_PUBLISH_REQUIRES_REVIEW`` (default off) is the later gate.
"""

from __future__ import annotations

from app.box.models import NodeAuditLog, PythonFile
from app.tenants import (
    get_user_tenant,
    is_node_reviewer,
    normalize_tenant,
    tenant_query_values,
)
from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied, ValidationError


def _requires_review_to_publish() -> bool:
    return bool(getattr(settings, "NODE_PUBLISH_REQUIRES_REVIEW", False))


def visible_python_files(user):
    tenant = get_user_tenant(user)
    qs = PythonFile.objects.filter(is_active=True)
    own = Q(uploaded_by=user)
    in_tenant = Q(tenant__in=tenant_query_values(tenant))
    public = Q(status=PythonFile.Status.PUBLIC) | Q(uploaded_by__isnull=True)
    if is_node_reviewer(user):
        review = Q(review_status=PythonFile.ReviewStatus.IN_REVIEW)
        return qs.filter(own | (in_tenant & (public | review)))
    return qs.filter(own | (in_tenant & public))


def log_node_event(
    python_file, *, actor, action, from_status="", to_status="", comment=""
):
    NodeAuditLog.objects.create(
        python_file=python_file,
        actor=actor,
        action=action,
        from_status=from_status or "",
        to_status=to_status or "",
        comment=comment or "",
        tenant=normalize_tenant(python_file.tenant),
    )


def _is_owner(python_file, user) -> bool:
    return bool(
        user
        and python_file.uploaded_by_id
        and python_file.uploaded_by_id == getattr(user, "id", None)
    )


def _closed_status_for_review(review_status: str) -> str:
    if review_status == PythonFile.ReviewStatus.IN_REVIEW:
        return PythonFile.Status.SUBMITTED
    if review_status == PythonFile.ReviewStatus.REVIEWED:
        return PythonFile.Status.APPROVED
    return PythonFile.Status.PRIVATE


def _set_review_status(python_file, review_status: str, *, keep_public: bool) -> None:
    python_file.review_status = review_status
    if keep_public or python_file.status == PythonFile.Status.PUBLIC:
        python_file.status = PythonFile.Status.PUBLIC
    else:
        python_file.status = _closed_status_for_review(review_status)


def submit_node(python_file, user):
    if not _is_owner(python_file, user):
        raise PermissionDenied("Only the owner can submit this node.")
    if python_file.review_status == PythonFile.ReviewStatus.IN_REVIEW:
        raise ValidationError("This node is already in review.")
    previous = python_file.status
    was_public = python_file.status == PythonFile.Status.PUBLIC
    python_file.submitted_at = timezone.now()
    python_file.review_comment = ""
    _set_review_status(
        python_file, PythonFile.ReviewStatus.IN_REVIEW, keep_public=was_public
    )
    python_file.save(
        update_fields=[
            "status",
            "review_status",
            "submitted_at",
            "review_comment",
            "updated_at",
        ]
    )
    log_node_event(
        python_file,
        actor=user,
        action="submitted",
        from_status=previous,
        to_status=python_file.status,
    )
    return python_file


def _reject_self_review(python_file, user):
    if python_file.uploaded_by_id and python_file.uploaded_by_id == user.id:
        raise PermissionDenied("A different reviewer must approve this node.")


def approve_node(python_file, user, *, make_public: bool = False, comment: str = ""):
    if not is_node_reviewer(user):
        raise PermissionDenied("Node reviewers only.")
    _reject_self_review(python_file, user)
    if python_file.review_status != PythonFile.ReviewStatus.IN_REVIEW:
        raise ValidationError("Only nodes in review can be approved.")
    previous = python_file.status
    was_public = python_file.status == PythonFile.Status.PUBLIC
    python_file.reviewed_at = timezone.now()
    python_file.reviewed_by = user
    python_file.review_comment = comment or ""
    _set_review_status(
        python_file,
        PythonFile.ReviewStatus.REVIEWED,
        keep_public=was_public or make_public,
    )
    python_file.save(
        update_fields=[
            "status",
            "review_status",
            "reviewed_at",
            "reviewed_by",
            "review_comment",
            "updated_at",
        ]
    )
    log_node_event(
        python_file,
        actor=user,
        action="published" if make_public or was_public else "approved",
        from_status=previous,
        to_status=python_file.status,
        comment=comment,
    )
    return python_file


def publish_node(python_file, user, *, comment: str = ""):
    if not _is_owner(python_file, user):
        raise PermissionDenied("Only the owner can open this node.")
    if python_file.status == PythonFile.Status.PUBLIC:
        raise ValidationError("This node is already open.")
    if _requires_review_to_publish() and python_file.review_status != (
        PythonFile.ReviewStatus.REVIEWED
    ):
        raise ValidationError("Review is required before opening this node.")
    previous = python_file.status
    python_file.status = PythonFile.Status.PUBLIC
    python_file.save(update_fields=["status", "updated_at"])
    log_node_event(
        python_file,
        actor=user,
        action="published",
        from_status=previous,
        to_status=python_file.status,
        comment=comment,
    )
    return python_file


def unpublish_node(python_file, user):
    if not _is_owner(python_file, user):
        raise PermissionDenied("Only the owner can close this node.")
    if python_file.status != PythonFile.Status.PUBLIC:
        raise ValidationError("This node is not open.")
    previous = python_file.status
    python_file.status = _closed_status_for_review(python_file.review_status)
    python_file.save(update_fields=["status", "updated_at"])
    log_node_event(
        python_file,
        actor=user,
        action="unpublished",
        from_status=previous,
        to_status=python_file.status,
    )
    return python_file


def reject_node(python_file, user, *, comment: str = ""):
    if not is_node_reviewer(user):
        raise PermissionDenied("Node reviewers only.")
    _reject_self_review(python_file, user)
    if python_file.review_status != PythonFile.ReviewStatus.IN_REVIEW:
        raise ValidationError("Only nodes in review can be rejected.")
    previous = python_file.status
    was_public = python_file.status == PythonFile.Status.PUBLIC
    python_file.reviewed_at = timezone.now()
    python_file.reviewed_by = user
    python_file.review_comment = comment or ""
    _set_review_status(
        python_file, PythonFile.ReviewStatus.UNREVIEWED, keep_public=was_public
    )
    python_file.save(
        update_fields=[
            "status",
            "review_status",
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
