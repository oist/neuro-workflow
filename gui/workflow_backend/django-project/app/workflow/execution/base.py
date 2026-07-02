"""Abstract execution backend for running workflow scripts."""

from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class ExecutionStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ExecutionStatus = ExecutionStatus.PENDING
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    submitted_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    remote_job_id: Optional[str] = None
    remote_run_dir: Optional[str] = None
    artifacts: dict = field(default_factory=dict)


class ExecutionBackend(ABC):
    """Interface for submitting and tracking workflow runs."""

    @abstractmethod
    def submit(
        self,
        workflow_id: str,
        project_name: str,
        code: str,
        *,
        run_id: Optional[str] = None,
        resource_requests: Optional[dict] = None,
    ) -> ExecutionResult:
        """Submit a workflow run. Returns immediately with a pending result.

        ``run_id`` lets the caller pin the run identifier (e.g. the DB
        WorkflowRun id) so staging dirs, remote dirs and later status polls all
        line up. If omitted, a fresh UUID is generated.
        """
        ...

    @abstractmethod
    def get_status(
        self,
        run_id: str,
        *,
        job_id: Optional[str] = None,
        remote_dir: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Poll the status of a previously submitted run.

        ``job_id`` and ``remote_dir`` are supplied by the caller from persisted
        state so the backend does not depend on process-local memory.
        ``project_id`` lets the backend locate the per-project working dir when
        fetching results.
        """
        ...

    @abstractmethod
    def get_logs(self, run_id: str, *, remote_dir: Optional[str] = None) -> str:
        """Return combined stdout+stderr collected so far."""
        ...

    @abstractmethod
    def cancel(self, run_id: str, *, job_id: Optional[str] = None) -> bool:
        """Attempt to cancel a running job. Returns True if cancelled."""
        ...
