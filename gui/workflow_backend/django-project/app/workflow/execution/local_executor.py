"""Local execution backend — runs scripts via subprocess on the app server."""

from __future__ import annotations

import logging
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from django.conf import settings

from .base import ExecutionBackend, ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)

# In-memory store for local runs (process-local, non-persistent).
_runs: dict[str, dict] = {}


class LocalExecutor(ExecutionBackend):
    """Run workflow Python scripts as subprocesses on the same host."""

    def __init__(self):
        self.code_dir = Path(settings.BASE_DIR) / "codes" / "projects"
        self.code_dir.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        workflow_id: str,
        project_name: str,
        code: str,
        *,
        run_id: Optional[str] = None,
        resource_requests: Optional[dict] = None,
    ) -> ExecutionResult:
        result = ExecutionResult(
            status=ExecutionStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
        )
        if run_id:
            result.run_id = run_id
        project_dir = self.code_dir / str(workflow_id)
        script_path = project_dir / "workflow.py"

        if not script_path.exists():
            project_dir.mkdir(parents=True, exist_ok=True)
            script_path.write_text(code)

        _runs[result.run_id] = {
            "result": result,
            "process": None,
        }

        thread = threading.Thread(
            target=self._run,
            args=(result.run_id, str(script_path)),
            daemon=True,
        )
        thread.start()
        return result

    def _run(self, run_id: str, script_path: str) -> None:
        entry = _runs.get(run_id)
        if not entry:
            return
        res: ExecutionResult = entry["result"]
        res.status = ExecutionStatus.RUNNING
        res.started_at = datetime.now(timezone.utc)

        try:
            proc = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            entry["process"] = proc
            stdout, stderr = proc.communicate()
            res.stdout = stdout
            res.stderr = stderr
            res.exit_code = proc.returncode
            res.status = (
                ExecutionStatus.COMPLETED
                if proc.returncode == 0
                else ExecutionStatus.FAILED
            )
        except Exception as exc:
            logger.exception("Local execution failed for run %s", run_id)
            res.error = str(exc)
            res.status = ExecutionStatus.FAILED
        finally:
            res.finished_at = datetime.now(timezone.utc)

    def get_status(
        self,
        run_id: str,
        *,
        job_id: Optional[str] = None,
        remote_dir: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ExecutionResult:
        entry = _runs.get(run_id)
        if not entry:
            r = ExecutionResult(run_id=run_id, status=ExecutionStatus.FAILED)
            r.error = "Unknown run_id"
            return r
        return entry["result"]

    def get_logs(self, run_id: str, *, remote_dir: Optional[str] = None) -> str:
        entry = _runs.get(run_id)
        if not entry:
            return ""
        res: ExecutionResult = entry["result"]
        parts = []
        if res.stdout:
            parts.append(res.stdout)
        if res.stderr:
            parts.append(res.stderr)
        if res.error:
            parts.append(f"ERROR: {res.error}")
        return "\n".join(parts)

    def cancel(self, run_id: str, *, job_id: Optional[str] = None) -> bool:
        entry = _runs.get(run_id)
        if not entry:
            return False
        proc = entry.get("process")
        if proc and proc.poll() is None:
            proc.terminate()
            entry["result"].status = ExecutionStatus.CANCELLED
            entry["result"].finished_at = datetime.now(timezone.utc)
            return True
        return False
