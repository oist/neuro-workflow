"""Remote execution backend - submits jobs to a Slurm cluster over SSH.

Targets the RIKEN compute server (login node reachable over SSH; jobs run via
``sbatch`` on shared storage under ``/data/neuro-workflow``). Authentication is
handled by an ssh-agent running in the backend container (see entrypoint.sh):
the executor does NOT pass ``-i`` unless ``SLURM_SSH_KEY`` is explicitly set, so
a passphrase-protected key unlocked once by an admin "just works".
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from django.conf import settings

from app.workflow.path_utils import batch_run_dir

from .base import ExecutionBackend, ExecutionResult, ExecutionStatus

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "templates" / "slurm_wrapper.sh.template"
)

_SLURM_STATE_MAP = {
    "PENDING": ExecutionStatus.PENDING,
    "RUNNING": ExecutionStatus.RUNNING,
    "COMPLETING": ExecutionStatus.RUNNING,
    "CONFIGURING": ExecutionStatus.RUNNING,
    "COMPLETED": ExecutionStatus.COMPLETED,
    "FAILED": ExecutionStatus.FAILED,
    "TIMEOUT": ExecutionStatus.FAILED,
    "CANCELLED": ExecutionStatus.CANCELLED,
    "NODE_FAIL": ExecutionStatus.FAILED,
    "OUT_OF_MEMORY": ExecutionStatus.FAILED,
    "BOOT_FAIL": ExecutionStatus.FAILED,
    "DEADLINE": ExecutionStatus.FAILED,
    "PREEMPTED": ExecutionStatus.FAILED,
}

_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]


def _ssh_cmd(host: str, user: str, cmd: str, key_path: Optional[str] = None) -> str:
    """Run a command on a remote host via SSH and return stdout."""
    ssh_args = ["ssh", *_SSH_OPTS]
    if key_path:
        ssh_args += ["-i", key_path, "-o", "IdentitiesOnly=yes"]
    ssh_args += [f"{user}@{host}", cmd]
    result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _rsync(
    src: str,
    dst: str,
    host: str,
    user: str,
    key_path: Optional[str] = None,
    to_remote: bool = True,
) -> None:
    """rsync files to/from the remote host (over the same SSH/agent auth)."""
    ssh_e = "ssh " + " ".join(_SSH_OPTS)
    if key_path:
        ssh_e += f" -i {key_path} -o IdentitiesOnly=yes"
    rsync_args = ["rsync", "-az", "--mkpath", "-e", ssh_e]
    if to_remote:
        rsync_args += [src, f"{user}@{host}:{dst}"]
    else:
        rsync_args += [f"{user}@{host}:{src}", dst]
    subprocess.run(rsync_args, check=True, capture_output=True, text=True, timeout=300)


class RemoteSlurmExecutor(ExecutionBackend):
    """Submit workflow scripts to a Slurm cluster via SSH.

    Configuration is read from environment variables:
      SLURM_HOST          - hostname of the login node
      SLURM_USER          - SSH user
      SLURM_SSH_KEY       - path to private key (optional; uses ssh-agent if unset)
      SLURM_REMOTE_DIR    - remote working dir root (shared FS, e.g. /data/neuro-workflow/runs)
      SLURM_PARTITION     - default Slurm partition (RIKEN default: ccalc)
      SLURM_ACCOUNT       - Slurm account (RIKEN: kobetsu_neuro-workflow)
      SLURM_REMOTE_VENV   - venv to activate in the job (e.g. /data/neuro-workflow/local/venv)
      SLURM_PYTHON_MODULE - Environment Module to load for Python (e.g. python/3.11.14)
    """

    def __init__(self):
        self.host = os.getenv("SLURM_HOST", "")
        self.user = os.getenv("SLURM_USER", "")
        self.key_path = os.getenv("SLURM_SSH_KEY") or None
        self.remote_dir = os.getenv("SLURM_REMOTE_DIR", "/data/neuro-workflow/runs")
        self.partition = os.getenv("SLURM_PARTITION", "ccalc")
        self.account = os.getenv("SLURM_ACCOUNT", "kobetsu_neuro-workflow")
        self.remote_venv = os.getenv(
            "SLURM_REMOTE_VENV", "/data/neuro-workflow/local/venv"
        )
        self.python_module = os.getenv("SLURM_PYTHON_MODULE", "python/3.11.14")

    # -- low-level helpers ---------------------------------------------------

    def _ssh(self, cmd: str) -> str:
        return _ssh_cmd(self.host, self.user, cmd, self.key_path)

    def _sync_to_remote(self, local: str, remote: str) -> None:
        _rsync(local, remote, self.host, self.user, self.key_path, to_remote=True)

    def _sync_from_remote(self, remote: str, local: str) -> None:
        _rsync(remote, local, self.host, self.user, self.key_path, to_remote=False)

    def _remote_run_dir(self, run_id: str) -> str:
        return f"{self.remote_dir}/{run_id}"

    def _fetch_results(
        self, run_id: str, remote_run_dir: str, project_id: str
    ) -> dict:
        """Rsync the job's ``results/`` dir back to the app server and index it.

        Results land in ``codes/projects/<project_id>/batch/<run_id>/results/``
        (co-located with the project). Returns an artifacts dict
        ``{"files": [{"path", "size"}, ...]}`` listing every fetched file
        (recursively) relative to that results dir. Best-effort: returns ``{}``
        if there are no results or the copy fails.
        """
        try:
            self._ssh(f"test -d {remote_run_dir}/results")
        except Exception:
            return {}

        local_dir = batch_run_dir(project_id, run_id) / "results"
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._sync_from_remote(f"{remote_run_dir}/results/", str(local_dir) + "/")
        except Exception as exc:
            logger.warning("Failed to fetch results for run %s: %s", run_id, exc)
            return {}

        files = []
        for f in sorted(local_dir.rglob("*")):
            if f.is_file():
                files.append(
                    {
                        "path": str(f.relative_to(local_dir)),
                        "size": f.stat().st_size,
                    }
                )
        return {"files": files}

    def _build_sbatch_extras(self, rr: dict) -> str:
        """Translate a resource_requests dict into #SBATCH directive lines.

        Partition/account default to the RIKEN values but can be overridden
        per-run. --gres is passed through verbatim (e.g. "gpu:L40:1" on gcalc1,
        "gpu:H100:1" on gcalc2); ccalc needs no gres.
        """
        lines = []
        partition = rr.get("partition") or self.partition
        account = rr.get("account") or self.account
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        if account:
            lines.append(f"#SBATCH --account={account}")
        if rr.get("nodes"):
            lines.append(f"#SBATCH --nodes={rr['nodes']}")
        if rr.get("ntasks"):
            lines.append(f"#SBATCH --ntasks={rr['ntasks']}")
        if rr.get("cpus_per_task"):
            lines.append(f"#SBATCH --cpus-per-task={rr['cpus_per_task']}")
        if rr.get("mem"):
            lines.append(f"#SBATCH --mem={rr['mem']}")
        if rr.get("time"):
            lines.append(f"#SBATCH --time={rr['time']}")
        if rr.get("gres"):
            lines.append(f"#SBATCH --gres={rr['gres']}")
        return "\n".join(lines)

    def _render_sbatch(
        self, run_id: str, project_name: str, remote_run_dir: str, rr: dict
    ) -> str:
        safe_job_name = "nw-" + re.sub(r"[^A-Za-z0-9_-]", "-", str(run_id))[:16]
        template = _TEMPLATE_PATH.read_text()
        replacements = {
            "{{JOB_NAME}}": safe_job_name,
            "{{RUN_DIR}}": remote_run_dir,
            "{{SBATCH_EXTRAS}}": self._build_sbatch_extras(rr),
            "{{RUN_ID}}": str(run_id),
            "{{PROJECT_NAME}}": str(project_name),
            "{{SCRIPT_NAME}}": "workflow.py",
            "{{PYTHON_MODULE}}": self.python_module,
            "{{VENV_PATH}}": self.remote_venv,
        }
        rendered = template
        for key, value in replacements.items():
            rendered = rendered.replace(key, value)
        return rendered

    # -- interface implementation -------------------------------------------

    def submit(
        self,
        workflow_id: str,
        project_name: str,
        code: str,
        *,
        run_id: Optional[str] = None,
        resource_requests: Optional[dict] = None,
    ) -> ExecutionResult:
        run_id = run_id or str(uuid.uuid4())
        result = ExecutionResult(
            run_id=run_id,
            status=ExecutionStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
        )
        remote_run_dir = self._remote_run_dir(run_id)
        result.remote_run_dir = remote_run_dir

        local_dir = batch_run_dir(workflow_id, run_id, create=True)
        (local_dir / "workflow.py").write_text(code)
        (local_dir / "run.sbatch").write_text(
            self._render_sbatch(
                run_id, project_name, remote_run_dir, resource_requests or {}
            )
        )

        # Stage the node implementation package alongside the workflow so the
        # generated script (which does ``from nodes.<cat>.<Node> import ...``)
        # can import it on the compute node. When ``python workflow.py`` runs in
        # the run dir, sys.path[0] is that dir, so ``<run_dir>/nodes`` resolves.
        nodes_src = Path(settings.MEDIA_ROOT)
        if nodes_src.is_dir():
            shutil.copytree(
                nodes_src,
                local_dir / "nodes",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                dirs_exist_ok=True,
            )

        try:
            self._ssh(f"mkdir -p {remote_run_dir}")
            self._sync_to_remote(f"{local_dir}/", remote_run_dir + "/")
            output = self._ssh(f"cd {remote_run_dir} && sbatch run.sbatch")
            match = re.search(r"Submitted batch job (\d+)", output)
            if match:
                result.remote_job_id = match.group(1)
                result.status = ExecutionStatus.PENDING
            else:
                result.status = ExecutionStatus.FAILED
                result.error = f"sbatch output not recognized: {output}"
        except Exception as exc:
            logger.exception("Failed to submit Slurm job for run %s", run_id)
            result.status = ExecutionStatus.FAILED
            result.error = str(exc)

        return result

    def get_status(
        self,
        run_id: str,
        *,
        job_id: Optional[str] = None,
        remote_dir: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ExecutionResult:
        result = ExecutionResult(run_id=run_id)
        result.remote_job_id = job_id

        if not job_id:
            result.status = ExecutionStatus.FAILED
            result.error = "No remote job ID recorded for this run"
            return result

        remote_run_dir = remote_dir or self._remote_run_dir(run_id)
        result.remote_run_dir = remote_run_dir

        try:
            out = self._ssh(
                f"sacct -j {job_id} --format=State --noheader --parsable2 | head -1"
            )
            state = out.strip().split("+")[0].upper() if out.strip() else ""
            result.status = _SLURM_STATE_MAP.get(state, ExecutionStatus.RUNNING)
        except Exception as exc:
            logger.warning("sacct failed for job %s: %s", job_id, exc)
            result.status = ExecutionStatus.RUNNING

        if result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
            try:
                result.exit_code = int(
                    self._ssh(f"cat {remote_run_dir}/exit_code.txt").strip()
                )
            except Exception:
                pass
            try:
                result.stdout = self._ssh(f"cat {remote_run_dir}/stdout.log")
            except Exception:
                pass
            try:
                result.stderr = self._ssh(f"cat {remote_run_dir}/stderr.log")
            except Exception:
                pass
            result.finished_at = datetime.now(timezone.utc)

        # On success, pull the result artifacts back to the app server. The
        # DetailView only polls while the run is non-terminal, so this runs once
        # (on the poll that first observes COMPLETED).
        if result.status == ExecutionStatus.COMPLETED and project_id:
            try:
                result.artifacts = self._fetch_results(
                    run_id, remote_run_dir, project_id
                )
            except Exception as exc:
                logger.warning("fetch_results failed for run %s: %s", run_id, exc)

        return result

    def get_logs(self, run_id: str, *, remote_dir: Optional[str] = None) -> str:
        remote_run_dir = remote_dir or self._remote_run_dir(run_id)
        parts = []
        for fname in ("stdout.log", "stderr.log"):
            try:
                content = self._ssh(
                    f"cat {remote_run_dir}/{fname} 2>/dev/null || true"
                )
                if content:
                    parts.append(content)
            except Exception:
                pass
        return "\n".join(parts)

    def cancel(self, run_id: str, *, job_id: Optional[str] = None) -> bool:
        if not job_id:
            return False
        try:
            self._ssh(f"scancel {job_id}")
            return True
        except Exception as exc:
            logger.warning("scancel failed for job %s: %s", job_id, exc)
            return False
