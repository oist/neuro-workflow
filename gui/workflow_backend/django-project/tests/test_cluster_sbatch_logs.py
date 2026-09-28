"""Tests for draft run.sbatch editing, pin, and Slurm log copy-back."""

from pathlib import Path

import pytest
from app.workflow.execution.base import ExecutionStatus
from app.workflow.execution.remote_slurm_executor import (
    RemoteSlurmExecutor,
    normalize_sbatch,
)
from app.workflow.models import FlowProject, WorkflowRun
from app.workflow.path_utils import batch_run_dir
from app.workflow.views import _resolve_run_artifact
from django.urls import reverse


def _make_project(owner, *, name="ClusterProj"):
    return FlowProject.objects.create(name=name, owner=owner, visibility="private")


def test_normalize_sbatch_injects_pinned_directives():
    remote = "/data/neuro-workflow/runs/abc"
    out = normalize_sbatch("echo hi", remote)
    assert out.startswith("#!/bin/bash\n")
    assert f"#SBATCH --chdir={remote}" in out
    assert f"#SBATCH --output={remote}/slurm-%j.out" in out
    assert f"#SBATCH --error={remote}/slurm-%j.err" in out
    assert "echo hi" in out


def test_normalize_sbatch_replaces_existing_output_paths():
    remote = "/data/neuro-workflow/runs/abc"
    raw = "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH --output=/tmp/evil.out",
            "#SBATCH --error=/tmp/evil.err",
            "#SBATCH --chdir=/tmp",
            "python workflow.py",
        ]
    )
    out = normalize_sbatch(raw, remote)
    assert "/tmp/evil" not in out
    assert out.count("#SBATCH --chdir=") == 1
    assert f"#SBATCH --output={remote}/slurm-%j.out" in out
    assert "python workflow.py" in out


def test_normalize_sbatch_rejects_empty_and_oversize():
    with pytest.raises(ValueError, match="empty"):
        normalize_sbatch("   ", "/r")
    with pytest.raises(ValueError, match="64 KiB"):
        normalize_sbatch("x" * (64 * 1024 + 1), "/r")


@pytest.mark.django_db
def test_prepare_creates_draft_without_sbatch(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    ssh_calls = []

    def boom(self, cmd):
        ssh_calls.append(cmd)
        raise AssertionError(f"prepare must not SSH: {cmd}")

    monkeypatch.setattr(RemoteSlurmExecutor, "_ssh", boom)
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    url = reverse("workflow:workflow-run-prepare", args=[project.id])
    resp = client.post(
        url,
        {"resource_requests": {"partition": "ccalc", "time": "00:05:00"}},
        format="json",
    )
    assert resp.status_code == 201, resp.content
    body = resp.json()
    assert body["status"] == "draft"
    assert body["backend"] == "slurm"
    assert ssh_calls == []
    path = batch_run_dir(project.id, body["id"]) / "run.sbatch"
    assert path.is_file()
    text = path.read_text()
    assert "#SBATCH --chdir=" in text
    assert "#SBATCH --output=" in text
    assert "sbatch" not in ssh_calls


@pytest.mark.django_db
def test_put_sbatch_without_chdir_is_pinned(
    auth_client, user_alice, tmp_path, settings
):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    prep = client.post(
        reverse("workflow:workflow-run-prepare", args=[project.id]),
        {"resource_requests": {"partition": "ccalc"}},
        format="json",
    )
    run_id = prep.json()["id"]
    url = reverse("workflow:workflow-run-sbatch", args=[project.id, run_id])
    resp = client.put(
        url,
        {"sbatch": "#!/bin/bash\necho CUSTOM_BODY\n"},
        format="json",
    )
    assert resp.status_code == 200, resp.content
    text = (batch_run_dir(project.id, run_id) / "run.sbatch").read_text()
    assert "CUSTOM_BODY" in text
    assert "#SBATCH --chdir=" in text
    assert "#SBATCH --output=" in text
    assert "#SBATCH --error=" in text


def test_submit_custom_sbatch_is_not_silently_replaced(tmp_path, settings, monkeypatch):
    settings.BASE_DIR = tmp_path
    settings.MEDIA_ROOT = str(tmp_path / "no-nodes")
    ex = RemoteSlurmExecutor()
    monkeypatch.setattr(ex, "_ssh", lambda cmd: "Submitted batch job 4242")
    monkeypatch.setattr(ex, "_sync_to_remote", lambda *a, **k: None)
    marker = "echo USER_EDITED_SBATCH"
    result = ex.submit(
        "proj-id",
        "proj-id",
        "print(1)\n",
        run_id="run-custom",
        resource_requests={"partition": "ccalc"},
        sbatch_text=f"#!/bin/bash\n{marker}\n",
    )
    assert result.status == ExecutionStatus.PENDING
    assert result.remote_job_id == "4242"
    text = (batch_run_dir("proj-id", "run-custom") / "run.sbatch").read_text()
    assert marker in text
    assert "#SBATCH --job-name=" not in text or marker in text
    assert "USER_EDITED_SBATCH" in text


@pytest.mark.django_db
def test_failed_copy_back_writes_err_and_artifact_download(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    run = WorkflowRun.objects.create(
        workflow=project,
        user=user_alice,
        backend=WorkflowRun.Backend.SLURM,
        status=WorkflowRun.Status.PENDING,
        slurm_job_id="1",
        remote_run_dir="/data/neuro-workflow/runs/run-1",
    )

    def fake_ssh(self, cmd):
        if "sacct" in cmd:
            return "FAILED"
        if cmd.startswith("ls "):
            return "slurm-1.err\nstdout.log\nstderr.log"
        if "exit_code.txt" in cmd:
            return "137"
        if "stdout.log" in cmd:
            return ""
        if "stderr.log" in cmd:
            return "oom"
        return ""

    def fake_sync_from(self, remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        name = Path(remote).name
        Path(local).write_text(f"copied:{name}")

    monkeypatch.setattr(RemoteSlurmExecutor, "_ssh", fake_ssh)
    monkeypatch.setattr(RemoteSlurmExecutor, "_sync_from_remote", fake_sync_from)
    monkeypatch.setattr(
        RemoteSlurmExecutor,
        "_fetch_results",
        lambda self, *a, **k: {"files": []},
    )

    client = auth_client(user_alice)
    detail = reverse("workflow:workflow-run-detail", args=[project.id, run.id])
    resp = client.get(detail)
    assert resp.status_code == 200, resp.content
    body = resp.json()
    assert body["status"] == "failed"
    err_path = batch_run_dir(project.id, run.id) / "logs" / "slurm-1.err"
    assert err_path.is_file()
    assert "slurm-1.err" in {
        item["path"].split("/")[-1] for item in body["artifacts"]["logs"]
    }

    art = reverse("workflow:workflow-run-artifact", args=[project.id, run.id])
    got = client.get(art, {"path": "logs/slurm-1.err"})
    assert got.status_code == 200
    assert b"copied:slurm-1.err" in b"".join(got.streaming_content)

    bad = client.get(art, {"path": "../x"})
    assert bad.status_code == 400


@pytest.mark.django_db
def test_nodes_package_is_not_downloadable(auth_client, user_alice, tmp_path, settings):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    run = WorkflowRun.objects.create(
        workflow=project,
        user=user_alice,
        backend=WorkflowRun.Backend.SLURM,
        status=WorkflowRun.Status.COMPLETED,
    )
    nodes = batch_run_dir(project.id, run.id, create=True) / "nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    secret = nodes / "foo.py"
    secret.write_text("print('nope')\n")
    client = auth_client(user_alice)
    art = reverse("workflow:workflow-run-artifact", args=[project.id, run.id])
    resp = client.get(art, {"path": "nodes/foo.py"})
    assert resp.status_code == 400
    with pytest.raises(ValueError):
        _resolve_run_artifact(project.id, run.id, "nodes/foo.py")


@pytest.mark.django_db
def test_community_cannot_submit_or_edit_sbatch(
    auth_client, user_bob, tmp_path, settings, monkeypatch
):
    from app.tenants import TENANT_COMMUNITY, set_user_tenant

    settings.BASE_DIR = tmp_path
    monkeypatch.setattr(RemoteSlurmExecutor, "_ssh", lambda self, cmd: "")
    set_user_tenant(user_bob, TENANT_COMMUNITY)
    project = _make_project(user_bob)
    project.tenant = TENANT_COMMUNITY
    project.save(update_fields=["tenant"])
    client = auth_client(user_bob)
    prepared = client.post(
        reverse("workflow:workflow-run-prepare", args=[project.id]),
        {"resource_requests": {"partition": "ccalc"}},
        format="json",
    )
    assert prepared.status_code == 201, prepared.content
    run_id = prepared.json()["id"]
    sbatch_url = reverse("workflow:workflow-run-sbatch", args=[project.id, run_id])
    got = client.get(sbatch_url)
    assert got.status_code == 403
    assert "project space" in str(got.json()).lower()
    put = client.put(sbatch_url, {"sbatch": "echo no\n"}, format="json")
    assert put.status_code == 403
    submitted = client.post(
        reverse("workflow:workflow-run-submit", args=[project.id]),
        {"backend": "slurm", "run_id": run_id},
        format="json",
    )
    assert submitted.status_code == 403


@pytest.mark.django_db
def test_prepare_from_run_pins_the_new_directory(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    monkeypatch.setattr(RemoteSlurmExecutor, "_ssh", lambda self, cmd: "")
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    first = client.post(
        reverse("workflow:workflow-run-prepare", args=[project.id]),
        {"resource_requests": {"partition": "ccalc"}},
        format="json",
    )
    assert first.status_code == 201, first.content
    old_id = first.json()["id"]
    old_text = (batch_run_dir(project.id, old_id) / "run.sbatch").read_text()
    second = client.post(
        reverse("workflow:workflow-run-prepare", args=[project.id]),
        {"from_run_id": old_id, "resource_requests": {"partition": "ccalc"}},
        format="json",
    )
    assert second.status_code == 201, second.content
    new_id = second.json()["id"]
    new_text = (batch_run_dir(project.id, new_id) / "run.sbatch").read_text()
    assert old_id != new_id
    assert old_id not in new_text
    assert f"runs/{new_id}" in new_text or new_id in new_text
    assert "#SBATCH --chdir=" in new_text
    assert old_text != new_text


@pytest.mark.django_db
def test_submit_without_sbatch_text_uses_the_file_on_disk(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    monkeypatch.setattr(RemoteSlurmExecutor, "_ssh", lambda self, cmd: "Submitted batch job 9")
    seen = {}

    def capture(self, workflow_id, project_name, code, **kwargs):
        seen["sbatch_text"] = kwargs.get("sbatch_text")
        from app.workflow.execution.base import ExecutionResult

        return ExecutionResult(
            run_id=kwargs.get("run_id") or "x",
            status=ExecutionStatus.PENDING,
        )

    monkeypatch.setattr(RemoteSlurmExecutor, "submit", capture)
    project = _make_project(user_alice)
    client = auth_client(user_alice)
    prepared = client.post(
        reverse("workflow:workflow-run-prepare", args=[project.id]),
        {"resource_requests": {"partition": "ccalc"}},
        format="json",
    )
    run_id = prepared.json()["id"]
    path = batch_run_dir(project.id, run_id) / "run.sbatch"
    path.write_text("#!/bin/bash\n# edited in jupyter\n")
    resp = client.post(
        reverse("workflow:workflow-run-submit", args=[project.id]),
        {"backend": "slurm", "run_id": run_id, "sbatch": ""},
        format="json",
    )
    assert resp.status_code == 202, resp.content
    assert seen["sbatch_text"] and "edited in jupyter" in seen["sbatch_text"]


@pytest.mark.django_db
def test_cancel_copies_slurm_logs(
    auth_client, user_alice, tmp_path, settings, monkeypatch
):
    settings.BASE_DIR = tmp_path
    project = _make_project(user_alice)
    run = WorkflowRun.objects.create(
        workflow=project,
        user=user_alice,
        backend=WorkflowRun.Backend.SLURM,
        status=WorkflowRun.Status.RUNNING,
        remote_run_dir="/data/neuro-workflow/runs/job",
        slurm_job_id="42",
    )
    monkeypatch.setattr(RemoteSlurmExecutor, "cancel", lambda self, run_id, job_id=None: True)
    monkeypatch.setattr(
        RemoteSlurmExecutor,
        "_fetch_job_files",
        lambda self, run_id, remote, project_id: {
            "logs": [{"path": "logs/slurm-42.out", "size": 3}]
        },
    )
    client = auth_client(user_alice)
    resp = client.post(
        reverse("workflow:workflow-run-cancel", args=[project.id, run.id]),
        {},
        format="json",
    )
    assert resp.status_code == 200, resp.content
    run.refresh_from_db()
    assert run.status == WorkflowRun.Status.CANCELLED
    assert run.artifacts["logs"][0]["path"] == "logs/slurm-42.out"
