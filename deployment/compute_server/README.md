# Remote Slurm execution - operator runbook (RIKEN compute server)

This enables the Django backend to submit a workflow as a **Slurm batch job** on
the RIKEN compute server over SSH, poll its status, and copy logs and results
back into the NeuroWorkflow project tree.

It is **additive and OFF by default**: the live JupyterHub run path is untouched,
and this path runs only when a run is submitted with `backend=slurm`. The
ssh-agent in the backend container holds no key until an admin unlocks it.

## User flow (GUI)

1. **Prepare (draft).** Opening **Run on Compute Cluster** creates a `draft`
   `WorkflowRun` and writes `run.sbatch` to
   `codes/projects/<project_id>/batch/<run_id>/run.sbatch` (the same tree
   Jupyter mounts). No SSH and no `sbatch` yet.
2. **Edit.** The Cluster Run modal shows the script. The user can edit it
   there or **Edit in Jupyter** (then **Reload from project**). Changing
   partition/CPU/memory/time regenerates the script unless the textarea is
   dirty (confirm before overwrite).
3. **Submit.** `POST /api/workflow/<id>/runs/submit/` with `backend: "slurm"`,
   the draft `run_id`, and the (possibly edited) `sbatch` text. The executor
   pins `#SBATCH --chdir` / `--output` / `--error` to the remote run dir, rsyncs
   the batch dir, and runs `sbatch run.sbatch`.
4. **Copy-back.** On the first poll that sees COMPLETED, FAILED, or CANCELLED,
   the backend rsyncs `slurm-*.out`, `slurm-*.err`, `stdout.log`, `stderr.log`,
   `exit_code.txt`, and `manifest.json` into
   `codes/projects/<project_id>/batch/<run_id>/logs/`, and `results/` into
   `.../results/` when that directory exists. The Runs panel lists **Logs** and
   **Results** for download. Failed jobs are copied too (that is when `.err`
   matters).

Closing the modal without submit leaves the draft in the Runs panel (delete
removes it). **Edit script & resubmit** copies the previous `run.sbatch` into a
new draft.

The full UI/API path is `POST /api/workflow/<id>/runs/submit/` with
`{"backend": "slurm"}` (not `/run-submit/`). Drafts use
`POST /api/workflow/<id>/runs/prepare/` and
`GET`/`PUT /api/workflow/<id>/runs/<run_id>/sbatch/`.


## Facts baked into the implementation (from RIKEN)

- Login node: `digitalbrain.brainminds.jp`, user `neuro-workflow`. Used only for
  `ssh`/`rsync`/`sbatch`/`sacct` (login node has a 4 GB / 512-proc cap; no heavy
  or persistent processes). All compute happens inside jobs.
- Account: `kobetsu_neuro-workflow` (the login user is `neuro-workflow`, but the
  Slurm association resolves to this account). Default partition: `ccalc` (CPU). GPU partitions:
  `gcalc1` (`--gres=gpu:L40:N`) and `gcalc2` (`--gres=gpu:H100:N`).
- Working dir: `/data/neuro-workflow` (shared FS, unlimited pool; mounted on all
  nodes). Runs are staged under `/data/neuro-workflow/runs/<run_id>`. `$HOME` has
  a strict quota - do not use it. The OS `/tmp` is tiny - the wrapper redirects
  `TMPDIR` into the run dir.
- No conda (institutional licensing). Python via Environment Modules
  (`module load python/3.11.14`) + a venv at `/data/neuro-workflow/local/venv`.
- `sacct` is enabled (used for status); interactive `srun` is currently
  unavailable - we only use `sbatch`.

## Configuration (env vars read by `RemoteSlurmExecutor`)

Defaults are set in `gui/docker-compose.prod.yml`; override in `gui/.env` if needed.

- `SLURM_HOST=digitalbrain.brainminds.jp`
- `SLURM_USER=neuro-workflow`
- `SLURM_REMOTE_DIR=/data/neuro-workflow/runs`
- `SLURM_PARTITION=ccalc`
- `SLURM_ACCOUNT=kobetsu_neuro-workflow`
- `SLURM_REMOTE_VENV=/data/neuro-workflow/local/venv`
- `SLURM_PYTHON_MODULE=python/3.11.14`
- `SLURM_SSH_KEY` is intentionally **unset** so auth uses the ssh-agent.

---

## Phase 1 - prove the pipe

### A. Compute-server venv (one-time)

On the compute server, as the `neuro-workflow` user:

```bash
ssh neuro-workflow@digitalbrain.brainminds.jp
# copy setup_compute_venv.sh over (scp) or paste it, then:
bash setup_compute_venv.sh            # base: neuroworkflow + numpy
```

### B. Deploy the backend (app server) - additive

```bash
cd /data/neuro-workflow/gui
docker compose -f docker-compose.yml -f docker-compose.prod.yml build backend
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d backend
```

Rebuild adds `openssh-client`+`rsync` and the ssh-agent entrypoint. The key is
mounted read-only; nothing else changes.

### C. Unlock the SSH key (admin, once per backend restart)

```bash
BACKEND=$(docker compose -f /data/neuro-workflow/gui/docker-compose.yml ps -q backend)
docker exec -e SSH_AUTH_SOCK=/tmp/ssh-agent.sock -it "$BACKEND" ssh-add /root/.ssh/id_ed25519
# enter the key passphrase once; verify:
docker exec -e SSH_AUTH_SOCK=/tmp/ssh-agent.sock "$BACKEND" ssh-add -l
```

If `ssh-add -l` shows the fingerprint, the backend can now submit jobs. On every
backend restart the agent memory clears - rerun this one command.

### D. Verify the pipe (admin)

First confirm the container can reach the login node and authenticate:

```bash
docker exec -e SSH_AUTH_SOCK=/tmp/ssh-agent.sock "$BACKEND" \
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  neuro-workflow@digitalbrain.brainminds.jp 'hostname; sacct --version'
```

Then exercise the executor directly (bypasses HTTP/Keycloak), using the test
workflow in this folder:

```bash
docker exec -e SSH_AUTH_SOCK=/tmp/ssh-agent.sock -i "$BACKEND" \
  python django-project/manage.py shell <<'PY'
from app.workflow.execution import RemoteSlurmExecutor
ex = RemoteSlurmExecutor()
code = open('/django-app/deployment/compute_server/test_workflow.py').read()
r = ex.submit('pipe-test', 'pipe-test', code,
              run_id='pipe-test-1', resource_requests={'time': '00:05:00'})
print('submit:', r.status, r.remote_job_id, r.remote_run_dir, r.error)
PY
```

Wait a few seconds, then poll:

```bash
docker exec -e SSH_AUTH_SOCK=/tmp/ssh-agent.sock -i "$BACKEND" \
  python django-project/manage.py shell <<'PY'
from app.workflow.execution import RemoteSlurmExecutor
ex = RemoteSlurmExecutor()
s = ex.get_status('pipe-test-1', job_id='<JOB_ID_FROM_SUBMIT>',
                  remote_dir='/data/neuro-workflow/runs/pipe-test-1')
print('status:', s.status, 'exit:', s.exit_code)
print(s.stdout)
PY
```

Success = status `COMPLETED`, exit code `0`, and stdout containing the test
summary. The full UI/API path (`POST /api/workflow/<id>/runs/submit/` with
`{"backend": "slurm"}`) uses exactly this executor. After a terminal status,
`get_status(..., project_id=...)` also copies `slurm-*.out/.err` into
`codes/projects/<project_id>/batch/<run_id>/logs/`.

---

## Phase 2 - real simulator (NEST)

Re-run the venv setup with the simulator extras (the compute nodes are Linux
x86_64, so the NEST PyPI wheel + Python 3.11 module path works without conda):

```bash
ssh neuro-workflow@digitalbrain.brainminds.jp
WITH_NEST=1 bash setup_compute_venv.sh
```

If a dependency must build from source and is killed by the login-node limits,
run the same command inside a short `sbatch` on `ccalc`. If the NEST wheel is
ever unavailable for the node architecture, fall back to a source build under
`/data/neuro-workflow/local` exposed via an Environment Module, and set
`SLURM_PYTHON_MODULE`/activation accordingly - the wrapper already does
`module load` + venv activate.

Then submit a NEST PointNeuron workflow with `backend=slurm` and verify it
returns artifacts (SONATA files, spike/voltage reports) from the run dir.
