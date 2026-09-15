# Deploy NeuroWorkflow catalog (mdb Stage 2)

How to reproduce the Catalog tab from `main` after this PR. Secrets stay in
gitignored `.env` files — never commit tokens, keys, or `.env`.

Implementation: `docs/CATALOG_SEARCH.md`. HTTP contract:
`deployment/MDB_CLIENT_CONTRACT.md`.

mdb itself lives in `oist/bm_mindsdb` (separate compose project). This repo
only talks to it over HTTP.

---

## What you are wiring

```
Keycloak user → nginx → NeuroWorkflow /api/catalog/* → mdb-mindsdb:8004
```

- Public ports stay SSH / 80 / 443. mdb binds **`127.0.0.1:8004`** only.
- **Do not** add nginx `/mdb`, iframe the mdb UI, or publish `0.0.0.0:8004`.
- **Do not** set `MDB_ADMIN_TOKEN` in NeuroWorkflow.
- **Do not** put `MDB_*` in `VITE_*`, `gui/.env`, or the frontend image.

---

## 1. Pin the Docker network (this repo)

`gui/docker-compose.yml` names the backend network:

```yaml
networks:
  workflow:
    name: neuro-workflow_workflow
```

After `docker compose up` in `gui/`, `docker network ls` should show
`neuro-workflow_workflow`. Backend hostname on that network is `backend`.

---

## 2. Attach mdb (bm_mindsdb repo, not this one)

On the mdb compose overlay, join the **existing** NeuroWorkflow network. Example
(hostnames and file names may differ slightly in `oist/bm_mindsdb`):

```yaml
services:
  mdb-mindsdb:
    networks:
      - default
      - neuro-workflow_workflow

networks:
  neuro-workflow_workflow:
    external: true
    name: neuro-workflow_workflow
```

Recreate **mdb-mindsdb only** (keep its volume). Do not recreate the
NeuroWorkflow backend (ssh-agent). Bind stays `127.0.0.1:8004`.

After attach, from the NeuroWorkflow **backend container**:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://mdb-mindsdb:8004/
# expect 200
```

Host check (operators / SSH tunnel):

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8004/
# expect 200
```

If mdb was recreated, start in-container MindsDB again with the **mdb admin**
token (SSH / mdb UI). NeuroWorkflow does not start or stop MindsDB.

---

## 3. NeuroWorkflow backend env (gitignored)

Copy placeholders from `gui/workflow_backend/env.template`. In
**`gui/workflow_backend/.env` only** (never print or commit the token):

```
MDB_BASE_URL=http://mdb-mindsdb:8004
MDB_API_TOKEN=<same search token as in the mdb .env>
# optional:
# MDB_TIMEOUT=15
```

Copy `MDB_API_TOKEN` from the mdb environment file on the host. Do not put
`MDB_ADMIN_TOKEN` here.

`get_mdb_config()` reads process env first, then the bind-mounted `.env`
(`override=False`), so a later token add can be picked up with gunicorn SIGHUP
without recreating the backend.

---

## 4. Load code without dropping Slurm ssh-agent

Backend code is bind-mounted in this compose. Prefer:

```bash
# inside neuro-workflow-backend-1: SIGHUP the gunicorn master (container PID 1)
python3 -c "import os,signal; os.kill(1, signal.SIGHUP)"
```

Recreate gunicorn/backend only if the new env vars are not visible after
SIGHUP. Recreate drops ssh-agent; you would need `ssh-add` again.

Frontend in production is a **built nginx image** (no src mount). Rebuild and
recreate **frontend only**:

```bash
cd gui
docker compose -f docker-compose.yml -f docker-compose.prod.yml build frontend
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --no-deps --no-build frontend
```

MCP: `workflow_mcp.py` is bind-mounted; `docker compose restart mcp` (or
`--no-deps`) is enough for the four catalog tools.

Do not `compose down`, do not recreate Hub / db / Keycloak for this feature.

---

## 5. Smoke

Unauthenticated API (route exists, Keycloak required):

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3000/api/catalog/statistics/
# 401
```

Logged-in user in the browser: **Catalog** in the header → `/catalog` →
statistics chips load → keyword search (e.g. `mouse`) returns rows → open a
record.

Failures to report to the mdb side: HTTP status and whether it was
statistics, search, or lookup. Typical causes: 401/403 from mdb (wrong/missing
search token) or timeout (wrong `MDB_BASE_URL`).

---

## Rollback

- Remove `MDB_BASE_URL` / `MDB_API_TOKEN` from `gui/workflow_backend/.env` and
  SIGHUP gunicorn (Catalog API returns `catalog_unconfigured`).
- Revert the frontend image to the previous build if you need the header link
  gone.
- Leave mdb running; it does not depend on NeuroWorkflow.
