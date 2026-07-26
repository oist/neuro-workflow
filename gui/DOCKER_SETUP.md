# Run neuro-workflow GUI from scratch (Docker)

This is the **intended** way to run the full stack: everything (DB, Django API, frontend, JupyterHub, MCP) runs in Docker.

## Prerequisites

- **Docker** and **Docker Compose** installed
- **Git** (repo already cloned at `neuro-workflow`)
- Production deployments using `docker-compose.prod.yml` require Docker Compose v2.24.4+
  because that override file uses standard Compose YAML merge tags.

## 1. Environment files

Three `.env` files are used. They **already exist** in this repo with values set for your machine; if you cloned elsewhere or need to recreate them:

| File | Purpose |
|------|---------|
| `gui/.env` | `NODES_DIR`, `HOST_PROJECT_PATH` (paths), plus chat keys: `OPENAI_API_KEY` (browser chat), `ANTHROPIC_API_KEY` (notebook Claude agent), `JUPYTERHUB_API_TOKEN` |
| `gui/workflow_backend/.env` | DB, Keycloak, Django secret, paths, optional `OPENAI_API_KEY` |
| `gui/workflow_frontend/.env` | `VITE_API_BASE_URL`, Keycloak, paths |

Copy from the corresponding `env.template` in each folder and set:

- **gui/.env**: `NODES_DIR` = path to `gui/workflow_backend/django-project/codes/nodes` on your host; `HOST_PROJECT_PATH` = path to `gui/workflow_backend/django-project`.
- **workflow_backend/.env**: Set `KEYCLOAK_URL`, `KEYCLOAK_REALM`, `KEYCLOAK_CLIENT_ID` to match the Keycloak service in `docker-compose.yml`; set `HOST_PROJECT_PATH` and other `*_PATH` to your repo’s `gui/workflow_backend/django-project` (and subdirs).
- **workflow_frontend/.env**: use relative service prefixes for app traffic:
  `VITE_API_BASE_URL=/api`, `VITE_JUPYTER_BASE_URL=/jupyter`, and
  `VITE_MCP_BASE_URL=/mcp`. For local development, set
  `VITE_KEYCLOAK_URL=http://localhost:8080/auth` so browser redirects use the
  host-reachable Keycloak URL. Production builds override this value to `/auth`
  behind nginx.

The `.env` files are gitignored, so pulling a branch that changes an
`env.template` does not update an existing local `.env`. Compare your local
files with the templates after pulling configuration changes.

## 2. Create directory for nodes mount (if missing)

The backend container mounts your host’s nodes directory. Ensure it exists:

```bash
cd /Users/kirill/Documents/digital_brain/neuro-workflow
mkdir -p gui/workflow_backend/django-project/codes/nodes
```

(If `gui/.env` uses a different `NODES_DIR`, create that path instead.)

## 3. (Optional) NEST JupyterLab image for JupyterHub

Only needed if you use JupyterHub (port 8000). From repo root:

```bash
cd gui/workflow_backend/django-project/neuroworkflow
docker build --platform linux/amd64 -t nest-jupyterlab -f Dockerfile.nest .
```

You can skip this and still run the Workflow UI; JupyterHub will fail to spawn servers until this image exists.

> This image also bundles the in-notebook Claude agent dependencies (Node.js + the `claude` CLI + `claude-agent-sdk`). The notebook chat agent additionally needs `ANTHROPIC_API_KEY` set in `gui/.env`. See `docs/NOTEBOOK_CHAT_AGENT.md`.

## 3b. Dataset catalog service (bm_mindsdb)

The `mdb` service backs the `MDB*` database nodes, the `/api/catalog/` endpoints, and the **Dataset Catalog** tab. By default it is built from a **sibling clone** of the bm_mindsdb repo:

```bash
cd ..                       # next to neuro-workflow/
git clone https://github.com/oist/bm_mindsdb
```

If you keep it elsewhere, or already have an image, set either in `gui/.env`:

```bash
MDB_CONTEXT=../../bm_mindsdb   # build context (default)
MDB_IMAGE=mdb:latest           # use a prebuilt image instead of building
```

Its port is **not published to the host** on purpose: mdb has no authentication of its own and exposes an arbitrary-SQL endpoint, so it is reachable only from inside the compose networks and through the authenticated `/api/catalog/` proxy. Uncomment the `ports:` block in the `mdb` service only for local debugging.

The catalog starts empty. Populate it once the stack is up (needs internet; each source succeeds or fails independently):

```bash
docker compose exec backend curl -sX POST http://mdb:8004/api/sync_apis
```

You can skip this service entirely — the four live per-source database nodes (`DANDIQueryNode` and friends) do not use it, and `/api/catalog/` simply reports `503 {"available": false}` when `MDB_BASE_URL` is unset.

## 4. Build and start all services

From the **gui** directory:

```bash
cd /Users/kirill/Documents/digital_brain/neuro-workflow/gui
docker compose build
docker compose up
```

(Use `docker-compose` if your Docker install only provides the hyphenated command.)

The backend runs committed Django migrations on startup. If you change Django
models, run `python django-project/manage.py makemigrations` deliberately,
review the generated migration files, and commit them with the model change.

First run can take several minutes (building backend and frontend images). When you see the frontend and backend ready:

- **Workflow UI**: http://localhost:5173  
- **Django API**: http://localhost:3000  
- **JupyterHub** (if image built): http://localhost:8000/jupyter/
- **MCP proxy**: http://localhost:8001  

## 5. Routing smoke checks

Run these checks after changing the frontend service URLs, Keycloak settings,
or JupyterHub base path:

1. Open http://localhost:5173 and log in with Keycloak.
2. Confirm Keycloak browser requests use `http://localhost:8080/auth` in local
   development, not the internal Docker hostname.
3. Confirm app requests use the frontend origin with relative prefixes:
   `/api`, `/jupyter`, and `/mcp`.
4. Open the user profile page. It should render identity fields from the token
   and the account-management link should open Keycloak.
5. Open the custom database manager and test list/create/edit/test/delete
   operations while authenticated.
6. Click the Jupyter button from a workflow node and confirm JupyterLab opens
   the expected project/file under `/jupyter/` without a 404.
7. Drop a Database node on the canvas and click its catalog button. The
   **Dataset Catalog** tab should render the mdb console, and in the browser's
   Network tab its requests must go to `/mdb/api/...` — never bare `/api/...`,
   which would hit the Django API instead.

For the RIKEN production overlay, also confirm that published service ports are
bound to localhost, as required by the server firewall policy:

```bash
cd gui
docker compose -f docker-compose.yml -f docker-compose.prod.yml config
```

## 6. Stop

In the same terminal where you ran `docker compose up`, press **Ctrl+C**. To remove containers and volumes:

```bash
cd gui
docker compose down
# optional: docker compose down -v   # deletes DB volume
```

## Summary

| Step | Command (from repo root) |
|------|---------------------------|
| 1 | Ensure `gui/.env`, `gui/workflow_backend/.env`, `gui/workflow_frontend/.env` exist (they do; edit if paths differ). |
| 2 | `mkdir -p gui/workflow_backend/django-project/codes/nodes` |
| 3 | (Optional) Build NEST image: `cd gui/workflow_backend/django-project/neuroworkflow && docker build --platform linux/amd64 -t nest-jupyterlab -f Dockerfile.nest .` |
| 4 | `cd gui && docker compose build && docker compose up` |
| 5 | Open http://localhost:5173 |

**This Docker stack is only for neuro-workflow (Workflow UI at 5173).** BrainScaler (port 5001) is a separate app: run it with **Docker** (full stack) from `brainscaler/brainscaler_frontend` (`docker compose build && docker compose up` there) or with **conda** (`conda activate neuro` then `python aifront.py` there). See `brainscaler/README.md`. The “Workflow” link in BrainScaler points to http://localhost:5173, which is this Docker frontend.
