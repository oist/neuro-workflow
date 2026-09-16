## Installation

Create the Docker image for the NEST JupyterLab environment

```bash
cd ./gui/workflow_backend/django-project/neuroworkflow
docker build -t nest-jupyterlab -f Dockerfile.nest .
```

Edit the .env files to set environment variables.
rename env.template to .env and set environment variables

**`gui/.env`** — Docker Compose level:
| Variable | Description |
|---|---|
| `NODES_DIR` | Path to the nodes directory (`./workflow_backend/django-project/codes/nodes`) |
| `HOST_PROJECT_PATH` | Absolute path to `gui/workflow_backend/django-project` on the host machine |

Add 2 more .env files based on the templates.

- ./gui/workflow_backend/.env
- ./gui/workflow_frontend/.env

Catalog search (optional): set `MDB_BASE_URL` and `MDB_API_TOKEN` in
`gui/workflow_backend/.env` only. See `docs/CATALOG_SEARCH.md` and
`deployment/DEPLOY_MDB_CATALOG.md`. Never put mdb tokens in `VITE_*` or
`gui/.env`.

Start the Docker containers using docker-compose

```bash
cd ./gui
docker-compose build
docker-compose up
```

Open your web browser

```
http://localhost:5173
```
