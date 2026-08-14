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
| `JUPYTERHUB_ALLOWED_USERS` | Hub accounts: `internal,hackathon,user1` (see `deployment/JUPYTER_TWO_SPACES.md`) |

Add 2 more .env files based on the templates.

- ./gui/workflow_backend/.env
- ./gui/workflow_frontend/.env

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
