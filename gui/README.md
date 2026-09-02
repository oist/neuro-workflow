# Neuro-Workflow web application

Local install of the Docker Compose stack: React UI, Django API, PostgreSQL, JupyterHub, MCP, and Keycloak. This is the file the root [README](../README.md) **Installation → Web application** section links to.

Do **not** follow [`workflow_backend/README.md`](workflow_backend/README.md) for this stack — that file describes a separate compose tree. Always run Compose from `gui/`.

## Prerequisites

- Git
- Docker with Compose v2
- A clone of [oist/neuro-workflow](https://github.com/oist/neuro-workflow)

```bash
git clone https://github.com/oist/neuro-workflow.git
cd neuro-workflow
```

## Environment files

Copy the three templates. Fill in values; **do not commit** the resulting `.env` files or paste real API keys into git.

| File | Template | Role |
|---|---|---|
| `gui/.env` | [`gui/env.template`](env.template) | Compose interpolation (`NODES_DIR`, `HOST_PROJECT_PATH`, `OPENAI_*`, `ANTHROPIC_*`, `JUPYTERHUB_API_TOKEN`) |
| `gui/workflow_backend/.env` | [`gui/workflow_backend/env.template`](workflow_backend/env.template) | Django, Postgres, Keycloak |
| `gui/workflow_frontend/.env` | [`gui/workflow_frontend/env.template`](workflow_frontend/env.template) | Vite + browser Keycloak URL |

```bash
cp gui/env.template gui/.env
cp gui/workflow_backend/env.template gui/workflow_backend/.env
cp gui/workflow_frontend/env.template gui/workflow_frontend/.env
```

Then edit:

- **`HOST_PROJECT_PATH`** — absolute path to `gui/workflow_backend/django-project` on **this** machine.
- **`NODES_DIR`** — path Compose mounts as the node library (template default: `../src/neuroworkflow/nodes`).
- **`DJANGO_SECRET_KEY`**, **`DB_PASSWORD`**, **`JUPYTERHUB_API_TOKEN`** — replace template placeholders for anything beyond a throwaway laptop.
- **`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`** — only if you use the in-app or notebook agents. Leave the placeholders unused rather than committing real keys.
- **Keycloak** — defaults in the templates (`KEYCLOAK_REALM=neuroworkflow`, `KEYCLOAK_CLIENT_ID=neuroworkflow-app`, `VITE_KEYCLOAK_URL=http://localhost:8080/auth`) are correct for this local compose file.

### `BIND_HOST` (port publish)

Compose publishes service ports as `${BIND_HOST:-127.0.0.1}:<port>`. Set this in **`gui/.env`** (Compose reads `gui/.env` for interpolation):

- **Local laptop, browser on the same machine:** leave unset (binds `127.0.0.1`).
- **Local laptop, browser on another host on your LAN:** `BIND_HOST=0.0.0.0`.
- **Production behind nginx:** leave the default `127.0.0.1`. Publishing Docker ports on `0.0.0.0` on a public app server bypasses the host firewall. Do not do that.

`BIND_HOST` is documented in [`workflow_backend/env.template`](workflow_backend/env.template); Compose interpolates the value from `gui/.env`.

## NEST Jupyter image

Build the NEST JupyterLab image used by JupyterHub:

```bash
cd gui/workflow_backend/django-project/neuroworkflow
docker build -t nest-jupyterlab -f Dockerfile.nest .
cd ../../..
```

## Start the stack

```bash
cd gui
docker compose build
docker compose up
```

(`docker-compose` as a hyphenated command also works if that is what you have installed.)

## URLs (local)

With the default bind, open these on the same machine:

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:3000 |
| JupyterHub | http://localhost:8000 |
| MCP | http://localhost:8001 |
| Keycloak | http://localhost:8080 |

Log in through Keycloak (realm `neuroworkflow`). The frontend uses `onLoad: login-required`. Create a user in the Keycloak admin console if the realm has none.

After login:

- **Nodes → Node catalog** — glossary of workflow node types (name, category, ports, schema description).
- **Nodes → Upload** — register a new `.py` node type.
- If a deployment also shows a **Catalog** header link, that is a **dataset** browser (DANDI / CBS / …), not the node glossary.

How to add a node (GUI upload, Python library, in-app agent, Claude skill): root README section **[Add a node (manual / agent)](../README.md#add-a-node-manual--agent)**.

## Production reverse proxy

This file is the **local** Docker install. A public deployment should keep Docker ports on localhost and terminate TLS on a host reverse proxy (SSH / HTTP / HTTPS only). Do not copy a production nginx site into git from this README.
