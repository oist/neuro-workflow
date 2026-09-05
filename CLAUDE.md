# CLAUDE.md

## Project Overview

NeuroWorkflow is a Python library and web application for building and executing neural simulation workflows using a node-based system. Developed by the Neural Computation Unit (Doya Lab) at OIST, supported by Brain/MINDS 2.0.

## Repository Structure

```
neuro-workflow/
├── src/neuroworkflow/       # Core Python library (pip-installable)
│   ├── core/                # Node, Workflow, WorkflowBuilder, Schema, Port
│   ├── nodes/               # Built-in node collection (analysis, io, network, optimization, simulation, stimulus)
│   ├── cli/                 # CLI entry point
│   └── utils/
├── gui/                     # Web application (Docker Compose)
│   ├── workflow_frontend/   # React 19 + TypeScript + Vite
│   ├── workflow_backend/    # Django 5 + DRF + PostgreSQL
│   ├── mcp_server/          # MCP proxy server
│   └── docker-compose.yml
├── examples/                # Python API usage examples
├── notebooks/               # Jupyter notebook tutorials
├── data/                    # Sample data files
├── NODE_SCHEMA.md           # Node definition specification
├── CUSTOM_NODE_TUTORIAL.md  # Custom node creation guide
└── CustomNodeTemplate.py    # Template for new nodes
```

## Tech Stack

### Core Library (`src/neuroworkflow/`)
- Python 3.8+, NumPy
- Key classes: `Node`, `Workflow`, `WorkflowBuilder`, `PortType`, `PortDefinition`, `ParameterDefinition`
- Package config: `pyproject.toml` (setuptools)

### Frontend (`gui/workflow_frontend/`)
- React 19, TypeScript ~5.7, Vite 6
- @xyflow/react (React Flow) for node-based UI
- Chakra UI 2.8 for components
- Keycloak (OIDC) for auth via `keycloak-js`
- pnpm for package management
- Dev server: port 5173
- Path alias: `@` → `src/`
- API proxy: `/api` → `http://localhost:3000`

### Backend (`gui/workflow_backend/`)
- Django 5.0, Django REST Framework 3.14
- PostgreSQL 16
- Poetry for dependency management
- Python 3.11.11
- Dev server: port 3000
- JupyterHub for workflow code execution (port 8000)
- MCP server for AI tool integration (port 8001)

## Development Commands

### Web Application (Docker)
```bash
cd gui
docker-compose build
docker-compose up
# Frontend: http://localhost:5173
# Backend API: http://localhost:3000/api
# JupyterHub: http://localhost:8000
```

### Backend (local)
```bash
cd gui/workflow_backend
poetry install
poetry run python django-project/manage.py migrate
poetry run python django-project/manage.py runserver 0.0.0.0:3000
```

### Frontend (local)
```bash
cd gui/workflow_frontend
pnpm install
pnpm run dev          # local development
pnpm run dev:docker   # Docker environment
pnpm run build        # production build
```

### Core Library
```bash
pip install -e ".[dev]"          # editable install with dev tools
pip install -e ".[nest]"         # with NEST simulator support
pip install -e ".[visualization]" # with matplotlib/seaborn
```

### Testing & Linting
```bash
# Backend
cd gui/workflow_backend
poetry run pytest
poetry run black .
poetry run isort .
poetry run flake8

# Core library
pytest
black --line-length 88 src/
isort --profile black src/
```

## Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/api/workflow/` | List/create workflow projects |
| GET/PUT/DELETE | `/api/workflow/{id}/` | CRUD on a project |
| GET/PUT | `/api/workflow/{id}/flow/` | Get/save flow data (nodes + edges) |
| GET/POST | `/api/workflow/{id}/nodes/` | List/create nodes |
| GET/POST | `/api/workflow/{id}/edges/` | List/create edges |
| POST | `/api/workflow/{id}/generate-code/` | Generate Python code from workflow |
| POST | `/api/workflow/{id}/run/` | Execute workflow (streaming) |
| GET/POST | `/api/chat/profiles/` | List/create the user's chat profiles (MCP tool allowlist + prompt override) |

## Environment Variables

Three `.env` files are needed for the web application:

1. **`gui/.env`** — Docker Compose level (NODES_DIR, OPENAI_API_KEY, ANTHROPIC_API_KEY, JUPYTERHUB_API_TOKEN)
2. **`gui/workflow_backend/.env`** — Django (DB_*, KEYCLOAK_*, DJANGO_SECRET_KEY, paths)
3. **`gui/workflow_frontend/.env`** — Vite (VITE_API_BASE_URL, VITE_KEYCLOAK_*)

Template: `gui/workflow_backend/env.template`

## Architecture Notes

- Nodes in `src/neuroworkflow/nodes/` must be synced to `gui/workflow_backend/django-project/codes/nodes/` for the web app. Docker Compose mounts `NODES_DIR` to handle this.
- Core library code in `src/neuroworkflow/core/` is also synced to `gui/workflow_backend/django-project/codes/neuroworkflow/core/`.
- Workflow execution uses JupyterHub's kernel WebSocket API — code is generated from the node graph and sent to a Jupyter kernel for execution.
- Authentication is handled by Keycloak (OIDC). The frontend uses `keycloak-js` (`onLoad: "login-required"`); the backend verifies access tokens via the realm's JWKS endpoint in `app/auth/authentication.py:KeycloakAuthentication`.
- The **browser chat** feature uses the OpenAI API with Function Calling and MCP integration. Per-user **Chat Profiles** restrict which MCP tools it may use and can override the system prompt (see `docs/CHAT_PROFILES.md`).
- The **in-notebook chat agent** (`src/neuroworkflow/agent/`, synced to `codes/neuroworkflow/agent/`) uses the **Claude Agent SDK** running in the Jupyter kernel. It reaches Anthropic through the backend `/api/chat/anthropic` proxy (`ANTHROPIC_BASE_URL`), so the API key stays on the backend; workflow tools still go through the MCP proxies with the user's Keycloak token. The `claude` CLI + `claude-agent-sdk` are bundled in the nest kernel image (`Dockerfile.nest`). See `docs/NOTEBOOK_CHAT_AGENT.md`.

## Code Style

- Python: black (line-length 88), isort (profile: black)
- TypeScript: ESLint
- Django: flake8
- Commit messages: English, concise description of changes

## GitHub

- Repository: `oist/neuro-workflow`
- Main branch: `main`
- PR workflow: feature branches → PR → merge to main

---

## Behavioral Guidelines

Reduce common LLM coding mistakes. **Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
