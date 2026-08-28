# Catalog search (mdb Stage 2)

NeuroWorkflow users search public neuroscience dataset catalogs (DANDI, CBS,
Brain/MINDS, BMB human, SRPBS_TS/`aws`) **inside the app**. The browser never
talks to mdb. Introduced as Stage 2 of the mdb ↔ NeuroWorkflow contract.

Companion (ops, tokens, mdb paths): `deployment/MDB_CLIENT_CONTRACT.md` on the
control/docs tree. This file is the NeuroWorkflow implementation.

## Architecture

```
User browser  →  NeuroWorkflow (Keycloak, nginx 80/443)
                      ↓  same-origin /api/catalog/*  (user JWT only)
              NeuroWorkflow backend  (MDB_API_TOKEN)
                      ↓  Authorization: Bearer <search token>
              mdb-mindsdb:8004   (private Docker network — stage 1)
```

- No nginx `/mdb`, no public 8004, no iframe of the mdb dashboard.
- `MDB_*` lives only in `gui/workflow_backend/.env` (never `VITE_*`).
- Catalog is not copied into Postgres.
- Not a generic reverse proxy: user input never becomes a URL or mdb path.

## Routes

| NW (Keycloak required) | mdb |
|---|---|
| `GET  /api/catalog/statistics/` | `GET /api/api_statistics` |
| `POST /api/catalog/search/` | `POST /api/catalog_search` (`mode` forced to `keyword`) |
| `GET  /api/catalog/lookup/` | `GET /api/catalog_lookup` (`table=api_datasets` only) |
| `GET  /api/catalog/datasets/` | `GET /api/api_datasets` |

Trailing slashes match Django `APPEND_SLASH`. POST search must use the slash
or the body is dropped on redirect.

**Not implemented (Stage 2):** catalog chat, agent/intelligent search, sync,
`MDB_ADMIN_TOKEN`, `/api/local_catalog/…` BIDS paths.

Allowed `source` values: `dandi`, `cbs`, `brainminds`, `bmb_human`, `aws`
(UI label **SRPBS_TS**). Lookup `table` must be `api_datasets`. Search `limit`
is clamped 1–200 (default 50).

## Error codes

JSON body: `{ "status": "error", "code": "<code>", "error": "<message>" }`.

| code | HTTP | When |
|---|---|---|
| `catalog_unconfigured` | 503 | `MDB_BASE_URL` or `MDB_API_TOKEN` missing |
| `catalog_unavailable` | 503 / 502 | mdb down, timeout, or mdb 503/5xx |
| `catalog_auth` | 502 | mdb 401/403 (generic message; no token text) |
| `catalog_not_found` | 404 | mdb 404 |
| `invalid_query` / `invalid_mode` / `invalid_source` / `invalid_table` / `invalid_id` / `invalid_limit` | 400 | bad client input |

Unauthenticated callers get 401 from Keycloak/DRF.

## Environment

`gui/workflow_backend/env.template`:

```
MDB_BASE_URL=http://mdb-mindsdb:8004
MDB_API_TOKEN=
MDB_TIMEOUT=15
```

`MDB_ADMIN_TOKEN` is unused in NeuroWorkflow Stage 2 (sync stays SSH-only).

`get_mdb_config()` reads process env first, then the bind-mounted backend
`.env` via python-dotenv (`override=False`) so tokens can be added later and
picked up with gunicorn SIGHUP **without** recreating the backend (ssh-agent).
Never log token values.

## UI

Header **Catalog** → `/catalog`.

- Stats strip and source chips from `source_counts`
- Empty `q` → browse `GET /datasets/`; non-empty → `POST /search/`
- Bookmarkable `?q=&source=&limit=`
- Row click → lookup drawer: DOI / paper / DANDI links, copy id, copy JSON
- States: loading, empty, unconfigured, unavailable, auth, not found

The frontend calls only `/api/catalog/*` with `createAuthHeaders()`.

## MCP

Four tools on the workflow MCP server forward the **user JWT** to Django
(same pattern as `list_projects`). They are keyword catalog only — not the
MindsDB agent:

- `catalog_statistics`
- `catalog_search(query, source?, limit?)`
- `catalog_lookup(source, id)`
- `catalog_datasets(source?, limit?)`

## Tests

```bash
cd gui/workflow_backend
poetry run pytest django-project/tests/test_catalog.py -q

cd gui/workflow_frontend
pnpm test   # or npm test — vitest catalogApi
pnpm exec tsc -b
```

## Deploy (not part of the Stage 2 PR merge)

Do this only after mdb stage 1 (mdb joins `neuro-workflow_workflow`) and an
explicit ops OK:

1. From the **backend container**: `curl -sS -o /dev/null -w '%{http_code}\n' http://mdb-mindsdb:8004/` → 200
2. Set `MDB_BASE_URL` and `MDB_API_TOKEN` in `gui/workflow_backend/.env` (never `VITE_*`)
3. Prefer gunicorn SIGHUP so ssh-agent is preserved; recreate backend only if env is not bind-mounted
4. Rebuild/recreate **frontend only** (`--no-deps`) so `/catalog` is in the prod bundle
5. Do not publish 8004, do not add nginx `/mdb`

Host `127.0.0.1:8004` remains for operators (SSH tunnel).
