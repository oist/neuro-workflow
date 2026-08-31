# mdb client contract — NeuroWorkflow catalog

This is the HTTP contract NeuroWorkflow’s **backend** should call. The browser
must never talk to port 8004 and must never see mdb tokens.

Companion (NeuroWorkflow implementation + UI): `docs/CATALOG_SEARCH.md`.  
Companion (how to wire the two stacks): `deployment/DEPLOY_MDB_CATALOG.md`.

mdb is a separate repo (`oist/bm_mindsdb`). Do not put mdb source or secrets in
this repository.

---

## Architecture (fixed)

```
User browser  →  NeuroWorkflow (Keycloak, nginx 80/443)
                      ↓  same-origin /api/...  (no mdb token)
              NeuroWorkflow backend
                      ↓  Authorization: Bearer <MDB_API_TOKEN>
              mdb-mindsdb:8004   (Docker network neuro-workflow_workflow)
```

Only SSH / HTTP / HTTPS should be public; Docker `ports:` must bind
`127.0.0.1`. **Do not** add nginx `/mdb`, iframe the mdb dashboard, put tokens
in `VITE_*`, or publish `0.0.0.0:8004`.

`MDB_BASE_URL=http://172.17.0.1:8004` **does not work** while 8004 is
loopback-only, and it is **not** a public bind. Use a shared Docker network and
`http://mdb-mindsdb:8004`. Keep host `127.0.0.1:8004` for operators (SSH tunnel).

mdb has no Keycloak. **Catalog admin for NeuroWorkflow: none.** Sync stays
SSH-only (`MDB_ADMIN_TOKEN` is not used by NW). Daily harvest is
`MDB_AUTO_SYNC_INTERVAL` inside mdb.

---

## Tokens (backend `.env` only)

| Variable | Who may send it | Used for |
|---|---|---|
| `MDB_API_TOKEN` | NW backend only (`gui/workflow_backend/.env`). Never `VITE_*`, never frontend env, never `gui/.env`. Never commit the value. | Stage 2 search / statistics / lookup |
| `MDB_ADMIN_TOKEN` | Operators (SSH mdb UI / curl). **Not** used by NeuroWorkflow. Never commit. | Sync, ingest, NWB, MindsDB process |

Header: `Authorization: Bearer <token>`.  
Mutating POST/PUT/PATCH also need `Content-Type: application/json` (415
otherwise). Empty body → send `{}`.

**Always send `MDB_API_TOKEN` from the proxy**, even on GETs that mdb currently
leaves ungated (`/api/api_statistics`, keyword `GET /api/catalog_search`). Do
not depend on ungated holes.

---

## HTTP errors from mdb

| Code | When |
|---|---|
| **401** | Token required but missing, or token unset on mdb |
| **403** | Wrong token (search token on an admin route, or vice versa) |
| **400** | Bad input (`query`/`id` missing, unsupported `mode`/`table`) |
| **404** | Dataset / conversation not found |
| **410** | Retired: `/api/execute_sql`, `/api/predict_*`, … |
| **415** | Mutating POST without `Content-Type: application/json` |
| **500** | Generic `{ "status": "error", "error": "Request failed" }` |
| **503** | Orchestrator not ready, or agent search “not ready” (MindsDB down) |

Auth error JSON: `{ "status": "error", "error": "<message>" }`.

---

## Stage 2 — proxy these (search / catalog UI)

From the NW backend container, `$MDB_BASE_URL` is `http://mdb-mindsdb:8004`.
Operators on the host use `http://127.0.0.1:8004`.

Catalog sources (internal keys): `dandi`, `cbs`, `brainminds`, `bmb_human`, plus
local `aws` (display `SRPBS_TS`). Records include `source` and `source_display`.

### Health / counts

```bash
curl -sS -H "Authorization: Bearer $MDB_API_TOKEN" \
  "$MDB_BASE_URL/api/api_statistics"
```

Optional: `GET /api/status` (counts + whether MindsDB is up).

### Keyword search (no MindsDB)

```bash
curl -sS -X POST "$MDB_BASE_URL/api/catalog_search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MDB_API_TOKEN" \
  -d '{"query":"mouse","mode":"keyword","limit":20}'
```

Optional `source` (e.g. `"dandi"`). `limit` clamped 1–200 (default 50).

Also: `GET $MDB_BASE_URL/api/catalog_search?q=mouse&mode=keyword&limit=20`.

Browse without a query: `GET /api/api_datasets?source=dandi&limit=20`.

### Record lookup

```bash
curl -sS -H "Authorization: Bearer $MDB_API_TOKEN" \
  "$MDB_BASE_URL/api/catalog_lookup?table=api_datasets&source=dandi&id=000015"
```

NeuroWorkflow Stage 2 allows **`table=api_datasets` only**.

---

## Present on mdb, not proxied in Stage 2

Agent / intelligent search and catalog agent chat exist on mdb (`mode=agent`,
`mode=intelligent`, `/api/mindsdb_agent/chat`). They need MindsDB + OpenAI on
the mdb side. NeuroWorkflow does **not** expose them in this PR.

Admin routes require `MDB_ADMIN_TOKEN` (search token → 403). Do not proxy these
to ordinary Keycloak users:

| Method | Path |
|---|---|
| POST | `/api/sync_apis` |
| POST | `/api/ingest_local_catalog` |
| POST | `/api/process_nwb_file` |
| POST | `/api/start_mindsdb_server` |
| POST | `/api/stop_mindsdb_server` |
| POST | `/api/connect_mindsdb_datasource` |
| POST | `/api/mindsdb_agent/setup` |
| POST | `/api/mindsdb_agent/drop` |

Never proxy `/api/execute_sql` or retired predict routes (410). Never call
MindsDB on 47334 (unpublished, in-container only).

---

## Frozen Stage 2 URL layout

Keycloak-gated. Forward JSON as-is. Do not copy the catalog into Postgres.

| NW (Keycloak) | mdb |
|---|---|
| `GET  /api/catalog/statistics/` | `GET /api/api_statistics` |
| `POST /api/catalog/search/` | `POST /api/catalog_search` (**`mode=keyword` only**) |
| `GET  /api/catalog/lookup/` | `GET /api/catalog_lookup` |
| `GET  /api/catalog/datasets/` | `GET /api/api_datasets` |

**Not in this PR:** `/api/catalog/chat`, `/api/catalog/sync`, agent/intelligent
modes, `GET /api/local_catalog/…`.

---

## Frozen answers

1. **URL layout** — table above.
2. **Catalog admin** — **none**. No Sync button in NeuroWorkflow.
3. **Docker network** — mdb joins existing **`neuro-workflow_workflow`** as
   hostname `mdb-mindsdb`. `MDB_BASE_URL=http://mdb-mindsdb:8004`. Keep host
   `127.0.0.1:8004`. Attach **mdb**, not the NW backend (avoids gunicorn
   recreate / Slurm ssh-agent drop).
4. **Stage 2 scope** — keyword + statistics + lookup + browse only.
5. **`MDB_*`** — `gui/workflow_backend/.env` only; values are never committed.
