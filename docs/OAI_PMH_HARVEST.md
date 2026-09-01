# OAI-PMH Harvest & Search

Make research-data records from an [OAI-PMH](https://www.openarchives.org/pmh/)
repository available to workflows. The first target is the RIKEN CBS **MDRS
Data Repository** (`https://neurodata.riken.jp/api/oai/`).

OAI-PMH is a *harvesting* protocol — it enumerates records but has no search
verb. The design therefore splits into two components:

1. **Harvester (accumulate)** — the backend periodically runs
   `manage.py harvest_oai`, which walks `ListRecords` incrementally and upserts
   every record into PostgreSQL (`harvested_records`).
2. **Search (query)** — the GUI dataset search box and the workflow nodes read
   that local copy; nothing scans the upstream repository per request.

## How it works

```
[harvester] manage.py harvest_oai (looped by the compose `harvester` service)
               │  OAIPMHClient, direct mode (OAI_PMH_BASE_URL / OAI_PMH_API_KEY)
               ▼
[repository] https://neurodata.riken.jp/api/oai/  ──ListRecords──>  PostgreSQL
                                                    harvested_records / harvest_runs

[browser] GET /api/harvest/oai/search/?q=…            Keycloak auth, DB query
[kernel]  OAIPMHRecordsNode ──records──> OAIPMHDownloadNode ──> <results_path>/oai_pmh/<folder>/<file>
             │ GET /api/harvest/records/?identifiers=…   (service token, DB query)
             │ GET /api/harvest/oai/files/<uuid>/download/ (service token, streamed relay)
             ▼
[backend] app/harvest/  (repository key attached server-side for downloads)
```

Workflow nodes run inside JupyterHub kernels, not in Django, so the repository
address and API key are **never sent to the kernel**. Kernels authenticate to
the backend with the shared service token (`NEUROWORKFLOW_SERVICE_TOKEN` in
`jupyterhub_config.py`) and can only read harvested records by identifier or
stream one file by UUID.

### Harvester

`manage.py harvest_oai` performs one run:

- **Incremental by watermark**: the next run passes the highest datestamp
  observed by the last *successful* run as `from=`. Only completed runs advance
  the watermark, so a failed run can never skip records; the inclusive boundary
  record is re-fetched and absorbed by the upsert.
- **All-or-nothing**: on an upstream error nothing is stored, an error row is
  written to `harvest_runs`, and the command exits non-zero. The compose loop
  simply retries on the next interval.
- Deleted records (`<header status="deleted">`) flip the `deleted` flag but
  keep previously stored content. `--full` re-harvests everything and also
  marks records missing upstream as deleted (full resync).
- `harvest_runs` keeps the last 200 runs for inspection.
- When `OAI_PMH_BASE_URL` is unset the command exits 0 with a notice, so
  unconfigured deployments stay quiet.

Run it manually with `docker compose exec backend python3
django-project/manage.py harvest_oai` (add `--full` to resync). The
`harvester` compose service loops it every `OAI_PMH_HARVEST_INTERVAL` seconds
(default 900). **Run one harvester instance only** — the loop is sequential and
takes no locks.

### GUI dataset search

The node Config modal shows a **Dataset Search** section for
`OAIPMHRecordsNode`: type a keyword, tick the matching datasets, and *Apply
selection* writes their identifiers into the node's `identifiers` parameter.
It is served by `GET /api/harvest/oai/search/?q=&set=&limit=` — a
**browser-plane** endpoint authenticated with the user's Keycloak token, unlike
the kernel-plane routes (service token). Keep the two auth planes separate; the
service token must never be handed to browsers.

Search is a case-insensitive AND match of each term against a per-record
haystack (identifier, name, description, laboratory, path, file names) stored
in `harvested_records.search_text`. The response's `harvested_at` reports the
last successful harvest (shown in the UI); before the first harvest, search
returns an empty result with `harvested_at: null`.

## Configuration

Add to `gui/workflow_backend/.env` (template: `gui/workflow_backend/env.template`)
and recreate the containers (`docker compose up -d backend harvester`):

```
OAI_PMH_BASE_URL=https://neurodata.riken.jp/api/oai/
OAI_PMH_API_KEY=<key>
OAI_PMH_API_KEY_HEADER=X-MDRS-API-Key
OAI_PMH_FILE_DOWNLOAD_URL=https://neurodata.riken.jp/api/v3/files/{file_id}/download/
OAI_PMH_TIMEOUT=60
OAI_PMH_HARVEST_INTERVAL=900
OAI_PMH_HARVEST_TIMEOUT=300
```

`OAI_PMH_TIMEOUT` applies to the file download proxy; the harvester uses the
separate `OAI_PMH_HARVEST_TIMEOUT` because some MDRS `ListRecords` pages take
minutes to serialize server-side (see Limitations).

These variables are backend-only. Do not add them to `gui/.env` (it is loaded
into the JupyterHub container) or as `VITE_*` (bundled into the browser app).

## Security posture

- Kernels can no longer issue OAI-PMH verbs at all (the earlier allowlisted
  passthrough proxy was removed with the move to the local store): the kernel
  plane serves only harvested records by identifier and file downloads by UUID.
- The key never leaves the backend; it appears in no response, log, or kernel
  environment. The nodes have **no address/key parameters**, so users cannot
  redirect the harvest or change credentials from the GUI.
- The service token is shared by all kernels (same posture as the Anthropic
  proxy), so every kernel user can read harvested records and download files
  through the configured key.
- Only the harvester talks to the repository. OAI-PMH reports protocol errors
  with HTTP 200 (`<error errorCode="badAuthentication">…`), and the client
  detects them from the XML.

## Nodes (`database` category)

### `OAIPMHRecordsNode` — fetch selected records

| Parameter | Default | Meaning |
|---|---|---|
| `identifiers` | `""` | Comma/newline-separated OAI identifiers (filled by the GUI dataset search) |
| `timeout` | `30` | Seconds per HTTP request |

Outputs: `records` (LIST of dicts: `identifier`, `datestamp`, `set_specs`,
`deleted`, `metadata_prefix`, `metadata`, `files [{id, name, mime_type, size}]`)
and `metadata` (DICT: `status`, `count`, `total` = identifier count, `error`,
`error_code`). `metadata` on each record is the folder dict (`name`, `path`,
`size`, `parent`, the decoded `metadata` JSON array, `files`). Identifiers
missing from the harvested copy are reported as `error_code: "not_found"`
while the found records are still returned. The node never raises: failures
land in `metadata.error` / `error_code`.

### `OAIPMHDownloadNode` — fetch data files

Input `records` from the records node. Files are written to
`<results_path>/<subdir>/<folder name or id>/<file name>` (names are reduced to
`[A-Za-z0-9._-]`, so no path traversal). Records without a `files` list are
resolved from the backend's harvested store first.

| Parameter | Default | Meaning |
|---|---|---|
| `subdir` | `oai_pmh` | Sub-directory under the results path |
| `max_files_per_record` | `10` | `0` = all files |
| `timeout` | `60` | Seconds per HTTP request |
| `skip_existing` | `True` | Keep files already present |

Outputs: `file_paths` (LIST of paths present after the run) and
`download_metadata` (`status`, `downloaded`, `skipped`, `failed`,
`records_without_files`, `errors`).

## Direct mode (CLI / library use)

Outside the GUI, set `OAI_PMH_BASE_URL` (and optionally `OAI_PMH_API_KEY`,
`OAI_PMH_API_KEY_HEADER`, `OAI_PMH_FILE_DOWNLOAD_URL`) in the process
environment and the client calls the repository directly:

```python
from neuroworkflow.utils.oai_pmh import OAIPMHClient
env = OAIPMHClient().list_records("mdrs", set_spec="dataset", max_records=5)
print(env["status"], env["count"], env["total"])
```

To debug the upstream from inside the stack, curl it directly from the backend
container (the proxy no longer exists):

```
docker compose exec backend bash -c \
  'curl -H "$OAI_PMH_API_KEY_HEADER: $OAI_PMH_API_KEY" "${OAI_PMH_BASE_URL}?verb=ListMetadataFormats"'
```

## Limitations

- The MDRS repository holds several thousand records (60+ pages of 100), and
  pages covering some records (measured 2026-09: those updated 2025-03-31,
  folders with very large file lists) take 1–2 minutes each to serialize
  upstream. The client retries transport errors and 503s three times per
  request, and the harvester's per-request timeout defaults to 300 s, but the
  initial (and `--full`) harvest can still take on the order of 15 minutes.
  Incremental runs only fetch changed records and are fast.
- Search results are at most one harvest interval stale; the UI shows the last
  harvest time. Before the first harvest completes, search is empty.
- The harvester assumes ISO-8601 UTC datestamps whose lexicographic order is
  chronological (true for MDRS, which uses second granularity). A repository
  with day-granularity datestamps would make the watermark boundary coarser.
- Only the `mdrs` metadata format is harvested (the one carrying names,
  descriptions and file ids).
- Downloads stream through the backend, occupying one request thread per
  file for the whole transfer; very large datasets or many parallel downloads
  will slow the API. No range/resume support.
- The download route expects UUID file ids (RIKEN MDRS). Other repositories
  would need a different `OAI_PMH_FILE_DOWNLOAD_URL` and id pattern.
- Canvas nodes placed before the `identifiers` parameter existed carry an older
  schema copy; the Dataset Search section then shows a hint to re-place the
  node (the parameter update endpoint rejects keys missing from the copy).

## Files

- Backend (models, harvester, search, records API, download relay):
  `gui/workflow_backend/django-project/app/harvest/` (tests in
  `tests/test_harvest_command.py`, `tests/test_oai_pmh_search.py`,
  `tests/test_harvest_records_api.py`, `tests/test_oai_pmh_proxy.py`)
- Harvester service: `gui/docker-compose.yml` (`harvester`)
- GUI search section: `gui/workflow_frontend/src/views/home/components/OAIPMHRecordSearch.tsx` (wired in `nodeDetailModal.tsx`)
- Client: `src/neuroworkflow/utils/oai_pmh.py` (kernel copy: `gui/workflow_backend/django-project/codes/neuroworkflow/utils/oai_pmh.py`)
- Nodes: `src/neuroworkflow/nodes/database/` (kernel copy: `gui/workflow_backend/django-project/codes/nodes/database/`)
- Offline tests: `tests/test_oai_pmh_client.py`
