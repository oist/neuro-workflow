# OAI-PMH Harvest Nodes

Fetch research-data records — and the data files they reference — from an
[OAI-PMH](https://www.openarchives.org/pmh/) repository into a workflow. The
first target is the RIKEN CBS **MDRS Data Repository**
(`https://neurodata.riken.jp/api/oai/`).

## How it works

```
[kernel]  OAIPMHHarvestNode ──records──> OAIPMHDownloadNode ──> <results_path>/oai_pmh/<folder>/<file>
             │  neuroworkflow.utils.oai_pmh (stdlib only)
             │  X-Api-Key: NEUROWORKFLOW_SERVICE_TOKEN
             ▼
[backend] GET /api/harvest/oai/?verb=…                 allowlisted verbs/args, key attached, XML relayed as-is
          GET /api/harvest/oai/files/<uuid>/download/  streamed file relay
             │  OAI_PMH_BASE_URL / OAI_PMH_API_KEY / OAI_PMH_API_KEY_HEADER / OAI_PMH_FILE_DOWNLOAD_URL
             ▼
[repository] https://neurodata.riken.jp/api/oai/ , /api/v3/files/{id}/download/
```

Workflow nodes run inside JupyterHub kernels, not in Django, so the repository
address and API key are **never sent to the kernel**. The nodes call the backend
proxy (`gui/workflow_backend/django-project/app/harvest/`) with the shared
service token that every kernel already receives (`NEUROWORKFLOW_SERVICE_TOKEN`
in `jupyterhub_config.py`); the backend validates the request and adds the key.

## Configuration

Add to `gui/workflow_backend/.env` (template: `gui/workflow_backend/env.template`)
and recreate the backend container (`docker compose up -d backend`):

```
OAI_PMH_BASE_URL=https://neurodata.riken.jp/api/oai/
OAI_PMH_API_KEY=<key>
OAI_PMH_API_KEY_HEADER=X-MDRS-API-Key
OAI_PMH_FILE_DOWNLOAD_URL=https://neurodata.riken.jp/api/v3/files/{file_id}/download/
OAI_PMH_TIMEOUT=60
```

These variables are backend-only. Do not add them to `gui/.env` (it is loaded
into the JupyterHub container) or as `VITE_*` (bundled into the browser app).

## Security posture

- The proxy is **not a generic reverse proxy**: only the verbs `Identify`,
  `ListMetadataFormats`, `ListSets`, `ListIdentifiers`, `ListRecords`,
  `GetRecord` and the arguments `metadataPrefix`, `set`, `from`, `until`,
  `identifier`, `resumptionToken` are accepted (anything else → 400); the
  upstream URL is built only from the configured base URL and those arguments.
  File downloads accept a UUID only.
- The key never leaves the backend; it appears in no response, log, or kernel
  environment. The nodes have **no address/key parameters**, so users cannot
  redirect the harvest or change credentials from the GUI.
- The service token is shared by all kernels (same posture as the Anthropic
  proxy), so every kernel user can harvest and download through the configured
  key.
- Upstream bodies are relayed unchanged. OAI-PMH reports protocol errors with
  HTTP 200 (`<error errorCode="badAuthentication">…`), and the client detects
  them from the XML.

## Nodes (`database` category)

### `OAIPMHHarvestNode` — `ListRecords`

| Parameter | Default | Meaning |
|---|---|---|
| `metadata_prefix` | `mdrs` | `mdrs` includes folder details and file ids (needed for downloads); `oai_dc` is title/publisher/date only |
| `set_spec` | `""` | OAI set, e.g. `dataset`, `public`, `project:bm2.0` |
| `from_date` / `until_date` | `""` | UTC bounds, `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ssZ` |
| `max_records` | `100` | Cap; pages are followed via `resumptionToken` |
| `timeout` | `30` | Seconds per HTTP request |

Outputs: `records` (LIST of dicts: `identifier`, `datestamp`, `set_specs`,
`deleted`, `metadata_prefix`, `metadata`, `files [{id, name, mime_type, size}]`)
and `metadata` (DICT: `status`, `count`, `total`, `error`, `error_code`).
For `mdrs`, `metadata` is the folder dict (`name`, `path`, `size`, `parent`,
the decoded `metadata` JSON array, `files`); for `oai_dc` it is
`{field: [values]}`. The node never raises: failures land in
`metadata.error` / `error_code`.

### `OAIPMHDownloadNode` — fetch data files

Input `records` from the harvest node. Files are written to
`<results_path>/<subdir>/<folder name or id>/<file name>` (names are reduced to
`[A-Za-z0-9._-]`, so no path traversal). Records without a `files` list (e.g.
`oai_dc`) are resolved with `GetRecord` in the `mdrs` format first.

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

## Limitations

- Downloads stream through the backend, occupying one request thread per
  file for the whole transfer; very large datasets or many parallel downloads
  will slow the API. No range/resume support.
- The download route expects UUID file ids (RIKEN MDRS). Other repositories
  would need a different `OAI_PMH_FILE_DOWNLOAD_URL` and id pattern.
- `metadata_prefix` values are repository-specific; use `ListMetadataFormats`
  (`curl -H "X-Api-Key: $JUPYTERHUB_API_TOKEN" "http://backend:3000/api/harvest/oai/?verb=ListMetadataFormats"`
  from inside the stack) to see what a repository offers.

## Files

- Backend proxy: `gui/workflow_backend/django-project/app/harvest/` (tests in `tests/test_oai_pmh_proxy.py`)
- Client: `src/neuroworkflow/utils/oai_pmh.py` (kernel copy: `gui/workflow_backend/django-project/codes/neuroworkflow/utils/oai_pmh.py`)
- Nodes: `src/neuroworkflow/nodes/database/` (kernel copy: `gui/workflow_backend/django-project/codes/nodes/database/`)
- Offline tests: `tests/test_oai_pmh_client.py`
