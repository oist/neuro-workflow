# Node Figure Display

Figures (matplotlib output) produced during a workflow run are attributed to the
canvas node that created them and shown **directly on that node** — a count
badge and a thumbnail, with a click-to-enlarge modal. Figures are also persisted
on the backend and restored when the project is reopened. Introduced in PR #87.

Before this feature, figures only appeared in the run log modal, in stream
order, with no indication of which node produced them, and vanished on reload.

## How it works

The whole workflow still runs as **one Jupyter cell**; nothing about execution
changes. Attribution happens on the backend by correlating two things the run
already emits, in order:

1. the `"Executing node: <var_name>"` line the core loop prints to stdout
   before each node, and
2. the `display_data` (image/png) messages matplotlib publishes.

```
Generate Code                          Run (SSE stream)
─────────────                          ────────────────
code_generation_service                Jupyter kernel ──▶ jupyter_execution_service
  ├─ workflow.py                          │  stdout "Executing node: instance_X_002"
  └─ node_map.json                        │  display_data (image/png)
     {var_name → canvas node id}          ▼
                                       NodeAttributor (run_attribution.py)
                                          ├─ parses markers, tracks current node
                                          ├─ tags image events: node_id, node_name,
                                          │    figure_index; emits node_executing
                                          ├─ tees PNGs → results/figures/<node>/
                                          └─ writes manifest.json (finally block)
                                          ▼
Frontend                               SSE: stdout / image / node_executing / done
────────
projectSelector ── routes events ──▶ runStore (Zustand, non-persisted)
calculationNode ── per-id selector ──▶ badge + thumbnail + executing spinner
nodeFiguresModal ── all figures enlarged
homeView ── on project open: fetch manifest via /api/viewer/ ──▶ restore
```

### node_map.json — why a generation-time sidecar

Generated variable names are normally the node's `instanceName`, **except** when
`instanceName === label`: then the variable becomes `instance_<Label>_<NNN>`,
numbered by the node array order *at generation time*. Since **Run does not
regenerate code**, the canvas can drift from the script that actually runs, and
re-deriving variable names on the frontend would silently break. Instead,
`generate_code_from_flow_data` writes a sidecar next to `workflow.py`:

```json
{ "version": 1, "var_to_node": { "instance_TVBVisualizationNode_002": "<react-flow-node-id>" } }
```

This is correct by construction for the script it was generated with. A missing
or corrupt map degrades gracefully: all images get `node_id: null` and land in
the unattributed bucket (still visible in the log modal as before).

### Marker parsing and ordering

`NodeAttributor.MARKER_RE` is anchored at line start (`^Executing node: ...`),
so error output like `"Error executing node: ..."` never matches. Stdout
arrives in arbitrary chunks, so a partial-line tail is carried between events.

Ordering relies on an ipykernel guarantee: **stdout is flushed before
display_data is published**, and iopub messages for a single cell arrive in
execution order. If that were ever violated, the worst case is a figure
attributed to the neighboring node — easy to spot because every image event
carries its claimed `node_id`.

### Node-boundary flush (core change)

Nodes that never call `plt.show()` (e.g. `SNNbuilder_Raster`) would otherwise
have their figures flushed by matplotlib-inline's post-execute hook at **cell
end**, misattributing them to the last node. The core execution loop therefore
calls `_flush_inline_figures()` right after each `node.process()` — a no-op
outside Jupyter/inline-matplotlib environments.

This lives in `src/neuroworkflow/core/workflow.py` (both `Workflow.execute` and
`WorkflowBuilder._execute_with_tracking`) and the synced copy under
`gui/workflow_backend/django-project/codes/neuroworkflow/core/workflow.py`; the
two files must stay byte-identical.

Caveat: `flush_figures` **closes** pyplot-registered figures. A downstream node
receiving a live `Figure` object on an OBJECT port can still save/draw it, but
can no longer re-show it via pyplot. No current node depends on this.

## SSE events

Two additions to the run stream — existing handlers are untouched:

```
event: node_executing
data: {"node_name": "instance_TVBVisualizationNode_002", "node_id": "calc_..." | null}

event: image
data: {"content": "<base64>", "mime": "image/png",
       "node_id": "calc_..." | null, "node_name": "..." | null, "figure_index": 0}
```

`figure_index` counts per node (unattributed figures share one counter).

## Persistence

While streaming, each image is also decoded and written to disk under the
project directory ("latest run only" — the directory is wiped at run start):

```
codes/projects/<project_id>/results/figures/
  ├── <sanitized_node_id>/fig_000.png
  ├── _unattributed/fig_000.png
  └── manifest.json   { version, finished_at, status: ok|error|aborted, figures: [...] }
```

`manifest.json` is written in the SSE generator's `finally` block, so it exists
after normal completion, errors, and client aborts alike. On project open, the
frontend fetches it via the existing **unauthenticated** file route
`GET /api/viewer/<project_id>/<subpath>` and rebuilds the per-node figure map.

> **Security note:** because `/api/viewer/` is intentionally unauthenticated,
> persisted figures are readable by anyone who knows the project UUID — the
> same exposure class as the brain-viewer data served from the same route.
> Tracked as a follow-up alongside issue #86 / project visibility.

## Frontend state

Run figures live in a dedicated non-persisted Zustand store,
`src/stores/runStore.ts` — deliberately **not** in `flowStore`, because node
data there enters the zundo undo history and the debounced node-persistence
PUT (which would push base64 into the DB). Details:

- `figuresByNode` is keyed by React Flow node id; `calculationNode` subscribes
  with per-id selectors, so during a run only nodes that actually receive
  figures re-render.
- `MAX_FIGURES_PER_NODE = 20` (newest kept) applies to both streamed
  (`addFigure`) and restored (`setAllFigures`) figures.
- Streamed figures are `data:` URIs; restored ones are `/api/viewer/` URLs.
- The manifest restore in `homeView` is guarded by a ref against project
  switches, so a slow response for a previously selected project can't
  overwrite the current one.
- `executingNodeId` (from `node_executing` events) drives a spinner in the
  node header while that node runs.

## Limitations

- Only `image/png` is handled — the kernel currently streams nothing else
  (no SVG/HTML/plotly).
- Latest-run semantics: starting a run deletes the previous run's figures.
- If a node is deleted after Generate Code and the workflow is run without
  regenerating, that variable's figures are unattributed (`node_id: null`).
- Interactive runs are not recorded as `WorkflowRun` records; this feature is
  independent of the async run management API.

## Files

| Layer | File |
|---|---|
| Backend | `app/workflow/code_generation_service.py` (writes `node_map.json`) |
| Backend | `app/workflow/run_attribution.py` (NodeAttributor, figure tee, manifest) |
| Backend | `app/workflow/views.py` (wires the attributor into the run stream) |
| Core | `src/neuroworkflow/core/workflow.py` + synced `codes/` copy (boundary flush) |
| Frontend | `src/stores/runStore.ts` (figure/executing state) |
| Frontend | `src/views/home/components/projectSelector.tsx` (SSE → store routing) |
| Frontend | `src/views/home/components/calculationNode.tsx` (badge/thumbnail/spinner) |
| Frontend | `src/views/home/components/nodeFiguresModal.tsx` (enlarged view) |
| Frontend | `src/views/home/homeView.tsx` (restore on project open) |
| Frontend | `src/api/workflowRunApi.ts` (`fetchRunFigureManifest`) |
| Tests | `django-project/tests/test_run_attribution.py` |

## Testing

Backend unit tests cover marker parsing (including chunk splits and
`"Error executing node:"` non-matches), attribution, per-node figure indexing,
disk tee + manifest, and degraded map loading. The backend image has no dev
dependencies, so run them in an ephemeral container:

```bash
cd gui && docker-compose run --rm --no-deps backend bash -c \
  "poetry install --no-root --only dev --quiet && \
   python -m pytest django-project/tests/test_run_attribution.py -q"
```
