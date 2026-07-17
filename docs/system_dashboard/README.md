# NeuroWorkflow System Dashboard

A single self-contained Python script that builds an interactive HTML dashboard of
the whole NeuroWorkflow system: every workflow shown as a cluster of connected
node instances, plus a catalog of all available node types (used / unused).

It reads **only the filesystem** — no database, no server connection needed.

## Requirements

- Python 3.8+
- **No third-party packages** (standard library only: `argparse`, `json`, `re`,
  `pathlib`, `datetime`).

The graph rendering library is **vis-network**. By default the generated page
loads it from a CDN (works as long as the machine has internet). For a fully
offline dashboard, drop `vis-network.min.js` next to the script — if present it is
inlined into the HTML. (It is intentionally not committed to keep the repo light.)

## What it reads

Point `--codes-dir` at a NeuroWorkflow `codes/` folder:

```
codes/
├── nodes/<category>/*.py        # available node types  -> the palette
└── projects/<id>/workflow.py    # workflows (node instances + connections)
                                 # (also matches projects/<Name>/<Name>.py)
```

## Usage

```bash
# default: generates dashboard.html from ../../gui/.../codes
python build_dashboard.py

# explicit
python build_dashboard.py \
    --codes-dir /path/to/codes \
    --output my_dashboard.html \
    --title "NeuroWorkflow system — before hackathon"
```

Open the resulting `.html` in a browser. The left panel lists shared node types,
workflows, and the node catalog; the graph shows two modes (Workflows ↔ types,
and Type connections). Physics animates continuously.

## How it works (for the person improving it)

`build_dashboard.py` is organised as:

1. **`scan_node_catalog(codes_dir)`** — walk `codes/nodes/<cat>/*.py`, keep files
   that declare `NODE_DEFINITION` or subclass `Node`.
2. **`scan_workflows(codes_dir)`** — for each `codes/projects/<id>/`, parse the
   generated workflow script with regexes (`RE_WF_NAME`, `RE_IMPORT`,
   `RE_INSTANCE`, `RE_CONNECT`) to extract the workflow name, node instances
   (class + category) and edges.
3. **`build_model(codes_dir)`** — collapse to one node per node *type* (class),
   compute which types are shared across workflows, assign colors, build the
   graph nodes/edges + summary stats.
4. **`render_html(model, title)`** — inject the JSON payload + the vis-network lib
   into `_HTML_TEMPLATE` (the whole front-end is that one template string).

Everything is deliberately in one file so it can be copied and run anywhere.

### Ideas / good places to improve

- The workflow parsing is **regex over generated code** — brittle if the code
  generator changes. Could be made more robust (AST parsing, or reading the DB /
  flow JSON instead).
- The front-end (`_HTML_TEMPLATE` + inlined JS) could move to a real template file
  and/or add filtering, search, per-category coloring, export to PNG, etc.
- Continuous physics is heavy on large graphs (200+ node types); a pause/resume
  toggle would help.
- No tests yet — a couple of fixtures (a tiny fake `codes/` tree) would make it
  safe to refactor.
