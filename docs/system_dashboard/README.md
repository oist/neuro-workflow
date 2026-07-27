# NeuroWorkflow System Dashboard

A single self-contained Python script that builds an interactive HTML dashboard of
the whole NeuroWorkflow system: every workflow shown as a cluster of connected
node instances, plus a catalog of all available node types (used / unused).

It reads **only the filesystem** — no database, no server connection needed.

## Requirements

- Python 3.8+
- **No third-party packages** (standard library only: `argparse`, `ast`, `json`,
  `re`, `pathlib`, `datetime`, `unittest`).

The graph rendering library is **vis-network**. By default the generated page
loads it from a CDN (works as long as the machine has internet). For a fully
offline dashboard, drop `vis-network.min.js` next to the script — if present it is
inlined into the HTML. (It is intentionally not committed to keep the repo light.)

## Privacy (important)

The scanner walks **every** `codes/projects/<id>/` folder on disk. That includes
**private** projects (names, which node types they use, and how they are wired).
Treat generated HTML as **internal / access-controlled** unless you add filtering
or anonymization before publishing (e.g. on a public portal).

HTML embedding escapes titles and JSON so hostile workflow names cannot break out
of the page `<script>` block (XSS). That does **not** replace access control:
still do not publish the file until a public-only (or anonymized) mode exists.

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

# built-in smoke tests (tiny fake codes/ tree)
python build_dashboard.py --self-test
```

Open the resulting `.html` in a browser. The left panel lists shared node types,
workflows, and the node catalog; the graph shows two modes (Workflows ↔ types,
and Type connections). Use **search** to filter the side lists, and **pause /
resume physics** on large graphs.

## How it works

`build_dashboard.py` is organised as:

1. **`scan_node_catalog(codes_dir)`** — walk `codes/nodes/<cat>/*.py`, keep files
   that declare `NODE_DEFINITION` or subclass `Node`.
2. **`scan_workflows(codes_dir)`** — for each `codes/projects/<id>/`, parse the
   generated workflow script. Prefer **AST** parsing; fall back to regexes if the
   file is not valid Python.
3. **`build_model(codes_dir)`** — collapse to one node per node *type* (class),
   compute which types are shared across workflows, assign colors, build the
   graph nodes/edges + summary stats.
4. **`render_html(model, title)`** — inject the JSON payload + the vis-network lib
   into `_HTML_TEMPLATE` (the whole front-end is that one template string).

Everything is deliberately in one file so it can be copied and run anywhere.

## Improvements in this iteration

- AST-first workflow parsing (regex kept as fallback)
- Side-panel **search / filter**
- **Pause / resume physics** for large graphs
- Clickable used types in the catalog
- Built-in `--self-test` with a tiny fixture (no external deps)
- Explicit **privacy** note in the UI footer and this README
- **XSS-safe** HTML embedding for titles and the JSON payload

### Possible follow-ups

- Filter to public projects only (needs DB or a visibility manifest) — required
  before any portal / public hosting of generated HTML
- Portal preview that serves a periodically regenerated HTML
- Export graph to PNG; per-category color modes
