# NeuroWorkflow System Dashboard

A single self-contained Python script that builds an interactive HTML dashboard of
the whole NeuroWorkflow system: every workflow shown as a cluster of connected
node instances, plus a catalog of all available node types (used / unused).

It reads the filesystem `codes/` tree. For **public-only** mode (the default) it
also consults the Django Postgres DB (`flow_projects.visibility`).

## Requirements

- Python 3.8+
- **Standard library** for parsing / HTML generation
- For `--visibility public` (default): **`psycopg2` or `psycopg`**, *or* an
  `--allowlist-file`. The NeuroWorkflow backend container already has `psycopg2`.

The graph rendering library is **vis-network**. By default the generated page
loads it from a CDN (works as long as the machine has internet). For a fully
offline dashboard, drop `vis-network.min.js` next to the script — if present it is
inlined into the HTML. (It is intentionally not committed to keep the repo light.)

## Privacy (important)

| Mode | What is included |
|------|------------------|
| `--visibility public` (**default**) | Active projects with `visibility=public` in the DB (matched to on-disk UUID or legacy name folders) |
| `--visibility all` | Every project folder under `codes/projects/` (includes **private** graphs) |

HTML embedding escapes titles and JSON so hostile workflow names cannot break out
of the page `<script>` block (XSS).

Do **not** publish HTML built with `--visibility all`. Public-only HTML is
appropriate for a controlled portal preview; still treat it as sensitive ops
output until product owners agree to host it.

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
# default: PUBLIC projects only (needs DB env or --database-url)
python build_dashboard.py \
    --codes-dir /path/to/codes \
    --output dashboard-public.html

# full internal ops view (private projects included — do not publish)
python build_dashboard.py --visibility all --output dashboard-all.html

# stdlib-only public filter via an allowlist file (one UUID/folder name per line)
python build_dashboard.py --allowlist-file public-ids.txt

# built-in smoke tests
python build_dashboard.py --self-test
```

### Running on snnbuilder (recommended)

The backend container can see both Postgres and `codes/`:

```bash
docker cp docs/system_dashboard/build_dashboard.py \
  neuro-workflow-backend-1:/tmp/build_dashboard.py

docker exec neuro-workflow-backend-1 \
  python /tmp/build_dashboard.py \
    --codes-dir /django-app/django-project/codes \
    --output /tmp/dashboard-public.html \
    --title "NeuroWorkflow — public projects"

docker cp neuro-workflow-backend-1:/tmp/dashboard-public.html ./dashboard-public.html
```

Open the resulting `.html` in a browser. Use **search** to filter the side lists,
and **pause / resume physics** on large graphs.

## How it works

1. **`scan_node_catalog`** — palette from `codes/nodes/`.
2. **Resolve allowlist** — DB public projects, or `--allowlist-file`, or none if
   `--visibility all`.
3. **`scan_workflows`** — parse each allowed project’s generated script (AST, regex fallback).
4. **`build_model` / `render_html`** — graph + XSS-safe HTML.

## Improvements in this iteration

- AST-first workflow parsing (regex kept as fallback)
- Side-panel **search / filter**
- **Pause / resume physics** for large graphs
- Clickable used types in the catalog
- Built-in `--self-test`
- **XSS-safe** HTML embedding
- **Public-only filtering** (default) via DB or allowlist

### Possible follow-ups

- Portal preview that serves a periodically regenerated **public** HTML
- Export graph to PNG; per-category color modes
