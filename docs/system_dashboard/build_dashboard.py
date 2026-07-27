#!/usr/bin/env python3
"""
Build a self-contained HTML dashboard showing the status of the NeuroWorkflow
system: every workflow as a colored cluster of connected node instances, plus a
catalog of all available node types (used / unused).

Data source: the filesystem `codes/` tree (no database connection needed).
  - codes/nodes/<category>/*.py        -> available node types (the palette)
  - codes/projects/<id>/workflow.py    -> workflows (node instances + connections)
    (also matches codes/projects/<Name>/<Name>.py)

Usage:
    python build_dashboard.py \
        --codes-dir ../../gui/workflow_backend/django-project/codes \
        --output dashboard.html \
        --title "NeuroWorkflow system — before hackathon"

Run it again after the hackathon with a different --output/--title to compare.

Privacy note: by default this includes only **public** active projects (via the
Django DB). Use ``--visibility all`` for a private ops view of every folder on
disk. Do not publish HTML that was built with ``--visibility all``.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import unittest
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlunparse

# --- regex fallback over generated workflow scripts ---------------------------
RE_WF_NAME = re.compile(r"""WorkflowBuilder\(\s*["']([^"']+)["']""")
# Category may render with a slash in generated code (e.g. "nodes.i/o.X"); accept it.
RE_IMPORT = re.compile(r"""^\s*from\s+nodes\.([\w/]+)\.(\w+)\s+import\s+(\w+)""", re.M)
# Project-local node imports, e.g. `from ChurchlandDatasetLoaderNode import ...`
RE_IMPORT_LOCAL = re.compile(r"""^\s*from\s+([\w.]+)\s+import\s+(\w*Node)\b""", re.M)
RE_INSTANCE = re.compile(r"""(\w+)\s*=\s*(\w+)\(\s*["']([^"']+)["']\s*\)""")
RE_CONNECT = re.compile(
    r"""connect\(\s*["']([^"']+)["']\s*,\s*["'][^"']*["']\s*,\s*"""
    r"""["']([^"']+)["']\s*,\s*["'][^"']*["']\s*\)"""
)

PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#e6beff", "#00b894",
]


def scan_node_catalog(codes_dir: Path):
    """Available node types: codes/nodes/<category>/<NodeName>.py."""
    catalog = []
    nodes_root = codes_dir / "nodes"
    if not nodes_root.is_dir():
        return catalog
    for cat_dir in sorted(p for p in nodes_root.iterdir() if p.is_dir()):
        if cat_dir.name.startswith("__") or cat_dir.name.startswith("."):
            continue
        for py in sorted(cat_dir.glob("*.py")):
            if py.stem == "__init__":
                continue
            try:
                txt = py.read_text(encoding="utf-8", errors="replace")
            except Exception:
                txt = ""
            if "NODE_DEFINITION" not in txt and "(Node)" not in txt:
                continue
            catalog.append({"name": py.stem, "category": cat_dir.name})
    return catalog


def _find_workflow_script(project_dir: Path):
    """Pick the generated workflow script in a project folder."""
    candidates = [
        p for p in project_dir.glob("*.py")
        if ".ipynb_checkpoints" not in p.parts
    ]
    if not candidates:
        return None
    by_name = {p.name.lower(): p for p in candidates}
    if "workflow.py" in by_name:
        return by_name["workflow.py"]
    dir_match = f"{project_dir.name.lower()}.py"
    if dir_match in by_name:
        return by_name[dir_match]
    return candidates[0]


def _literal_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _parse_workflow_ast(text: str, fallback_name: str):
    """Prefer AST parsing of generated workflow scripts; raise on failure."""
    tree = ast.parse(text)

    wf_name = fallback_name
    cls_category = {}
    instances = {}
    edges = []

    for node in tree.body:
        # from nodes.<cat>.<Mod> import <Class>
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if len(parts) >= 3 and parts[0] == "nodes":
                cat = parts[1].replace("/", "")
                for alias in node.names:
                    cls_category[alias.name] = cat
            else:
                for alias in node.names:
                    name = alias.name
                    if name.endswith("Node") and name not in cls_category:
                        cls_category[name] = "project-local"

    for node in ast.walk(tree):
        # WorkflowBuilder("Name", ...)
        if isinstance(node, ast.Call):
            func = node.func
            fname = None
            if isinstance(func, ast.Name):
                fname = func.id
            elif isinstance(func, ast.Attribute):
                fname = func.attr
            if fname == "WorkflowBuilder" and node.args:
                lit = _literal_str(node.args[0])
                if lit:
                    wf_name = lit

            # workflow_builder.connect("src", "port", "tgt", "port")
            if fname == "connect" and len(node.args) >= 3:
                src = _literal_str(node.args[0])
                tgt = _literal_str(node.args[2])
                if src and tgt:
                    edges.append({"src": src, "tgt": tgt})

        # var = SomeNode("instance_name")
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if not isinstance(call.func, ast.Name):
                continue
            cls = call.func.id
            if not (cls in cls_category or cls.endswith("Node")):
                continue
            if not call.args:
                continue
            inst = _literal_str(call.args[0])
            if not inst:
                continue
            category = cls_category.get(
                cls, "project-local" if cls.endswith("Node") else "unknown"
            )
            instances[inst] = {"cls": cls, "category": category}

    return {"name": wf_name, "instances": instances, "edges": edges}


def _parse_workflow_regex(text: str, fallback_name: str):
    """Original regex extractor — kept as a robust fallback."""
    m = RE_WF_NAME.search(text)
    wf_name = m.group(1) if m else fallback_name

    cls_category = {
        cls: cat.replace("/", "")
        for cat, _mod, cls in (
            (mm.group(1), mm.group(2), mm.group(3)) for mm in RE_IMPORT.finditer(text)
        )
    }
    for _mod, cls in RE_IMPORT_LOCAL.findall(text):
        if cls not in cls_category and cls != "WorkflowBuilder":
            cls_category[cls] = "project-local"
    node_classes = set(cls_category)

    instances = {}
    for _var, cls, inst in RE_INSTANCE.findall(text):
        if cls in node_classes or cls.endswith("Node"):
            category = cls_category.get(
                cls, "project-local" if cls.endswith("Node") else "unknown"
            )
            instances[inst] = {"cls": cls, "category": category}

    edges = [{"src": src, "tgt": tgt} for src, tgt in RE_CONNECT.findall(text)]
    return {"name": wf_name, "instances": instances, "edges": edges}


def parse_workflow(script: Path):
    """Extract workflow name, node instances (+class/category), and edges.

    Tries AST first (less brittle across generator tweaks); falls back to regex
    if the file is not parseable as Python.
    """
    text = script.read_text(encoding="utf-8", errors="replace")
    fallback_name = script.parent.name
    try:
        return _parse_workflow_ast(text, fallback_name)
    except SyntaxError:
        return _parse_workflow_regex(text, fallback_name)


def scan_workflows(codes_dir: Path, allowed_dir_names: set[str] | None = None):
    """Scan project workflow scripts.

    If ``allowed_dir_names`` is set, only project directories whose names appear
    in that set are included (used for public-only filtering).
    """
    projects_root = codes_dir / "projects"
    workflows = []
    if not projects_root.is_dir():
        return workflows
    for proj in sorted(p for p in projects_root.iterdir() if p.is_dir()):
        if proj.name.startswith(".") or proj.name.startswith("__"):
            continue
        if allowed_dir_names is not None and proj.name not in allowed_dir_names:
            continue
        script = _find_workflow_script(proj)
        if not script:
            continue
        wf = parse_workflow(script)
        if not wf["instances"] and not wf["edges"]:
            continue
        wf["id"] = proj.name
        workflows.append(wf)
    return workflows


def legacy_project_dirname(name: str, project_id: str) -> str:
    """Mirror ``path_utils.legacy_project_dir`` folder naming."""
    legacy_name = (name or project_id).replace(" ", "").capitalize()
    legacy_name = re.sub(r"[^A-Za-z0-9_.-]", "_", legacy_name) or project_id
    return legacy_name


def load_allowlist_file(path: Path) -> set[str]:
    """Load folder names / project UUIDs (one per line; ``#`` comments ok)."""
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.add(line)
    return keys


def resolve_database_url(explicit: str | None) -> str | None:
    """Build a Postgres URL from CLI / env (Django-style ``DB_*`` supported)."""
    if explicit:
        return explicit
    for key in ("DATABASE_URL", "PORTAL_DASHBOARD_DATABASE_URL"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    host = (os.environ.get("DB_HOST") or "").strip()
    if not host:
        return None
    port = (os.environ.get("DB_PORT") or "5432").strip()
    name = (os.environ.get("DB_NAME") or "django_").strip()
    user = (os.environ.get("DB_USER") or "postgres").strip()
    password = os.environ.get("DB_PASSWORD") or ""
    # quote password so special characters stay valid in the URL
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}"
    return urlunparse(("postgresql", netloc, f"/{name}", "", "", ""))


def _connect_postgres(database_url: str):
    """Return a DB-API connection (psycopg2 or psycopg v3)."""
    try:
        import psycopg2  # type: ignore

        return psycopg2.connect(database_url)
    except ImportError:
        pass
    try:
        import psycopg  # type: ignore

        return psycopg.connect(database_url)
    except ImportError as exc:
        raise RuntimeError(
            "Public-project filtering needs psycopg2 or psycopg. "
            "Install one of them, pass --allowlist-file, run inside the "
            "backend container (has psycopg2), or use --visibility all."
        ) from exc


def fetch_public_project_dir_names(database_url: str) -> set[str]:
    """Return on-disk folder names for active public FlowProjects.

    Includes both stable UUID dirs and legacy name-based dirs.
    """
    conn = _connect_postgres(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, COALESCE(name, '')
                FROM flow_projects
                WHERE is_active = TRUE AND visibility = 'public'
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    keys: set[str] = set()
    for project_id, name in rows:
        keys.add(project_id)
        keys.add(legacy_project_dirname(name, project_id))
    return keys


def build_model(
    codes_dir: Path,
    allowed_dir_names: set[str] | None = None,
    *,
    visibility: str = "all",
):
    catalog = scan_node_catalog(codes_dir)
    workflows = scan_workflows(codes_dir, allowed_dir_names=allowed_dir_names)

    used_classes = set()
    for wf in workflows:
        for inst in wf["instances"].values():
            used_classes.add(inst["cls"])
    for entry in catalog:
        entry["used"] = entry["name"] in used_classes

    for i, wf in enumerate(workflows):
        wf["color"] = PALETTE[i % len(PALETTE)]
    wf_name = {wf["id"]: wf["name"] for wf in workflows}

    class_wfids = {}
    class_category = {}
    for wf in workflows:
        for meta in wf["instances"].values():
            class_wfids.setdefault(meta["cls"], set()).add(wf["id"])
            class_category[meta["cls"]] = meta["category"]

    graph_nodes = []
    for cls in sorted(class_wfids):
        wfids = sorted(class_wfids[cls])
        graph_nodes.append({
            "id": cls,
            "label": cls,
            "category": class_category.get(cls, "unknown"),
            "n_workflows": len(wfids),
            "shared": len(wfids) > 1,
            "wfids": wfids,
            "wfnames": [wf_name[w] for w in wfids],
        })

    graph_edges, seen = [], set()
    for wf in workflows:
        insts = wf["instances"]
        for e in wf["edges"]:
            s = insts.get(e["src"], {}).get("cls")
            t = insts.get(e["tgt"], {}).get("cls")
            if not s or not t:
                continue
            key = (s, t, wf["id"])
            if key in seen:
                continue
            seen.add(key)
            graph_edges.append({
                "from": s, "to": t, "workflow": wf["id"], "color": wf["color"]
            })

    wf_summaries = [{
        "id": wf["id"], "name": wf["name"], "color": wf["color"],
        "n_nodes": len(wf["instances"]), "n_edges": len(wf["edges"]),
    } for wf in workflows]

    shared_types = sorted(
        ({"name": n["id"], "n": n["n_workflows"], "wfnames": n["wfnames"]}
         for n in graph_nodes if n["shared"]),
        key=lambda x: (-x["n"], x["name"]),
    )

    return {
        "workflows": wf_summaries,
        "nodes": graph_nodes,
        "edges": graph_edges,
        "shared_types": shared_types,
        "catalog": catalog,
        "visibility": visibility,
        "stats": {
            "n_workflows": len(workflows),
            "n_types_used": len(graph_nodes),
            "n_shared": len(shared_types),
            "n_node_types": len(catalog),
            "visibility": visibility,
        },
    }


def _html_escape(text: str) -> str:
    """Escape text for HTML element / attribute context (e.g. <title>)."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _json_for_script(obj) -> str:
    """Serialize JSON safe to embed inside a <script> tag.

    Standard json.dumps leaves ``</script>`` intact, which browsers treat as
    ending the script element (XSS if a workflow name / title is hostile).
    Escaping ``<`` (and siblings) to Unicode escapes keeps the JSON valid for
    ``JSON.parse`` / JS object literals while neutralizing breakout.
    """
    return (
        json.dumps(obj, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_html(model: dict, title: str) -> str:
    payload = _json_for_script(model)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    visibility = (model.get("visibility") or model.get("stats", {}).get("visibility") or "all")
    if visibility == "public":
        privacy = (
            "Privacy: only active projects with visibility=public (from the DB "
            "or an allowlist) are included."
        )
    else:
        privacy = (
            "Privacy: this page includes every matching project folder on disk "
            "(including private ones). Do not publish."
        )

    lib_file = Path(__file__).resolve().parent / "vis-network.min.js"
    if lib_file.is_file():
        vis_lib = "<script>\n" + lib_file.read_text(encoding="utf-8") + "\n</script>"
    else:
        vis_lib = (
            '<script src="https://unpkg.com/vis-network/standalone/umd/'
            'vis-network.min.js"></script>'
        )

    return (
        _HTML_TEMPLATE.replace("__TITLE__", _html_escape(title))
        .replace("__GENERATED__", _html_escape(generated))
        .replace("__VISIBILITY__", _html_escape(visibility))
        .replace("__PRIVACY__", _html_escape(privacy))
        .replace("__VIS_LIB__", vis_lib)
        .replace("__DATA__", payload)
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
__VIS_LIB__
<style>
  html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#222}
  #wrap{display:flex;height:100vh}
  #side{width:340px;overflow-y:auto;border-right:1px solid #e2e8f0;padding:14px 16px;box-sizing:border-box}
  #graph{flex:1;position:relative}
  h1{font-size:16px;margin:0 0 2px}
  .sub{color:#888;font-size:11px;margin-bottom:12px}
  .stats{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
  .stat{background:#f1f5f9;border-radius:8px;padding:6px 10px;font-size:12px}
  .stat b{display:block;font-size:18px}
  h2{font-size:12px;text-transform:uppercase;color:#64748b;letter-spacing:.04em;margin:16px 0 6px}
  .wf,.sh{display:flex;align-items:center;gap:8px;padding:5px 6px;border-radius:6px;cursor:pointer;font-size:13px}
  .wf:hover,.sh:hover{background:#f1f5f9}
  .wf.active,.sh.active{background:#e2e8f0}
  .wf.hidden,.sh.hidden,.nt.hidden,.cat.hidden{display:none}
  .dot{width:11px;height:11px;border-radius:50%;flex:0 0 auto}
  .wf .nm,.sh .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .wf .ct,.sh .ct{color:#94a3b8;font-size:11px}
  .cat{font-size:12px;margin:8px 0 2px;color:#475569;font-weight:600}
  .nt{font-size:12px;padding:2px 0 2px 10px;color:#334155;cursor:pointer}
  .nt:hover{background:#f8fafc}
  .nt.unused{color:#cbd5e1}
  .nt .u{font-size:10px;color:#94a3b8}
  #reset,#physics{display:inline-block;font-size:11px;color:#2563eb;cursor:pointer;padding:4px 8px;border:1px solid #bfdbfe;border-radius:6px;margin:0 4px 4px 0}
  #reset:hover,#physics:hover{background:#eff6ff}
  #modes{margin-bottom:10px}
  .mode{display:inline-block;font-size:11px;padding:4px 8px;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;margin:0 4px 4px 0;color:#475569}
  .mode.active{background:#1e293b;color:#fff;border-color:#1e293b}
  #search{width:100%;box-sizing:border-box;margin:0 0 10px;padding:7px 10px;border:1px solid #cbd5e1;border-radius:8px;font-size:13px}
  #search:focus{outline:none;border-color:#93c5fd;box-shadow:0 0 0 2px #dbeafe}
  .privacy{font-size:10px;color:#94a3b8;margin-top:18px;line-height:1.4}
</style>
</head>
<body>
<div id="wrap">
  <div id="side">
    <h1>__TITLE__</h1>
    <div class="sub">generated __GENERATED__ · filesystem snapshot · visibility=__VISIBILITY__</div>
    <div class="stats" id="stats"></div>
    <input id="search" type="search" placeholder="Filter workflows / node types…" autocomplete="off"/>
    <div id="modes">
      <span class="mode active" data-m="usage">Workflows &harr; types</span>
      <span class="mode" data-m="conn">Type connections</span>
      <span id="reset">&#8635; show all</span>
      <span id="physics">&#10074;&#10074; pause physics</span>
    </div>
    <h2>Shared node types (used by &gt;1 workflow)</h2>
    <div id="shared"></div>
    <h2>Workflows</h2>
    <div id="wflist"></div>
    <h2>Available node types</h2>
    <div id="catalog"></div>
    <div class="privacy">__PRIVACY__</div>
  </div>
  <div id="graph"></div>
</div>
<script>
const DATA = __DATA__;
const UNIQUE_COLOR = "#cbd5e1";
const SHARED_COLOR = "#f59e0b";

const s = DATA.stats;
document.getElementById('stats').innerHTML = [
  ['workflows', s.n_workflows], ['node types used', s.n_types_used],
  ['shared types', s.n_shared], ['types available', s.n_node_types],
].map(([k,v])=>`<div class="stat"><b>${v}</b>${k}</div>`).join('');

const wfColor={}, wfName={}, typeById={};
DATA.workflows.forEach(w=>{wfColor[w.id]=w.color; wfName[w.id]=w.name;});
DATA.nodes.forEach(n=>typeById[n.id]=n);
const DIM_BG='#eef2f7', DIM_BD='#e2e8f0', DIM_FT='#cbd5e1';
const HUB='wf::';

function nodeColor(n){ return n.shared ? SHARED_COLOR : UNIQUE_COLOR; }
function nodeTitle(n){
  return `${n.label}  (${n.category})\nused by ${n.n_workflows} workflow(s):\n  - `
         + n.wfnames.join('\n  - ');
}
function typeNode(n){
  return {id:n.id, label:n.label, title:nodeTitle(n), value:n.n_workflows,
          color:{background:nodeColor(n), border:'#475569'}, shape:'dot',
          borderWidth:2, font:{size:18,color:'#1e293b',face:'monospace'}};
}

function buildUsage(){
  const ns=[], es=[]; let i=0;
  DATA.workflows.forEach(w=>ns.push({
    id:HUB+w.id, label:w.name, title:'workflow: '+w.name, shape:'box', margin:9,
    color:{background:w.color, border:'#1e293b'}, font:{color:'#fff', size:18, bold:true}
  }));
  DATA.nodes.forEach(n=>ns.push(typeNode(n)));
  DATA.nodes.forEach(n=>n.wfids.forEach(wid=>es.push({
    id:i++, from:HUB+wid, to:n.id, color:{color:wfColor[wid]}, width:3,
    smooth:{enabled:true,type:'continuous'}
  })));
  return {nodes:new vis.DataSet(ns), edges:new vis.DataSet(es)};
}

function buildConn(){
  const nodes=new vis.DataSet(DATA.nodes.map(typeNode));
  const edges=new vis.DataSet(DATA.edges.map((e,i)=>({
    id:i, from:e.from, to:e.to, _wf:e.workflow, color:{color:e.color}, width:4,
    smooth:{enabled:true,type:'continuous'}
  })));
  return {nodes,edges};
}

const OPTS = {
  nodes:{scaling:{min:22,max:60,label:{enabled:true,min:14,max:30}}},
  physics:{barnesHut:{gravitationalConstant:-18000, springLength:200, avoidOverlap:0.8},
           stabilization:{iterations:250}, enabled:true},
  interaction:{hover:true, tooltipDelay:120}
};
let mode='usage';
let physicsOn=true;
let cur=buildUsage();
const net=new vis.Network(document.getElementById('graph'), cur, OPTS);

function setMode(m){
  mode=m;
  cur = (m==='usage') ? buildUsage() : buildConn();
  net.setData(cur);
  net.setOptions({physics:{enabled:physicsOn}});
  document.querySelectorAll('.mode').forEach(x=>x.classList.toggle('active', x.dataset.m===m));
  net.fit({animation:false});
}
document.querySelectorAll('.mode').forEach(el=>{ el.onclick=()=>setMode(el.dataset.m); });

const physicsBtn=document.getElementById('physics');
physicsBtn.onclick=()=>{
  physicsOn=!physicsOn;
  net.setOptions({physics:{enabled:physicsOn}});
  physicsBtn.innerHTML = physicsOn ? '&#10074;&#10074; pause physics' : '&#9654; resume physics';
};

function clearActive(){ document.querySelectorAll('.wf,.sh').forEach(x=>x.classList.remove('active')); }
function recolor(keepNodeIds, keepEdgeFn){
  const keep=new Set(keepNodeIds);
  cur.nodes.update(cur.nodes.getIds().map(id=>{
    const on=keep.has(id);
    let col;
    if(String(id).startsWith(HUB)) col = on?{background:wfColor[String(id).slice(HUB.length)],border:'#1e293b'}:{background:DIM_BG,border:DIM_BD};
    else { const t=typeById[id]; col = on?{background:nodeColor(t),border:'#475569'}:{background:DIM_BG,border:DIM_BD}; }
    return {id, color:col, font:{color: on?(String(id).startsWith(HUB)?'#fff':'#1e293b'):DIM_FT}};
  }));
  cur.edges.update(cur.edges.get().map(ed=>({id:ed.id, hidden:!keepEdgeFn(ed)})));
  if(keep.size) net.fit({nodes:[...keep], animation:true});
}
function resetView(){
  document.querySelectorAll('.wf,.sh').forEach(x=>x.classList.remove('active'));
  setMode(mode);
  net.fit({animation:true});
}
function focusWorkflow(id, el){
  document.querySelectorAll('.wf,.sh').forEach(x=>x.classList.remove('active'));
  if(el) el.classList.add('active');
  const keep = DATA.nodes.filter(n=>n.wfids.includes(id)).map(n=>n.id);
  if(mode==='usage'){
    keep.push(HUB+id);
    recolor(keep, ed=>ed.from===HUB+id);
  } else {
    recolor(keep, ed=>ed._wf===id);
  }
}
function focusNode(name, el){
  document.querySelectorAll('.wf,.sh').forEach(x=>x.classList.remove('active'));
  if(el) el.classList.add('active');
  const t=typeById[name]; if(!t) return;
  const keep=[name];
  if(mode==='usage') t.wfids.forEach(w=>keep.push(HUB+w));
  recolor(keep, ed=> (mode==='usage') ? ed.to===name : (ed.from===name||ed.to===name));
}

const shared = document.getElementById('shared');
if(!DATA.shared_types.length){ shared.innerHTML = '<div class="nt unused">none yet</div>'; }
DATA.shared_types.forEach(t=>{
  const el=document.createElement('div'); el.className='sh';
  el.dataset.q = (t.name+' '+t.wfnames.join(' ')).toLowerCase();
  el.innerHTML = `<span class="dot" style="background:${SHARED_COLOR}"></span>`+
                 `<span class="nm" title="${t.wfnames.join(', ')}">${t.name}</span>`+
                 `<span class="ct">${t.n} wf</span>`;
  el.onclick = ()=>focusNode(t.name, el);
  shared.appendChild(el);
});

const wflist = document.getElementById('wflist');
DATA.workflows.forEach(w=>{
  const el = document.createElement('div');
  el.className='wf';
  el.dataset.q = (w.name+' '+w.id).toLowerCase();
  el.innerHTML = `<span class="dot" style="background:${w.color}"></span>`+
                 `<span class="nm" title="${w.name}">${w.name}</span>`+
                 `<span class="ct">${w.n_nodes}n/${w.n_edges}e</span>`;
  el.onclick = ()=>focusWorkflow(w.id, el);
  wflist.appendChild(el);
});
document.getElementById('reset').onclick = resetView;

const cat = document.getElementById('catalog');
const byCat = {};
DATA.catalog.forEach(c=>{(byCat[c.category]=byCat[c.category]||[]).push(c)});
Object.keys(byCat).sort().forEach(c=>{
  const h=document.createElement('div'); h.className='cat'; h.textContent=c;
  h.dataset.q = c.toLowerCase();
  cat.appendChild(h);
  byCat[c].sort((a,b)=>a.name.localeCompare(b.name)).forEach(nt=>{
    const d=document.createElement('div');
    d.className='nt'+(nt.used?'':' unused');
    d.dataset.q = (nt.name+' '+c).toLowerCase();
    d.innerHTML = nt.name + (nt.used?'':' <span class="u">unused</span>');
    if(nt.used) d.onclick = ()=>focusNode(nt.name, null);
    cat.appendChild(d);
  });
});

function applyFilter(q){
  const query = (q||'').trim().toLowerCase();
  document.querySelectorAll('.wf,.sh,.nt,.cat').forEach(el=>{
    if(!query){ el.classList.remove('hidden'); return; }
    const hay = el.dataset.q || '';
    el.classList.toggle('hidden', !hay.includes(query));
  });
  // keep category headers visible if any child matches
  document.querySelectorAll('#catalog .cat').forEach(h=>{
    let n=h.nextElementSibling, any=false;
    while(n && !n.classList.contains('cat')){
      if(n.classList.contains('nt') && !n.classList.contains('hidden')) any=true;
      n=n.nextElementSibling;
    }
    if(query) h.classList.toggle('hidden', !any && !(h.dataset.q||'').includes(query));
  });
}
document.getElementById('search').addEventListener('input', e=>applyFilter(e.target.value));
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Tiny self-test (stdlib unittest). Run: python build_dashboard.py --self-test
# ---------------------------------------------------------------------------

_MINI_WORKFLOW = '''\
from neuroworkflow.core.workflow import WorkflowBuilder
from nodes.analysis.SpikeAnalyzerNode import SpikeAnalyzerNode
from nodes.network.DemoNetNode import DemoNetNode

def main():
    workflow_builder = WorkflowBuilder("Mini Demo")
    spikes = SpikeAnalyzerNode("spikes")
    spikes.configure()
    net = DemoNetNode("net")
    net.configure()
    workflow_builder.add_node(spikes)
    workflow_builder.add_node(net)
    workflow_builder.connect("spikes", "out", "net", "in")
    workflow = workflow_builder.build()
'''

_MINI_NODE_A = "class SpikeAnalyzerNode(Node):\n    NODE_DEFINITION = {}\n"
_MINI_NODE_B = "class DemoNetNode(Node):\n    NODE_DEFINITION = {}\n"
_MINI_NODE_UNUSED = "class UnusedNode(Node):\n    NODE_DEFINITION = {}\n"


def _write_mini_codes(root: Path) -> Path:
    codes = root / "codes"
    (codes / "nodes" / "analysis").mkdir(parents=True)
    (codes / "nodes" / "network").mkdir(parents=True)
    (codes / "projects" / "proj-mini").mkdir(parents=True)
    (codes / "nodes" / "analysis" / "SpikeAnalyzerNode.py").write_text(_MINI_NODE_A)
    (codes / "nodes" / "network" / "DemoNetNode.py").write_text(_MINI_NODE_B)
    (codes / "nodes" / "analysis" / "UnusedNode.py").write_text(_MINI_NODE_UNUSED)
    (codes / "projects" / "proj-mini" / "workflow.py").write_text(_MINI_WORKFLOW)
    return codes


class DashboardSelfTest(unittest.TestCase):
    def test_mini_codes_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            codes = _write_mini_codes(Path(tmp))
            model = build_model(codes)
            self.assertEqual(model["stats"]["n_workflows"], 1)
            self.assertEqual(model["stats"]["n_types_used"], 2)
            self.assertEqual(model["stats"]["n_node_types"], 3)
            self.assertEqual(model["workflows"][0]["name"], "Mini Demo")
            names = {c["name"]: c["used"] for c in model["catalog"]}
            self.assertTrue(names["SpikeAnalyzerNode"])
            self.assertTrue(names["DemoNetNode"])
            self.assertFalse(names["UnusedNode"])
            self.assertEqual(len(model["edges"]), 1)
            html = render_html(model, "test")
            self.assertIn("Mini Demo", html)
            self.assertIn("pause physics", html)
            self.assertIn("Filter workflows", html)

    def test_ast_and_regex_agree_on_mini(self):
        parsed_ast = _parse_workflow_ast(_MINI_WORKFLOW, "fallback")
        parsed_re = _parse_workflow_regex(_MINI_WORKFLOW, "fallback")
        self.assertEqual(parsed_ast["name"], parsed_re["name"])
        self.assertEqual(set(parsed_ast["instances"]), set(parsed_re["instances"]))
        self.assertEqual(
            {(e["src"], e["tgt"]) for e in parsed_ast["edges"]},
            {(e["src"], e["tgt"]) for e in parsed_re["edges"]},
        )

    def test_html_escapes_hostile_title_and_workflow_name(self):
        hostile_name = "</script><img src=x onerror=alert(1)>"
        hostile_title = "</title><script>alert(2)</script>"
        model = {
            "generated_at": "now",
            "stats": {
                "n_workflows": 1,
                "n_types_used": 0,
                "n_shared": 0,
                "n_node_types": 0,
            },
            "workflows": [
                {
                    "id": "w1",
                    "name": hostile_name,
                    "n_instances": 0,
                    "n_edges": 0,
                    "color": "#000",
                }
            ],
            "nodes": [],
            "edges": [],
            "shared_types": [],
            "catalog": [],
        }
        html = render_html(model, hostile_title)
        self.assertNotIn(hostile_name, html)
        self.assertNotIn(hostile_title, html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertIn("&lt;/title&gt;", html)
        # Payload must still round-trip as JSON for the page script.
        import re

        m = re.search(r"const DATA = (\{.*?\});\s*\n", html, re.S)
        self.assertIsNotNone(m)
        data = json.loads(m.group(1))
        self.assertEqual(data["workflows"][0]["name"], hostile_name)

    def test_allowlist_filters_projects(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codes = _write_mini_codes(root)
            # Second project that should be excluded by allowlist.
            other = codes / "projects" / "proj-secret"
            other.mkdir()
            (other / "workflow.py").write_text(
                _MINI_WORKFLOW.replace("Mini Demo", "Secret Demo"),
                encoding="utf-8",
            )
            all_model = build_model(codes, visibility="all")
            self.assertEqual(all_model["stats"]["n_workflows"], 2)

            public_model = build_model(
                codes,
                allowed_dir_names={"proj-mini"},
                visibility="public",
            )
            self.assertEqual(public_model["stats"]["n_workflows"], 1)
            self.assertEqual(public_model["workflows"][0]["name"], "Mini Demo")
            self.assertEqual(public_model["visibility"], "public")
            html = render_html(public_model, "public only")
            self.assertIn("visibility=public", html)
            self.assertIn("only active projects with visibility=public", html)

    def test_legacy_project_dirname_matches_backend(self):
        self.assertEqual(
            legacy_project_dirname("Demo project", "abc"),
            "Demoproject",
        )
        self.assertEqual(
            legacy_project_dirname("ANTs fMRI!", "abc"),
            "Antsfmri_",
        )


def resolve_allowed_dir_names(args) -> tuple[set[str] | None, str]:
    """Return (allowlist or None, visibility label)."""
    visibility = args.visibility
    if args.allowlist_file:
        keys = load_allowlist_file(Path(args.allowlist_file))
        return keys, visibility

    if visibility == "all":
        return None, "all"

    # visibility == public
    db_url = resolve_database_url(args.database_url)
    if not db_url:
        raise SystemExit(
            "visibility=public requires DB access or an allowlist.\n"
            "Provide --database-url / DATABASE_URL / DB_* env vars, "
            "or --allowlist-file, or pass --visibility all."
        )
    keys = fetch_public_project_dir_names(db_url)
    return keys, "public"


def main(argv=None):
    here = Path(__file__).resolve().parent
    default_codes = (
        here / ".." / ".." / "gui" / "workflow_backend" / "django-project" / "codes"
    )

    ap = argparse.ArgumentParser(
        description="Build the NeuroWorkflow system dashboard (HTML)."
    )
    ap.add_argument(
        "--codes-dir",
        default=str(default_codes),
        help="Path to the django-project/codes folder.",
    )
    ap.add_argument(
        "--output",
        default=str(here / "dashboard.html"),
        help="Output HTML path.",
    )
    ap.add_argument(
        "--title",
        default="NeuroWorkflow system status",
        help="Dashboard title.",
    )
    ap.add_argument(
        "--visibility",
        choices=("public", "all"),
        default="public",
        help="public (default): only active DB-public projects. "
        "all: every project folder on disk (internal ops).",
    )
    ap.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL for visibility=public "
        "(else DATABASE_URL / DB_HOST+DB_* env).",
    )
    ap.add_argument(
        "--allowlist-file",
        default=None,
        help="Optional file of project folder names / UUIDs (one per line). "
        "Skips the DB when set.",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in unit tests and exit.",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DashboardSelfTest)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    codes_dir = Path(args.codes_dir).resolve()
    if not codes_dir.is_dir():
        sys.exit(f"codes dir not found: {codes_dir}")

    allowed, visibility = resolve_allowed_dir_names(args)
    model = build_model(
        codes_dir, allowed_dir_names=allowed, visibility=visibility
    )
    html = render_html(model, args.title)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")

    st = model["stats"]
    print(f"Read: {codes_dir}")
    print(f"  visibility={visibility}")
    if allowed is not None:
        print(f"  allowlist_keys={len(allowed)}")
    print(
        f"  workflows={st['n_workflows']}  node_types_used={st['n_types_used']}  "
        f"shared={st['n_shared']}  types_available={st['n_node_types']}"
    )
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
