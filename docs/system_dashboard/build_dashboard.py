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
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# --- regex patterns over the generated workflow scripts -----------------------
RE_WF_NAME = re.compile(r"""WorkflowBuilder\(\s*["']([^"']+)["']""")
# Category may render with a slash in generated code (e.g. "nodes.i/o.X"); accept it.
RE_IMPORT = re.compile(r"""^\s*from\s+nodes\.([\w/]+)\.(\w+)\s+import\s+(\w+)""", re.M)
# Project-local node imports, e.g. `from ChurchlandDatasetLoaderNode import ...`
# (custom nodes uploaded with a project, not from the shared nodes/ package).
RE_IMPORT_LOCAL = re.compile(r"""^\s*from\s+([\w.]+)\s+import\s+(\w*Node)\b""", re.M)
RE_INSTANCE = re.compile(r"""(\w+)\s*=\s*(\w+)\(\s*["']([^"']+)["']\s*\)""")
RE_CONNECT = re.compile(
    r"""connect\(\s*["']([^"']+)["']\s*,\s*["'][^"']*["']\s*,\s*"""
    r"""["']([^"']+)["']\s*,\s*["'][^"']*["']\s*\)"""
)

# distinct, readable colors cycled across workflows
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
        if cat_dir.name.startswith("__"):
            continue
        for py in sorted(cat_dir.glob("*.py")):
            if py.stem == "__init__":
                continue
            try:
                txt = py.read_text(encoding="utf-8", errors="replace")
            except Exception:
                txt = ""
            # a real node declares NODE_DEFINITION or subclasses Node; skip helper .py
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
    # prefer workflow.py, then <DirName>.py, else the first script
    by_name = {p.name.lower(): p for p in candidates}
    if "workflow.py" in by_name:
        return by_name["workflow.py"]
    dir_match = f"{project_dir.name.lower()}.py"
    if dir_match in by_name:
        return by_name[dir_match]
    return candidates[0]


def parse_workflow(script: Path):
    """Extract workflow name, node instances (+class/category), and edges."""
    text = script.read_text(encoding="utf-8", errors="replace")

    m = RE_WF_NAME.search(text)
    wf_name = m.group(1) if m else script.parent.name

    # class -> category from the shared-palette imports ("i/o" -> "io")
    cls_category = {cls: cat.replace("/", "") for cat, _mod, cls in
                    ((mm.group(1), mm.group(2), mm.group(3)) for mm in RE_IMPORT.finditer(text))}
    # project-local custom node imports -> category "project-local"
    for _mod, cls in RE_IMPORT_LOCAL.findall(text):
        if cls not in cls_category and cls != "WorkflowBuilder":
            cls_category[cls] = "project-local"
    node_classes = set(cls_category)

    # instances: var = Class("instance_name"). Keep known node classes OR any class
    # whose name ends in "Node" (covers project-local nodes not matched by imports).
    instances = {}  # instance_name -> {cls, category}
    for _var, cls, inst in RE_INSTANCE.findall(text):
        if cls in node_classes or cls.endswith("Node"):
            category = cls_category.get(cls, "project-local" if cls.endswith("Node") else "unknown")
            instances[inst] = {"cls": cls, "category": category}

    # edges from connect("src", ..., "tgt", ...). Endpoints that don't resolve to a
    # captured instance are dropped in build_model — we never fabricate a node type.
    edges = [{"src": src, "tgt": tgt} for src, tgt in RE_CONNECT.findall(text)]

    return {"name": wf_name, "instances": instances, "edges": edges}


def scan_workflows(codes_dir: Path):
    projects_root = codes_dir / "projects"
    workflows = []
    if not projects_root.is_dir():
        return workflows
    for proj in sorted(p for p in projects_root.iterdir() if p.is_dir()):
        script = _find_workflow_script(proj)
        if not script:
            continue
        wf = parse_workflow(script)
        if not wf["instances"] and not wf["edges"]:
            continue
        wf["id"] = proj.name
        workflows.append(wf)
    return workflows


def build_model(codes_dir: Path):
    catalog = scan_node_catalog(codes_dir)
    workflows = scan_workflows(codes_dir)

    # which node types are actually used in some workflow
    used_classes = set()
    for wf in workflows:
        for inst in wf["instances"].values():
            used_classes.add(inst["cls"])
    for entry in catalog:
        entry["used"] = entry["name"] in used_classes

    # workflow colors
    for i, wf in enumerate(workflows):
        wf["color"] = PALETTE[i % len(PALETTE)]
    wf_name = {wf["id"]: wf["name"] for wf in workflows}

    # ---- one node per node TYPE (class), shared across workflows ----
    class_wfids = {}      # cls -> set of workflow ids that use it
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

    # ---- edges between node types, one per (src_cls, tgt_cls, workflow) ----
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
            graph_edges.append({"from": s, "to": t, "workflow": wf["id"], "color": wf["color"]})

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
        "stats": {
            "n_workflows": len(workflows),
            "n_types_used": len(graph_nodes),
            "n_shared": len(shared_types),
            "n_node_types": len(catalog),
        },
    }


def render_html(model: dict, title: str) -> str:
    payload = json.dumps(model)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Inline the vis-network library if vendored locally (fully offline / no CDN,
    # so a poor connection doesn't slow loading). Fall back to the CDN otherwise.
    lib_file = Path(__file__).resolve().parent / "vis-network.min.js"
    if lib_file.is_file():
        vis_lib = "<script>\n" + lib_file.read_text(encoding="utf-8") + "\n</script>"
    else:
        vis_lib = ('<script src="https://unpkg.com/vis-network/standalone/umd/'
                   'vis-network.min.js"></script>')

    return _HTML_TEMPLATE.replace("__TITLE__", title) \
                         .replace("__GENERATED__", generated) \
                         .replace("__VIS_LIB__", vis_lib) \
                         .replace("__DATA__", payload)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
__VIS_LIB__
<style>
  html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#222}
  #wrap{display:flex;height:100vh}
  #side{width:320px;overflow-y:auto;border-right:1px solid #e2e8f0;padding:14px 16px;box-sizing:border-box}
  #graph{flex:1}
  h1{font-size:16px;margin:0 0 2px}
  .sub{color:#888;font-size:11px;margin-bottom:12px}
  .stats{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
  .stat{background:#f1f5f9;border-radius:8px;padding:6px 10px;font-size:12px}
  .stat b{display:block;font-size:18px}
  h2{font-size:12px;text-transform:uppercase;color:#64748b;letter-spacing:.04em;margin:16px 0 6px}
  .wf,.sh{display:flex;align-items:center;gap:8px;padding:5px 6px;border-radius:6px;cursor:pointer;font-size:13px}
  .wf:hover,.sh:hover{background:#f1f5f9}
  .wf.active,.sh.active{background:#e2e8f0}
  .dot{width:11px;height:11px;border-radius:50%;flex:0 0 auto}
  .wf .nm,.sh .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .wf .ct,.sh .ct{color:#94a3b8;font-size:11px}
  .cat{font-size:12px;margin:8px 0 2px;color:#475569;font-weight:600}
  .nt{font-size:12px;padding:2px 0 2px 10px;color:#334155}
  .nt.unused{color:#cbd5e1}
  .nt .u{font-size:10px;color:#94a3b8}
  #reset{display:inline-block;font-size:11px;color:#2563eb;cursor:pointer;padding:4px 8px;border:1px solid #bfdbfe;border-radius:6px;margin:0 0 4px 0}
  #reset:hover{background:#eff6ff}
  #modes{margin-bottom:10px}
  .mode{display:inline-block;font-size:11px;padding:4px 8px;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer;margin:0 4px 4px 0;color:#475569}
  .mode.active{background:#1e293b;color:#fff;border-color:#1e293b}
</style>
</head>
<body>
<div id="wrap">
  <div id="side">
    <h1>__TITLE__</h1>
    <div class="sub">generated __GENERATED__ · filesystem snapshot</div>
    <div class="stats" id="stats"></div>
    <div id="modes">
      <span class="mode active" data-m="usage">Workflows &harr; types</span>
      <span class="mode" data-m="conn">Type connections</span>
      <span id="reset">&#8635; show all</span>
    </div>
    <h2>Shared node types (used by &gt;1 workflow)</h2>
    <div id="shared"></div>
    <h2>Workflows</h2>
    <div id="wflist"></div>
    <h2>Available node types</h2>
    <div id="catalog"></div>
  </div>
  <div id="graph"></div>
</div>
<script>
const DATA = __DATA__;
const UNIQUE_COLOR = "#cbd5e1";   // node used by a single workflow
const SHARED_COLOR = "#f59e0b";   // node shared across workflows

// ---- stats ----
const s = DATA.stats;
document.getElementById('stats').innerHTML = [
  ['workflows', s.n_workflows], ['node types used', s.n_types_used],
  ['shared types', s.n_shared], ['types available', s.n_node_types],
].map(([k,v])=>`<div class="stat"><b>${v}</b>${k}</div>`).join('');

// ---- lookups ----
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

// MODE 1 — bipartite: workflow hubs <-> node types (usage). Shows workflows AND sharing.
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

// MODE 2 — node-type connection graph (edges colored by workflow).
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
           stabilization:{iterations:250}},
  interaction:{hover:true, tooltipDelay:120}
};
let mode='usage';
let cur=buildUsage();
const net=new vis.Network(document.getElementById('graph'), cur, OPTS);

// Physics stays ON — the graph keeps gently settling/animating (same look as
// the "before hackathon" dashboard). No freeze after stabilization.

function setMode(m){
  mode=m;
  cur = (m==='usage') ? buildUsage() : buildConn();
  net.setData(cur);
  document.querySelectorAll('.mode').forEach(x=>x.classList.toggle('active', x.dataset.m===m));
  net.fit({animation:false});
}
document.querySelectorAll('.mode').forEach(el=>{ el.onclick=()=>setMode(el.dataset.m); });

// ---- highlight helpers (work in both modes) ----
function clearActive(){ document.querySelectorAll('.wf,.sh,.mode-x').forEach(x=>x.classList.remove('active')); }
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
  setMode(mode);                 // rebuild fresh = full colors restored
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
  // keep the type + every workflow (hub, usage mode) that uses it
  const t=typeById[name]; if(!t) return;
  const keep=[name];
  if(mode==='usage') t.wfids.forEach(w=>keep.push(HUB+w));
  recolor(keep, ed=> (mode==='usage') ? ed.to===name : (ed.from===name||ed.to===name));
}

// ---- shared node types panel (ranked by sharing) ----
const shared = document.getElementById('shared');
if(!DATA.shared_types.length){ shared.innerHTML = '<div class="nt unused">none yet</div>'; }
DATA.shared_types.forEach(t=>{
  const el=document.createElement('div'); el.className='sh';
  el.innerHTML = `<span class="dot" style="background:${SHARED_COLOR}"></span>`+
                 `<span class="nm" title="${t.wfnames.join(', ')}">${t.name}</span>`+
                 `<span class="ct">${t.n} wf</span>`;
  el.onclick = ()=>focusNode(t.name, el);
  shared.appendChild(el);
});

// ---- workflow list + focus/highlight ----
const wflist = document.getElementById('wflist');
DATA.workflows.forEach(w=>{
  const el = document.createElement('div');
  el.className='wf';
  el.innerHTML = `<span class="dot" style="background:${w.color}"></span>`+
                 `<span class="nm" title="${w.name}">${w.name}</span>`+
                 `<span class="ct">${w.n_nodes}n/${w.n_edges}e</span>`;
  el.onclick = ()=>focusWorkflow(w.id, el);
  wflist.appendChild(el);
});
document.getElementById('reset').onclick = resetView;

// ---- catalog by category, used/unused ----
const cat = document.getElementById('catalog');
const byCat = {};
DATA.catalog.forEach(c=>{(byCat[c.category]=byCat[c.category]||[]).push(c)});
Object.keys(byCat).sort().forEach(c=>{
  const h=document.createElement('div'); h.className='cat'; h.textContent=c; cat.appendChild(h);
  byCat[c].sort((a,b)=>a.name.localeCompare(b.name)).forEach(nt=>{
    const d=document.createElement('div');
    d.className='nt'+(nt.used?'':' unused');
    d.innerHTML = nt.name + (nt.used?'':' <span class="u">unused</span>');
    cat.appendChild(d);
  });
});
</script>
</body>
</html>
"""


def main(argv=None):
    here = Path(__file__).resolve().parent
    default_codes = here / ".." / ".." / "gui" / "workflow_backend" / "django-project" / "codes"

    ap = argparse.ArgumentParser(description="Build the NeuroWorkflow system dashboard (HTML).")
    ap.add_argument("--codes-dir", default=str(default_codes),
                    help="Path to the django-project/codes folder.")
    ap.add_argument("--output", default=str(here / "dashboard.html"),
                    help="Output HTML path.")
    ap.add_argument("--title", default="NeuroWorkflow system status",
                    help="Dashboard title.")
    args = ap.parse_args(argv)

    codes_dir = Path(args.codes_dir).resolve()
    if not codes_dir.is_dir():
        sys.exit(f"codes dir not found: {codes_dir}")

    model = build_model(codes_dir)
    html = render_html(model, args.title)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")

    st = model["stats"]
    print(f"Read: {codes_dir}")
    print(f"  workflows={st['n_workflows']}  node_types_used={st['n_types_used']}  "
          f"shared={st['n_shared']}  types_available={st['n_node_types']}")
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
