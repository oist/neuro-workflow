# Optimization in the GUI — design

The optimization engine (`docs/OPTIMIZATION.md`) and the `NW_Optimization` node are done and
tested from Python. This document is the design for the GUI half: how a study is stored, what the
generated code looks like, and the panel, canvas and backend pieces that make a search usable from
the editor. It replaces the earlier handoff note.

**The decision this design rests on.** The whole study — what to explore, what to hit, how to
search — lives on the `NW_Optimization` node, as ordinary node parameters (`explore`,
`objectives`, plus the algorithm settings). A user sees the study in one panel, an editor edits
one node, two studies over the same model are two nodes, and the neuron nodes stay clean. The
`optimizable` / `optimization_range` / `unit` fields a node author may put on a parameter are
defaults the panel prefills a range from; the engine reads the node.

Runnable examples of what the generator should emit: `notebooks/generated_optimization_example.py`
and `notebooks/generated_multiobjective_example.py` (and their `.ipynb`).

## The one idea to hold on to

**Optimization is not a step inside a workflow.** A workflow is a DAG that runs once; a search is a
loop that runs it many times, so the loop lives outside the graph. The `NW_Optimization` node does
not consume or produce data — it declares the study. Its presence on the canvas is the signal to
generate an optimization run instead of a single execution.

---

## 1. Where the study is stored

`NW_Optimization` carries three lists next to its algorithm settings. Values sit where every other
parameter value sits — `node.data.schema.parameters.<name>.default_value` — so saving the flow,
`update_node_parameter` from the MCP tools and code generation all work without new plumbing.

```jsonc
"explore": [
  {"node_id": "<React Flow node id>",   // GUI only; the engine ignores it
   "address": "conn.syn_weight",        // Node.parameter, or Node.parameter.key
   "low": 1.0, "high": 100.0,
   "unit": "pA",                        // optional; defaults to the parameter's unit
   "integer": false}                    // optional; inferred from the declared default
],
"objectives": [
  {"node_id": "<id of ana>",
   "name": "exc_rate",
   "measures": "ana.firing_rate_hz.exc",  // Node.output_port[.key]
   "goal": "in_range",                  // in_range | minimize | maximize
   "low": 40.0, "high": 50.0,           // in_range only
   "unit": "Hz"}
],
"enabled": true                          // added with the generator: one study generates at a time
```

**Names versus ids.** An address starts with the node's *instance name*, which is what the engine
resolves and what a Python user types. Instance names can be renamed in the editor, so the GUI
stores the React Flow `node_id` next to the address and the generator rewrites the address prefix
from the id at generation time. `FlowNode.id` survives renames; the emitted code carries only the
address. Python users omit `node_id`.

**Several studies.** Two `NW_Optimization` nodes are two studies — a coarse search and a fine one,
or different targets. `enabled` (a boolean parameter added together with the generator's
optimization mode) picks which one generates; generating with two enabled is refused with a clear
message rather than guessed at.

## 2. What the generator emits

If the canvas holds an enabled `NW_Optimization` node, generate an optimization run; otherwise
generate what it generates today. No second button. Everything above `workflow_builder.build()` is
unchanged — node creation, `configure()`, `add_node`, `connect`. Only the tail differs
(`code_generation_service.py`, the `success = workflow.execute()` line of the base template):

```python
    workflow = workflow_builder.build()

    opt = NW_Optimization("opt")          # configured, never add_node'd
    opt.configure(
        algorithm="cmaes", pop_size=16, max_generations=12, seed=1,
        results_path="results/optimization",
        explore=[{"address": "conn.syn_weight", "low": 1.0, "high": 100.0, "unit": "pA"}],
        objectives=[{"name": "exc_rate", "measures": "ana.firing_rate_hz.exc",
                     "goal": "in_range", "low": 40.0, "high": 50.0, "unit": "Hz"}],
    )
    spec = opt.build_spec(workflow)       # baseline run, validation, clipping to constraints
    print(spec.summary())
    result = optimize(workflow, spec=spec, results_path=opt.results_path())
    if result.best is None:               # only a run with no usable measurement has failed
        return 1
    print(result.configure_snippet())
```

Three things to know while implementing it:

- **The study block is a copy, with ids resolved.** The generator emits every `NW_Optimization`
  parameter, not only the modified ones — `_get_modified_parameters_for_configure` looks solely
  at `default_value` changes and is the wrong filter here — and rewrites each entry's address
  from its `node_id` to the node's current instance name, dropping `node_id` from the output.
- **The optimization node is configured but not added to the workflow.** It declares the study
  and takes no part in execution.
- **A search has no boolean outcome.** Not reaching the target is a *result*, not a failure — the
  run answered "nothing in these ranges hits the target range". Only a run where no trial
  produced a usable measurement (`result.best is None`) should exit non-zero.

**Measurables after every run.** In normal mode the generator appends, after `workflow.execute()`,
a dump of `discover_measurables(workflow)` to `results/measurables.json`. That file is what the
objective picker lists (section 3), so a user runs the workflow once and then picks a target
from what it actually produced.

## 3. The study panel

Opening an `NW_Optimization` node shows the study panel instead of the generic node modal
(branch on `data.label === "NW_Optimization"`, the way the brain-viewer node is recognised in
`calculationNode.tsx`). A Chakra `Modal` with three tabs: **Study | Run | Results**. Study is the
first thing to build; Run and Results come with the ledger endpoints (section 5).

### Study tab

1. **How to search.** `algorithm` as a `Select` (it declares `constraints.allowed_values`),
   `pop_size`, `max_generations`, `seed`, `options` (JSON), `results_path`. A budget line is always
   visible — "at most 16 × 20 = 320 runs, plus one baseline" — because with a simulation costing
   seconds that product is the number to think about first. When two or more objectives are
   declared and the algorithm is single-objective (`cmaes`), warn: "cmaes handles one objective;
   choose nsga2".
2. **Parameters to explore.** One row per entry: node · parameter (·key) · low–high · unit ·
   an `integer` badge · remove. *Add parameter* is a picker: choose a node on the canvas, then a
   parameter — numeric scalars, and the numeric keys of dict-valued parameters; strings, booleans
   and lists are not offered (`connection_rule` is an int but means "a rule"; `connections` is a
   list an address cannot index). The range is prefilled from the parameter's
   `optimization_range` (per key for a dict), else its `constraints.min/max`, else the current
   value ± 50 %. `constraints.min/max` are hard bounds: the inputs refuse a value outside them,
   since `configure()` would too. `unit` comes from the parameter's declared unit; `integer` is
   inferred from the declared default and can be overridden.
3. **Objectives.** One row per entry: name · measures · goal · low/high (for `in_range`) · unit ·
   remove. *Add objective* is a picker over `results/measurables.json` from the last run, showing
   each address with its value — `ana.firing_rate_hz.exc = 8.3` — because a port's contents can
   only be known from data. If the file does not exist: "Run the workflow once to list measurable
   values", and a text field, since typing an address by hand must keep working (an agent sets it
   that way).
4. **Footer.** One line mirroring `spec.summary()` — "2 parameters · 1 objective · cmaes · ≤ 321
   runs" — and *Generate code*. Validation: an empty explore list blocks generation; an empty
   objectives list warns; `low ≥ high` or a range outside constraints blocks saving.

Every edit is saved through the existing per-parameter endpoint,
`PUT /api/workflow/{id}/nodes/{node_id}/parameters/` with `parameter_key` set to `explore` or
`objectives` (the `updateParameter` path in `nodeDetailModal.tsx`). That keeps
`parameter_modifications` maintained, which the generator's other paths rely on.

### Show it on the graph

The panel is where the study is edited, but the canvas should show what is being tuned and
measured: the `NW_Optimization` node renders compactly (no ports; algorithm and "2 parameters ·
1 objective", a state chip while a run is going); a node with a parameter in `explore` gets an
orange **tune** badge in its header; a node a `measures` address resolves to gets a teal
**target** badge. Both are computed from the `NW_Optimization` nodes in the flow store by
`node_id` — display only, no edges.

### The node modal's per-parameter fields

`optimizable` / `optimization_range` stay in the generic node modal as "Author's default range
(used to prefill an optimization study)". The `is_objective` / `objective_range` rows are removed
from that modal: a target belongs to the study. Two existing rough edges to fix while there: the
range editor is a single-line `Input` 120 px wide receiving pretty-printed JSON (it wants a
`Textarea`), and `formatDataForDisplay` renders `{"V_th": [-60, -45]}` as `{V_th: -60,-45}`,
losing the brackets. `unit` must also reach the frontend: it is not extracted by
`python_analyzer.py` nor forwarded by `box/models.py:_convert_parameters` today, and the TS
`ParameterField.constraints` declares `options` where the Python side writes `allowed_values`.

## 4. Adopt the best configuration — an explicit action

The run does not write anything back: `apply_best()` changes the in-memory nodes inside that
process, and the `FlowNode` rows still hold what the user typed. Everything needed is in the
ledger:

```
<results_path>/<run_id>/status.json   →   best.params
                                          {"clamp.amp_na": 430.67, "conn.syn_weight": 7.84}
```

Each key is `instance_name.parameter[.key]`, so *Adopt best* is: resolve each address to a
`node_id` (through the study's own entries), show a confirmation dialog listing every change
(baseline → best, measured vs target), then call the per-parameter endpoint once per address
with `parameter_field="default_value"` — merging a dict key into the current dict — and prompt
to regenerate the code.

**With several objectives, let the user pick from the front.** A multi-objective run has no
single winner. `status.json` carries `pareto_front`, a list of configurations where improving
one objective would cost another, each with both its `measured` values and the `params` that
produced them. The natural interface is a plot: objective 1 against objective 2, one point per
front member, the target ranges shaded. Clicking a point adopts *that* configuration — the same
loop as the adopt button, using that member's `params`. With more than two objectives, a table
with one column per objective and a row to pick. The script falls back to the member whose
*worst* objective is closest to its target (`target_ranges_off`), which is a tie-break and
nothing more; deciding whether to favour firing rate over regularity is the scientist's call.

Keep it a deliberate button, never a side effect of the run. `run.json` records the baseline the
result is compared against; silently overwriting the user's parameters destroys that reference.

## 5. Watching and steering a run

Everything a UI needs is already written to files under `results_path/<run_id>/`, and monitoring
is reading them:

| file | for |
|---|---|
| `run.json` | the manifest: dimensions, objectives, algorithm, baseline. Written once |
| `trials.jsonl` | one line per evaluation, append-only |
| `status.json` | replaced each generation: `state`, counts, `best`, `progress.best_target_ranges_off_by_gen` |
| `control.json` | **written by the UI** to steer: `stop`, `pause`, `resume`, `inject` |

**Run tab.** The run itself is the ordinary ▶ button: the generated code is an optimization when
the study is on the canvas. The tab shows `state`, generation *i / N*, the `n_evals` /
`n_ok` / `n_failed` / `n_rejected` counts, `status.json.message`, a convergence curve from
`progress.best_target_ranges_off_by_gen` (in the objective's own unit when there is one
objective), and the latest trials (trial · generation · params · measured · miss · status).
Data comes from polling `GET /api/viewer/{project}/{results_path}/{run_id}/status.json` and
`trials.jsonl` every few seconds — the unauthenticated static route the figure manifest already
uses — in the manner of `runStatusPanel.tsx`. The engine's per-trial narration keeps flowing
through the existing log modal.

Pause / Resume / Stop write `control.json` through a new endpoint,
`POST /api/workflow/{id}/optimization/{run_id}/control/`, which validates the command, takes
`seq` as `status.control_ack.seq + 1` and resolves the run directory with the viewer's path
resolver (`app/workflow/viewer_tools/resolver.py`, which guards against traversal). `inject` sits
under an *Advanced* disclosure as JSON. A command is applied at most once: the loop acts only when
`control.seq > status.control_ack.seq`, and records the outcome in `control_ack`, so the panel
confirms delivery by polling the file it already watches. `stop` finishes the current generation;
stopping a wedged simulation mid-trial needs a kernel interrupt (section 6).

Label the two figures apart in the UI, because users confuse them: a miss is shown with its unit
(`off 8.2 ms`), the comparable figure as a multiple (`1.4x its target range`) and never with a
unit; say "target range", never "band"; always name the worst objective.

**Results tab.** A run selector over the history (a small
`GET /api/workflow/{id}/optimization/runs/` listing the run directories under `results_path`
with each `status.json` summary), then `stop_reason`, `message`, the best configuration
(address · baseline → best · measured vs target) and *Adopt best* — or the Pareto plot — from
section 4.

**Finding the run.** The generated script prints `Optimization run: <dir>`; `run_attribution.py`
already scans stdout for node markers and can turn that line into an `optimization_run_started`
event on the SSE stream, so the Run tab attaches to the right ledger as soon as it exists.

Charts are inline SVG components (a convergence line and a Pareto scatter); the frontend has no
charting dependency and these two do not justify one.

## 6. Backend pieces, by phase

| phase | piece | where |
|---|---|---|
| 1 | generator optimization mode (section 2), the `enabled` check, the measurables dump | `app/workflow/code_generation_service.py` |
| 1 | `unit` through AST extraction and serialization; `allowed_values` in the TS type | `app/box/services/python_analyzer.py`, `app/box/models.py`, `views/home/type.ts` |
| 2 | `GET …/optimization/runs/`, `POST …/optimization/{run_id}/control/` | new views, reusing `viewer_tools/resolver.py` |
| 2 | `Optimization run:` marker → SSE `optimization_run_started` | `app/workflow/run_attribution.py` |
| 3 | `POST …/run/stop/` (kernel interrupt; needs the kernel id kept per running workflow) | `app/workflow/views.py`, `jupyter_execution_service.py` |

A pre-existing gap matters for long runs: aborting the SSE fetch stops the stream but leaves the
kernel — and the loop — running, because the response's `finally` closes the event loop before
the execution service's `_delete_kernel` can run. A search that may run for hours needs that
fixed, and needs the kernel id tracked so an interrupt can reach it.

Frontend layout (`gui/workflow_frontend/src/`): `views/home/components/optimization/` for the
modal, its three tabs, the two pickers and the two charts; `stores/optimizationStore.ts` for run
state, kept out of the flow store for the same reason `runStore.ts` is (undo history and the
node-persistence PUT); `api/optimizationApi.ts` for ledger reads, run listing and control;
`utils/studyAddress.ts` for enumerating numeric parameters, building addresses and resolving
`node_id` to an instance name.

## 7. The kernel image needs Optuna

Workflows run in the nest kernel image. `NW_Optimization` defaults to `algorithm="cmaes"`, so
without `optuna` and `cmaes` an optimization generated in the GUI stops at an `ImportError`. The
line is in the pip block of `gui/workflow_backend/django-project/neuroworkflow/Dockerfile.nest`;
the image has **not been rebuilt**, so that is the first thing to do before the GUI half can be
exercised end to end. `cmaes` is a separate package from `optuna`, needed by the CMA-ES sampler
specifically; `nsga2` / `nsga3` / `tpe` need only `optuna`. Library users outside the image:
`pip install -e ".[optimization]"`.

`NW_Optimization.py` is synced into `gui/workflow_backend/django-project/codes/nodes/optimization/`
together with the `codes/neuroworkflow/optimization/` package generated code imports from.
`JointOptimizationNode` was deleted: it was an in-graph grid search, i.e. the design this
replaces. A saved project that placed it will no longer find it.

## 8. Two pre-existing generator bugs worth fixing while in there

**Instance names are not sanitized.** `var_name = instance_name` in `_generate_node_code_block`:
a node named `Excitatory Pop` generates `Excitatory Pop = NW_Population("Excitatory Pop")` — a
`SyntaxError` that kills the whole script. `_sanitize_variable_name()` exists but is applied to
node *ids*, never to `instanceName`.

**Booleans are emitted as integers.** `_convert_parameter_value` treats `True` as numeric, so
`plot_raster=True` is written as `plot_raster=1`. It works, being truthy, but the generated code
misstates the type.

## Recommended order

1. **Rebuild the nest image** — until then every optimization fails on import
2. `NW_Optimization` in the palette (files already synced; check the two list parameters show
   up as empty lists)
3. Phase 1: the Study tab, the generator's optimization mode, `unit` plumbing, canvas badges —
   configure → generate → ▶ → watch the log is then usable end to end
4. Phase 2: Run and Results tabs, the control and run-listing endpoints, Adopt best
5. Phase 3: kernel interrupt, polish for several studies
