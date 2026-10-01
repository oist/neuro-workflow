# Optimization in the GUI — handoff

The optimization engine, the schema fields it reads and the node that configures it are done and
tested from Python. What is missing is the GUI half. This is the list, with the files involved.

Background reading: `docs/OPTIMIZATION.md` (how the engine works). A runnable example of the code
the generator should produce: `notebooks/generated_optimization_example.py`
and its `.ipynb`.

**Status.** The library is done and tested. `optuna`/`cmaes` are declared in `Dockerfile.nest`;
rebuild the image before running a search from the GUI. The GUI half (Phase 1: the study panel,
the generator's optimization mode, the canvas badges) is built in the follow-up PR on this branch.

## The one idea to hold on to

**Optimization is not a step inside a workflow.** A workflow is a DAG that runs once; a search is a
loop that runs it many times, so the loop lives outside the graph. The `NW_Optimization` node does
not consume or produce data — it declares *how* to search. Its presence on the canvas is the signal
to generate an optimization run instead of a single execution.

## The settled design

Agreed with the library author: **the library's code representation does not change, and the GUI
aggregates the study on the `NW_Optimization` node.** Concretely:

| what | where it lives | what the generator emits |
|---|---|---|
| how to search | the node's own parameters (`algorithm`, `pop_size`, `max_generations`, `seed`, `options`, `results_path`) | `opt.configure(...)`, then `build_spec(workflow, opt.algorithm_config())` |
| what to explore | the target node's own parameter fields `optimizable` / `optimization_range` / `unit` — the fields the engine reads, one source of truth | `node.NODE_DEFINITION.parameters["p"].optimizable = True` (+ range, + unit) after `build()` |
| what to hit | GUI-only `data.study.objectives` on the `NW_Optimization` FlowNode: `{node_id, port, key?, name, goal, low?, high?, unit?}` | `spec.add_objective(name=..., measures="<var>.<port>[.<key>]", ...)` after `build_spec()` |

An objective is keyed by the React Flow `node_id`, and the generator resolves it to the generated
variable name (which is also the node's instance name in the script), so renaming a node cannot
break an address. Objectives a node author declared on a parameter (`is_objective`, e.g.
`NW_Population.mean_firing_rate`) keep working: `build_spec()` discovers them, and the study panel
lists them read-only. `goal="minimize"/"maximize"` is reachable only through the study, since the
schema route can express only a target range.

One `NW_Optimization` node on the canvas switches the generated script to a search; none keeps
today's script byte for byte; two are refused with a clear message (the explore flags live on the
model's nodes, so two studies over one model cannot coexist in this representation).

The panel opens in place of the generic node panel (`views/home/components/optimization/`), the
pure helpers are in `views/home/utils/studyAddress.ts`, and the generator side is
`_optimization_tail` and friends in `app/workflow/code_generation_service.py`, tested in
`tests/test_code_generation_optimization.py`.

---

## 1. The optimization panel

*Built in Phase 1 as described under "The settled design"; the notes below are the original brief.*

Editing `NW_Optimization` should open a panel with three parts. The first is ordinary parameters
(`algorithm` is a dropdown already, since it declares `allowed_values`). The other two are lists.

### Parameters to explore

Each entry: an **address**, a **range**, and a unit for display.

| parameter shape | address | range |
|---|---|---|
| scalar | `conn.syn_weight` | `[1.0, 100.0]` |
| key inside a dict | `exc.nest_params.I_e` | `[0.0, 400.0]` |

Adding one is: pick a node, pick a parameter (or a key within a dict parameter), then set the range.
Prefill it from the node's own declaration — `optimization_range` if present, otherwise the `min`/`max`
in `constraints` — and carry the `unit` across. Never propose a range outside `constraints`: those are
the model's hard bounds and `configure()` rejects values beyond them.

Only numeric things are tunable. Strings, booleans and lists are not: `connection_rule` is an int but
means "a rule", and `connections` is a list, which addresses cannot index. A sensible picker offers
scalar float/int parameters and the numeric keys of dict parameters.

### Objectives

Each entry: a **target parameter**, a **band**, and a **`measures` address**.

`measures` is the field that makes a target usable — without it a band is not interpretable, since
"firing rate" means different things in different nodes. Its value is an address into the run's
outputs, made of three parts:

```
ana . firing_rate_hz . exc
 |          |            |
 |          |            +-- a key inside that port's dict; here the population name,
 |          |                which comes from NW_Population's pop_name parameter
 |          +--------------- an output port of that node
 +-------------------------- the node's instance name
```

**It needs a picker, not a text box.** Valid values can only be known from *data*, because a port's
contents depend on what a run produced. After a baseline run, `discover_measurables(workflow)` returns
every numeric leaf with its current value — `ana.firing_rate_hz.exc = 8.3` — so the picker is that
list. Typing an address by hand must keep working, since an agent sets it that way.

Two or three objectives are legitimate and switch the algorithm to `nsga2`/`nsga3`, which return a
Pareto front instead of one answer. A single-objective algorithm refuses a multi-objective study
rather than silently collapsing it into a weighted sum.

### Show it on the graph

The panel is where the study is edited, but the canvas should show what is being tuned and measured:
a badge on a node whose parameter is in the exploration list, and a badge next to the **port** that a
`measures` address resolves to. Users think *"this output is my objective"*, so drawing it there costs
nothing and prevents hunting.

### If you also expose the per-parameter fields

They are the node author's defaults, so exposing them in the node modal is optional — but if you do,
the plumbing is nearly there. `update_node_parameter` already writes any field by name
(`app/workflow/views.py:645`); what is missing is `measures` and `unit` in the editable union at
`nodeDetailModal.tsx:173`, in the parameter type at `views/home/type.ts:32`, in the passthrough at
`box/models.py:204`, and in the AST extraction at `python_analyzer.py:466`.

Two existing rough edges to fix if you render `optimization_range` there: the editor is a single-line
`Input` 120px wide receiving pretty-printed JSON (it wants a `Textarea`), and `formatDataForDisplay`
renders `{"V_th": [-60, -45]}` as `{V_th: -60,-45}`, losing the brackets.

---

## 2. Add `NW_Optimization` to the palette

`NW_Optimization.py` **is already synced** into
`gui/workflow_backend/django-project/codes/nodes/optimization/`, together with the four changed
`NW_*` nodes and — new — the whole `codes/neuroworkflow/optimization/` package, which is what
generated code imports `build_spec` and `optimize` from. Nothing else needs copying.

No ports, no process steps — a workflow containing it executes exactly as it would without it.
`algorithm` carries `constraints={"allowed_values": [...]}`, so it renders as a dropdown for free.

`JointOptimizationNode` was deleted: it was an in-graph grid search, i.e. the design this replaces.
A saved project that placed it will no longer find it.

### The kernel image needs Optuna — the line is in this PR, the rebuild is not

Workflows run in the nest kernel image. `NW_Optimization` defaults to `algorithm="cmaes"`, so
without `optuna` and `cmaes` an optimization generated in the GUI stops at an `ImportError`.

`optuna cmaes` is now in the pip block at
`gui/workflow_backend/django-project/neuroworkflow/Dockerfile.nest:128`, so **the image needs a
rebuild** before the GUI half can be exercised. `cmaes` is a separate package from `optuna`, needed
by the CMA-ES sampler specifically; `nsga2`/`nsga3`/`tpe` need only `optuna`. Optuna also pulls in
alembic, colorlog, sqlalchemy and tqdm. The build has not been run, so the first rebuild is also the
first test of that line. Library users outside the image: `pip install -e ".[optimization]"`.

The line as committed:

```dockerfile
	pip install --no-cache-dir nestml nest-desktop "pandas>=2.2,<3.0" jupyterlab notebook \
		jupyterhub tvb-library tvb-framework statsmodels hdf5storage bmtk "numpy<2.5" \
		httpx ipywidgets optuna cmaes \
		"siibra==1.0.1a15" "traitlets==5.14.3" && \
```

---

## 3. One code-generation button

*Built in Phase 1. Objectives are emitted with `spec.add_objective()` rather than the schema
assignments shown below; the explore block is as shown.*

If the canvas contains an `NW_Optimization` node, generate an optimization run; otherwise generate
what it generates today. No second button.

Everything above `workflow_builder.build()` is unchanged — node creation, `configure()`, `add_node`,
`connect`. Only the tail differs. Today (`code_generation_service.py:328`):

```python
    success = workflow.execute()
```

With optimization:

```python
    # Parameters marked optimizable in the editor
    clamp.NODE_DEFINITION.parameters["amp_na"].optimizable        = True
    clamp.NODE_DEFINITION.parameters["amp_na"].optimization_range = [100.0, 1000.0]
    clamp.NODE_DEFINITION.parameters["amp_na"].unit               = "nA"

    conn.NODE_DEFINITION.parameters["syn_weight"].optimizable        = True
    conn.NODE_DEFINITION.parameters["syn_weight"].optimization_range = [1.0, 100.0]

    # Parameters marked as objectives in the editor
    exc.NODE_DEFINITION.parameters["mean_firing_rate"].is_objective    = True
    exc.NODE_DEFINITION.parameters["mean_firing_rate"].objective_range = [40.0, 50.0]
    exc.NODE_DEFINITION.parameters["mean_firing_rate"].measures        = "ana.firing_rate_hz.exc"

    spec   = build_spec(workflow, opt.algorithm_config())
    result = optimize(workflow, spec=spec, results_path=opt.results_path())
```

Copy the shape from `generated_optimization_example.py` — it runs.

Three things to know while implementing it:

- **The declarations are a translation, not a copy.** The study is stored on the optimization node,
  and the generator turns each entry into the assignment on the owning node shown above. Emitting the
  same shape a hand-written notebook uses means one code shape to understand, and `build_spec()` still
  performs its validation and range-clipping. The existing "changed parameters only" logic does not
  help here: `_get_modified_parameters_for_configure` (`code_generation_service.py:502`) looks solely
  at `default_value`.
- **The optimization node is configured but not added to the workflow.** It declares how to search
  and takes no part in execution.
- **A search has no boolean outcome.** Not reaching the target is a *result*, not a failure — the run
  answered "nothing in these ranges hits the band". Only a run where no trial produced a usable
  measurement (`result.best is None`) should exit non-zero.

---

## 4. Adopt the best configuration — an explicit action

The run does not write anything back: `apply_best()` changes the in-memory nodes inside that
process, and the `FlowNode` rows still hold what the user typed.

Everything needed is in the ledger:

```
<results_path>/<run_id>/status.json   →   best.params
                                          {"clamp.amp_na": 430.67, "conn.syn_weight": 7.84}
```

Each key is `instance_name.parameter[.key]`, so an "adopt" button is: resolve the instance name to a
`node_id`, call `update_node_parameter(..., parameter_field="default_value")` once per address, then
regenerate the code.

### With several objectives, let the user pick from the front

A multi-objective run has no single winner. `status.json` carries `pareto_front`, a list of
configurations where improving one objective would cost another, each with both its `measured`
values and the `params` that produced them.

The natural interface is a plot: objective 1 against objective 2, one point per front member, the
target bands shaded. Clicking a point adopts *that* configuration — the same
`update_node_parameter` loop as the adopt button, using that member's `params` instead of
`best.params`.

This is where the choice belongs. The script falls back to the member whose *worst* objective is
closest to its target, each miss sized against its own target range (`target_ranges_off`), which is
a tie-break and nothing more;
deciding whether to favour firing rate over regularity is the scientist's call, and a picture is how
it gets made.

Keep it a deliberate button, never a side effect of the run. `run.json` records the baseline the
result is compared against; silently overwriting the user's parameters destroys that reference and is
very hard to explain afterwards.

---

## 5. A panel over the ledger (later)

Everything a UI needs is already written to files under `results_path/<run_id>/`, and monitoring is
just reading them:

| file | for |
|---|---|
| `run.json` | the manifest: dimensions, objectives, algorithm, baseline. Written once |
| `trials.jsonl` | one line per evaluation, append-only |
| `status.json` | replaced each generation: `state`, counts, `best`, `progress.best_target_ranges_off_by_gen` |
| `control.json` | **written by the UI** to steer: `stop`, `pause`, `resume`, `inject` |

`progress.best_target_ranges_off_by_gen` exists so a convergence curve can be drawn from
`status.json` alone, without parsing every trial. Its unit is **target ranges missed by**: how far
the worst objective landed outside its target, divided by the width of that target range, so `1.4`
means the miss is 1.4 times as wide as the range the user asked for and `0` means every target is
met. Objectives in Hz and in ms cannot share an axis otherwise. The per-trial misses, each in its
own unit, are in `trials.jsonl` under `fitness`.

Label the two apart in the UI, because users confuse them: a miss is shown with its unit
(`off 8.2 ms`), the comparable figure as a multiple (`1.4x its target range`) and never with a unit.
`status.json.message` is a ready-made one-line summary including the target.

A command is applied at most once: the loop acts only when `control.seq > status.control_ack.seq`,
and records the outcome in `control_ack` — so the writer confirms delivery by polling the same file
it already watches. No locking, no sockets.

Reuse the viewer's path resolver (`app/workflow/viewer_tools/resolver.py`) — `existing_project_dir()`
plus `_resolve_explicit()`, which already guards against path traversal.

---

## 6. Two pre-existing bugs worth fixing while in there

**Instance names are not sanitized.** `code_generation_service.py:388,406`:

```python
var_name = instance_name
code_block = f"""    {var_name} = {label}("{var_name}")"""
```

A node named `Excitatory Pop` generates `Excitatory Pop = NW_Population("Excitatory Pop")` — a
`SyntaxError` that kills the whole script. `_sanitize_variable_name()` exists but is applied to node
*IDs*, never to `instanceName`. A rule was added to the chat system prompt
(`app/chat/services/chat_orchestrator.py:12`) so the assistant avoids it, but that does not help a
human typing in the modal.

**Booleans are emitted as integers.** `_convert_parameter_value` treats `True` as numeric, so
`plot_raster=True` is written as `plot_raster=1`. It works, being truthy, but the generated code
misstates the type.

---

## Recommended order

1. **Rebuild the nest image** — the `optuna cmaes` line is committed, but until the image is rebuilt
   every optimization fails on import, so nothing downstream can be tested end to end
2. `NW_Optimization` in the palette — nothing can be declared without it (files already synced)
3. The optimization panel: exploration list and objectives — **done**
4. Generator optimization mode — **done**
5. Adopt-best button
6. The ledger panel (run listing, `control.json` endpoint, kernel interrupt for a wedged trial)

Steps 1-4 make the loop usable end to end; a user can then configure a search, generate it and run
it. Step 5 closes it back to the editor. Step 6 is what turns a long run from opaque into watchable.
