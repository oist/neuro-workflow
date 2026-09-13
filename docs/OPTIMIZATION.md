# Parameter optimization

Tune parameters of an existing workflow until a measured quantity lands in a target band.

A workflow is a DAG: it runs once, feed-forward, and produces outputs. Optimization is a loop —
propose parameters, run the DAG, measure, propose again — and a loop cannot live inside an acyclic
graph. So the optimizer sits **outside** the workflow and calls it repeatedly:

```
ask → configure nodes → workflow.execute() → read measured value → fitness → tell → repeat
```

**The workflow is the objective function.** It is built once and reconfigured between trials; it is
never rebuilt.

## Quick start

```python
from neuroworkflow.optimization import AlgorithmConfig, build_spec, optimize

spec = build_spec(workflow, AlgorithmConfig(name="cmaes", pop_size=8, max_generations=10, seed=1))
print(spec.summary())                      # review before spending simulation time

result = optimize(workflow, spec=spec, results_path="./results/optimization")
print(result.configure_snippet())          # ready-to-paste best configuration
```

`build_spec()` runs the workflow once at its current values. That run is the baseline every result
is compared against, and it is what every `measures` address is resolved against.

Optuna algorithms — `cmaes` (the default), `tpe`, `nsga2`, `nsga3` — need
`pip install -e ".[optimization]"`. Only `algorithm="random"` runs on numpy alone, and it is
uniform sampling: a baseline to beat and a way to smoke-test the loop, never a substitute for a
search. The Jupyter GUI code generator and the nest-kernel image extra are **not** in this PR, so
an optimization generated in the GUI fails on import until `optuna` and `cmaes` are added to
`Dockerfile.nest`. Where the study is declared (per-parameter fields vs the `NW_Optimization` node)
is still unsettled — see `docs/OPTIMIZATION_GUI_HANDOFF.md`.

## Declaring what to optimize

The spec is read from the node schemas. Two declarations matter, both on `ParameterDefinition`.

### What to explore

```python
"amp_na": ParameterDefinition(
    default_value=0.15, unit="nA",
    constraints={"min": -1000.0, "max": 1000.0},   # hard validity — never proposed outside
    optimizable=True,
    optimization_range=[0.0, 800.0],               # where to search, within the constraints
),
```

`constraints` is the fence: what the model considers valid at all, enforced by `configure()`.
`optimization_range` is the patch of ground inside it that this study wants searched. A range
reaching past the constraints is clipped, and the clip is recorded in the spec.

For a **dict-valued parameter**, give a range per key. Each key becomes its own search dimension:

```python
"nest_params": ParameterDefinition(
    default_value={"C_m": 250.0, "V_th": -55.0, "I_e": 0.0},
    optimizable=True,
    optimization_range={"I_e": [0.0, 800.0], "V_th": [-60.0, -45.0]},
),
```

Tunable keys are read from the node's **live** value, so a key added through `configure()` is just
as tunable as one in the declared default.

### Whole-number parameters

A count of neurons is not a continuous quantity, but the samplers are continuous — CMA-ES will
propose `N = 2500.37`. The engine maps a proposal onto the axis **before** `configure()` sees it, so
the value that runs is the value the ledger records:

```
proposal 2500.37   →   N = 2500   →   configure(), the ledger, configure_snippet()
```

Nothing inside the algorithm changes; rounding is clamped inside the declared range so a proposal at
an edge cannot be rounded out of bounds.

An axis is treated as whole-number when the parameter's **declared default** is an `int` — `N = 2500`
already says the parameter counts things. The declared default is used rather than the current value
on purpose: that a user configured `amp_na=200` (an int literal for a quantity declared as `0.15` nA)
says nothing about the parameter's nature, and inferring from it would silently restrict that search
to whole nanoamps. Override either way with `constraints={"integer": True}` (or `False`), and for a
dict parameter per key: `constraints={"integer": {"n_syn": True}}`.

`spec.summary()` marks such an axis, so it is visible before a run:

```
dimensions: 2
    exc.N                                    [2000.0, 3000.0] integer
    exc.nest_params.I_e                      [0.0, 400.0] pA
```

A range containing no whole number at all (`[10.2, 10.8]`) is reported in `spec.skipped` rather than
searched.

### What to hit

```python
"mean_firing_rate": ParameterDefinition(
    default_value=10.0, unit="Hz",
    is_objective=True,
    objective_range=[8.0, 12.0],                    # reached anywhere inside this band
    measures="Analysis.firing_rate_hz.v1",          # the measurement it is compared against
),
```

A target declares a desired value; `measures` says **where the achieved value is read from**.
Without it the band is not interpretable — "firing rate" means different things in different nodes —
so a target without `measures` is skipped and reported.

### Addresses

Everything is addressed by a dotted path starting with the **node name** — the string passed to the
constructor, `node.name`, not the Python variable:

| kind | shape | example |
|---|---|---|
| parameter to tune | `Node.parameter[.key]` | `Population.nest_params.V_th` |
| value to measure | `Node.output_port[.key]` | `Analysis.firing_rate_hz.v1` |

`discover_measurables(workflow)` lists every numeric leaf a run actually produced — the menu a
`measures` address can point at. It reads the data, not a declaration, so it cannot go stale.

Setting a dict key goes through `configure()` with the whole dict rebuilt, never by mutating in
place, so a rejected value cannot leave the node half-changed.

## Algorithms

```python
from neuroworkflow.optimization import available
available()   # ['cmaes', 'nsga2', 'nsga3', 'optuna_random', 'random', 'tpe']
```

| name | method | objectives | needs | use it when |
|---|---|---|---|---|
| `random` | uniform sampling | any | — | smoke-testing the loop, or a baseline to beat |
| `cmaes` | CMA-ES | **1 only** | `optuna`, `cmaes` | default for continuous single-objective search |
| `tpe` | Tree-structured Parzen Estimator | any | `optuna` | trials are expensive and the budget is small |
| `nsga2` | NSGA-II | any (built for 2–3) | `optuna` | 2 or 3 objectives; returns a Pareto front |
| `nsga3` | NSGA-III | any (built for 3+) | `optuna` | more than 3 objectives, where NSGA-II degrades |
| `optuna_random` | Optuna's random sampler | any | `optuna` | comparing fairly against the Optuna backends |

Everything except `random` needs Optuna:

```bash
pip install optuna cmaes      # 'cmaes' is only required by the cmaes algorithm
```

Without it those names still appear in `available()` and raise an `ImportError` naming the install
command. Restart the kernel after installing.

**Single-objective backends refuse a multi-objective spec.** `cmaes` raises rather than collapsing
objectives into a weighted sum, because the weights would be an invented scientific judgement. Use
`nsga2` / `nsga3` and read the Pareto front instead.

### Configuration

`AlgorithmConfig` is plain data — editable by hand, by an agent, or loaded from JSON.

| field | default | meaning |
|---|---|---|
| `name` | `"cmaes"` | one of the names above; needs Optuna |
| `pop_size` | `16` | candidates per generation, i.e. workflow runs per generation |
| `max_generations` | `20` | generation budget |
| `seed` | `None` | makes a run reproducible |
| `options` | `{}` | extra keyword arguments forwarded verbatim to the underlying sampler |

At most `pop_size × max_generations` executions, plus one baseline. With a simulation costing
seconds, that product is the number to think about first.

For `nsga2` / `nsga3`, `pop_size` is also passed as the sampler's `population_size`.

`options` is merged into the sampler's constructor arguments, so anything the underlying Optuna
sampler accepts can be set without changing this package:

```python
AlgorithmConfig(name="cmaes", pop_size=8, seed=1, options={"sigma0": 0.2})
AlgorithmConfig(name="tpe",   pop_size=8, options={"n_startup_trials": 20})
```

The names are Optuna's — see its sampler documentation. A wrong option surfaces as a `TypeError`
when the run starts, not silently.

### Choosing one

- **1 objective, continuous parameters** → `cmaes`. It adapts a covariance model, so it handles
  correlated parameters (`I_e` trading off against `V_th`) far better than independent sampling.
- **1 objective, few trials affordable** → `tpe`.
- **2–3 objectives** → `nsga2`. There is no single best answer; it returns the Pareto front, the
  configurations where improving one objective costs another.
- **More than 3** → `nsga3`.
- **Checking the setup** → `random`. If random search hits the target immediately, the ranges are
  too generous or the band too wide.

### Adding one

An Optuna sampler is one line in `_OPTUNA_SAMPLERS` (`optimization/optimizers.py`):

```python
"gp": ("GPSampler", True, {}),      # name -> (sampler class, multi-objective?, default kwargs)
```

Anything else subclasses `Optimizer` and registers:

```python
from neuroworkflow.optimization import Optimizer, register_optimizer

class MySearch(Optimizer):
    supports_multi_objective = False
    def ask(self): ...                        # -> list of {address: value}
    def tell(self, candidates, fitnesses): ...
    def enqueue(self, params): ...            # optional: seed a candidate

register_optimizer("mysearch", MySearch)
```

Fitness is always a **list** — one entry per objective, in spec order — and always **minimized**;
the engine converts a target band into a distance before telling. A failed or rejected trial is
reported as `None`, not as a large number, so a backend can mark it failed rather than let an
invented penalty distort its model.

### Comparing objectives that are not in the same unit

Each entry of `fitness` is **how far that objective missed, in its own unit** — 14.1 Hz past a
firing-rate target, 8.2 ms past an interval target. Those numbers must never be added: the sum has
no unit and no meaning, and it silently favours whichever objective happens to use bigger numbers.

The search itself does not care, and needs no rescaling: a multi-objective sampler ranks by **Pareto
dominance**, which compares one objective against the same objective, and NSGA-II rescales each
objective internally before computing crowding distance. Mixed units are only a *reporting* problem
— ranking trials in a table, drawing a convergence curve, and picking one configuration to apply all
need a single number.

That number is **how many target ranges the worst objective missed by**:

```
                            how far this objective missed
target ranges missed by  =  ─────────────────────────────
                             the width of its target range

reported value = the largest of those, across the objectives
```

A target of 40–50 Hz is a range 10 Hz wide. A trial measuring 25.9 Hz missed it by 14.1 Hz, so
14.1 / 10 = **1.4**: the miss is 1.4 times as wide as the range that was asked for. `0` means the
value landed inside. There is no upper bound.

Dividing by the width of the range is not an invented weighting: by asking for 40–50, the user
already stated that a 10 Hz spread is acceptable, so the width is the yardstick they themselves set.
For a `minimize`/`maximize` goal, which has no range, the baseline value is used instead.

It is the **largest**, not the sum — what the literature calls the Chebyshev or achievement
scalarizing function. With target ranges you want *every* objective met, so a trial is only as good as
whatever it is doing worst, and `0` can only happen when all of them are inside. Unlike a weighted
sum, this can also single out configurations anywhere on the Pareto front, not only on its convex
hull.

Both numbers are on every trial row, and the output distinguishes them: a miss is printed with its
unit (`off 8.2 ms`), the comparable figure always as `1.4x its target range`, and it names the
objective that is worst. Ranking, `result.best` and `progress.best_target_ranges_off_by_gen` all use
it; what is told to the optimizer is still the per-objective list of raw misses.

With a single objective none of this applies — the output stays in that objective's own unit, and
the multiple is not shown.

## Running

```python
result = optimize(workflow, spec=spec, results_path="./results/optimization",
                  reject_fn=None, per_trial_results=True, verbose=True)
```

**`per_trial_results` (default `True`)** gives each trial its own directory under the run and points
`results_path` — in the workflow context and in every node's context — at it. Any workflow whose
nodes write files needs this: sharing one directory loses each trial's output, and rebuilding a file
in place can fail outright while a handle from the previous trial is still open.

**Reusing expensive artifacts across trials.** A node can tell the engine that something it
writes under `results_path` survives a trial unchanged:

```python
class NW_SimConfig(Node):
    REUSABLE_PATHS = ("network",)      # relative to results_path
```

Before each trial the engine copies those directories in from the baseline run, so a node that
recognises its own earlier work can skip redoing it — for the `NW_*` builders, that means the
SONATA network is built once instead of once per trial. The engine does not know what the
directory holds; it only copies what nodes declare, and a node that declares nothing is
unaffected. Copies rather than sharing one directory: a trial that *does* change the artifact
rebuilds it inside its own directory, which is exactly the isolation this provides.

**`reject_fn(workflow, measured) -> str | None`** is the dynamics check. A scalar fitness cannot tell
a healthy network from a pathological one that merely averages to the right number, so this hook
returns a reason to reject a trial whose number is right but whose behaviour is wrong.

`measured` holds **everything numeric the trial produced**, addressed as `node.port[.key]` — the same
mapping `discover_measurables()` returns — plus each objective under its declared name. That matters:
the signal a dynamics check needs is usually *not* the objective.

```python
def reject(workflow, measured):
    cv = measured.get("ana.isi_stats.exc.cv")      # not an objective, still available
    if cv is not None and cv < 0.1:
        return f"clock-like firing (ISI CV {cv:.2f}) — not a plausible regime"
    return None
```

A rejected trial is recorded with `status: "rejected"` and its reason, and reported to the optimizer
as a failure rather than with an invented penalty.

The run stops when a candidate lands inside every target band, when the generation budget is
exhausted, or when a `stop` command arrives. The reason is always stated.

## The ledger

Everything the engine knows is written to plain files under `results_path/<run_id>/`. Monitoring is
reading them; steering is writing one. No sockets, no queues.

```
run.json        manifest, written once
trials.jsonl    one line per evaluation, append-only
status.json     replaced each generation, the cheap poll
control.json    written by a human or agent to steer the loop
trials/0001/    that trial's results_path (with per_trial_results)
```

`run_id` is `opt_<YYYYMMDD>_<HHMMSS>`, suffixed `_2`, `_3` … if that directory already exists, so
two runs starting in the same second cannot share a ledger.

**Write discipline.** Whole-file writes go through a temp file and `os.replace`, so a reader never
sees half a document. `trials.jsonl` is appended and flushed per line, so the only partial read
possible is a truncated final line — readers skip an unparseable last line. One writer per file: the
loop owns `run.json` / `trials.jsonl` / `status.json`, and only agents or a UI write `control.json`.

### `run.json`

Written once, before the first evaluation: `schema_version`, `run_id`, `created_at`,
`workflow_name`, then the whole spec — `dimensions`, `objectives`, `algorithm`, `locked_protocol`,
`baseline`, `skipped`.

`locked_protocol` and `baseline` are the comparability guarantee. Everything not in the exploration
space is frozen for the run; if the target or the protocol changes, fitnesses stop being comparable,
so that is a **new run**, not an edit to this one.

### `trials.jsonl`

One object per line, appended the moment a trial finishes:

```json
{"schema_version": 1, "trial": 1, "generation": 1, "candidate": 0,
 "started_at": "2026-08-12T12:34:11", "duration_s": 12.4, "source": "ask",
 "params": {"Population.nest_params.I_e": 243.1},
 "results_path": ".../trials/0001",
 "measured": {"Population.mean_firing_rate": 9.6},
 "fitness": [0.0], "target_ranges_off": 0.0,
 "status": "ok", "reject_reason": null, "error": null}
```

- **`fitness` is what was told to the optimizer**, not merely what was measured — recording it is
  what makes a run replayable. It is `null` for failed and rejected trials. One raw distance per
  objective, each in that objective's own unit.
- **`target_ranges_off`** is the same trial reduced to one comparable number: how many target
  ranges the worst objective missed by (see above). `0` means every target is met. Used for ranking
  and display only, never told to the optimizer.
- **`status`** is `ok`, `failed` (simulator error, or the measured value was not a number) or
  `rejected` (the number was in range but `reject_fn` refused the dynamics). `rejected` is
  first-class, not a footnote: it is the whole reason a human or agent is in the loop.
- **`source`** is `ask` for algorithm proposals, `inject` for seeded candidates.

### `status.json`

Replaced at the end of every generation, and once more when the run ends, so a poller always sees it
finish:

```json
{"state": "done", "generation": 8, "max_generations": 8,
 "n_evals": 48, "n_ok": 45, "n_failed": 1, "n_rejected": 2,
 "best": { ...the winning trial row... },
 "pareto_front": [],
 "progress": {"best_target_ranges_off_by_gen": [1.62, 1.28, 0.78, 0.0], "gens_since_improvement": 0},
 "control_ack": {"seq": 1, "command": "stop", "result": "stopping"},
 "stop_reason": "a candidate is inside every target band",
 "message": "finished after 48 evaluations: ..."}
```

`state` is `running`, `paused` or `done`. `pareto_front` is populated only for multi-objective runs.
`best_target_ranges_off_by_gen` exists so a UI can draw the convergence curve from this file alone,
without parsing every trial. Being a multiple of each objective's own target range rather than a raw
value, a run with several objectives in different units still plots as one curve.

## Steering a running search

Write `control.json` into the run directory. The loop reads it **between generations**, never
mid-evaluation:

```json
{"seq": 1, "command": "stop", "issued_by": "me", "reason": "converged"}
```

| command | args | effect |
|---|---|---|
| `stop` | — | finish the current generation, record the reason, exit |
| `pause` / `resume` | — | hold before the next `ask` / continue |
| `inject` | `{"candidates": [{address: value}, ...]}` | seed candidates into the next generation |

`seq` is a monotonic integer chosen by the writer. A command is applied only when
`seq > status.control_ack.seq`, and the outcome is recorded back in `control_ack` — at-most-once
delivery with no locking, confirmable by polling the same `status.json` a reader already watches.

There is deliberately **no `retarget` command**: changing an objective mid-run invalidates every
fitness already recorded. Stop, and start a new run.

## Current limitations

- **No parallel evaluation.** Candidates in a generation run one after another.
- **Per-key `unit`.** A dict parameter has one `unit` for the whole parameter, so per-key
  dimensions of `nest_params` (pF, ms, mV) share it.
- **`measures` is resolved once**, at setup, against the baseline run. It names a node instance, so
  renaming a node invalidates it — the failure is reported at setup, not mid-run.
- **The engine sets `results_path`** to isolate trials. A node that writes to a hardcoded or
  parameter-supplied absolute path bypasses that convention and will still collide.
- **No repeated evaluation of a candidate.** Objectives are bands, not points, so seed-to-seed noise
  is absorbed by the band rather than averaged away.
