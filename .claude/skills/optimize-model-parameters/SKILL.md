---
name: optimize-model-parameters
description: Tune the parameters of an existing NeuroWorkflow graph until a measured quantity lands in a target range — choose what to explore and what to hit, check the model can reach the target, run the search, reject degenerate dynamics, and report the result. Use when the user wants to fit, tune, calibrate or optimize a workflow that already runs.
---

# Optimize Model Parameters

Take a workflow that already runs, and search its parameter space until a measured quantity —
a firing rate, an inter-spike interval, anything numeric a node outputs — lands where the user
wants it.

This skill is written for a person reading it as documentation *and* for an agent executing it.
Everything below is the real API; run it in a notebook, a script, or from an agent.

> **You orchestrate; the optimizer searches.** Do not try to be the optimizer — you decide what
> to tune, whether the target is reachable, whether a result is scientifically valid, and when to
> stop. CMA-ES proposes better next points than you can.

---

## Mental model — the loop wraps the graph

A workflow is a DAG: it runs once, feed-forward, and produces outputs. A search is a loop —
propose parameters, run the DAG, measure, propose again. A loop cannot live inside an acyclic
graph, so the optimizer sits **outside** the workflow and calls it repeatedly:

```
   ask ──►  configure nodes ──► workflow.execute() ──► read a measured value
                                                              │
             ┌────────────────────── tell ◄───── distance from target
             ▼
       propose the next candidates …
```

**The workflow is the objective function.** It is built **once** and reconfigured between
trials — never rebuilt.

---

## Quick start

```python
from neuroworkflow.optimization import AlgorithmConfig, build_spec, optimize

# 1. the search: which algorithm, how big a budget
spec = build_spec(workflow, AlgorithmConfig(name="cmaes", pop_size=8,
                                            max_generations=10, seed=1))

# 2. what to hit: a label, where it is measured, and the acceptable range
spec.add_objective(name="exc_firing_rate", measures="ana.firing_rate_hz.exc",
                   low=8.0, high=12.0, unit="Hz")

print(spec.summary())        # ALWAYS read this before spending simulation time

# 3. run it
result = optimize(workflow, spec=spec, results_path="./results/optimization")

print(result.stop_reason)
print(result.configure_snippet())    # ready-to-paste best configuration
result.apply_best(workflow)          # adopt it — a deliberate step, never automatic
```

`build_spec()` runs the workflow once at its current values. That run is the **baseline** every
result is compared against, and it is what every `measures` address is resolved against.

---

## What to explore, and what to hit

### What to explore — declared on the parameter

```python
clamp.NODE_DEFINITION.parameters["amp_na"].optimizable        = True
clamp.NODE_DEFINITION.parameters["amp_na"].optimization_range = [100.0, 1000.0]
clamp.NODE_DEFINITION.parameters["amp_na"].unit               = "nA"
```

For a **dict-valued** parameter, give a range per key — each key becomes its own dimension:

```python
exc.NODE_DEFINITION.parameters["nest_params"].optimizable = True
exc.NODE_DEFINITION.parameters["nest_params"].optimization_range = {
    "I_e":   [0.0, 400.0],
    "tau_m": [5.0, 50.0],
}
```

Two bounds that are not the same thing:

| | meaning | who enforces it |
|---|---|---|
| `constraints={"min":…, "max":…}` | what the model considers **valid at all** | `configure()` rejects violations |
| `optimization_range` | the patch **this study** wants searched | the optimizer samples inside it |

A range reaching past the constraints is clipped, and the clip is recorded in the spec. Never
propose a range outside `constraints`.

Each node **instance** carries its own definition, so marking `exc.nest_params` optimizable
leaves `inh` untouched even though both are `NW_Population`.

### What to hit — two ways, and one is preferred

**Preferred — declare it on the study:**

```python
spec.add_objective(name="exc_firing_rate", measures="ana.firing_rate_hz.exc",
                   low=8.0, high=12.0, unit="Hz")
```

An objective is a label, an address and a range. Nothing about it needs a parameter to hang it
on, so no node has to carry a value the simulation never reads, and the same workflow can be
optimized toward different targets without editing its nodes.

**Also supported — declare it on a node parameter**, for when a node author ships a sensible
default target:

```python
p = exc.NODE_DEFINITION.parameters["mean_firing_rate"]
p.is_objective, p.objective_range, p.unit = True, [8.0, 12.0], "Hz"
p.measures = "ana.firing_rate_hz.exc"
```

`build_spec()` discovers those. A target **without `measures` is skipped and reported** — a
range alone is not interpretable, since "firing rate" means different things in different nodes.

Goals other than a range: `goal="minimize"` or `goal="maximize"` (no `low`/`high`).

### Addressing

Everything is a dotted path starting with the **node's instance name** — the string passed to
the constructor, not the Python variable:

| kind | shape | example |
|---|---|---|
| parameter to tune | `Node.parameter[.key]` | `exc.nest_params.I_e` |
| value to measure | `Node.output_port[.key]` | `ana.firing_rate_hz.exc` |

**Never guess a `measures` address.** Ask the data:

```python
from neuroworkflow.optimization import discover_measurables
discover_measurables(workflow)     # every numeric leaf a run actually produced
# {'ana.firing_rate_hz.exc': 8.3, 'ana.isi_stats.exc.cv': 0.94, ...}
```

It reads the outputs of a real run, so it cannot go stale. Keys beginning with `_` are internal
plumbing and are deliberately excluded.

---

## Before you run: can the model even reach the target?

**Do this every time. It is the single most common way an optimization wastes hours.**

A search can only find what the model can produce. If the target lies outside what the
parameters can generate, the run completes, reports honestly, and tells you nothing — every
trial the same distance from a target none of them could reach.

This happened in this repository. `NW_Network_Optimization.ipynb` ran 24 trials, each measuring
**0 Hz**, because the explored current sat below the neuron's firing threshold:

```
g_L      = C_m / tau_m   = 250 pF / 10 ms   = 25 nS
ΔV       = V_th − E_L    = −55 − (−70)      = 15 mV
rheobase = ΔV × g_L                          = 375 pA   ← cannot spike below this
```

The notebook started at `I_e = 300 pA` and explored `[0, 400]`, so 94% of the space was silent —
and with no spikes anywhere, the recurrent weight had nothing to amplify, because silence is
self-sustaining. The loop was perfect; the model could not answer.

Worse, the target itself was unreachable: for a deterministic leaky integrate-and-fire neuron the
rate jumps from 0 to ~16 Hz as current crosses threshold, so 8–12 Hz existed only within a
**0.1 pA** window. No optimizer finds that, and none should be asked to.

**The check, before every run:**

1. **Is the baseline alive?** Read `spec.baseline["measurables"]`. All zeros usually means
   nothing is firing and the target is unreachable from here.
2. **Evaluate the extremes by hand.** Configure the low end of each range, execute, measure;
   then the high end. If the target does not sit between them, fix the ranges — not the
   optimizer.
3. **Know the analytic bound when one exists.** Rheobase for a LIF neuron; a synaptic weight that
   cannot produce a rate on its own; a target rate above `1000/t_ref`.
4. **Low rates need noise, not precision.** A deterministic neuron has a steep f–I curve near
   threshold. If the user wants a low rate, a stochastic input (Poisson) is the physiological
   answer; tightening the current range is not.

State what you checked. "The target is reachable: `I_e = 378 pA` gives 20 Hz and `417 pA` gives
40 Hz, both inside the explored range" is worth more than any amount of search machinery.

---

## Algorithms

```python
from neuroworkflow.optimization import available
available()   # ['cmaes', 'nsga2', 'nsga3', 'optuna_random', 'random', 'tpe']
```

| name | method | objectives | needs | use when |
|---|---|---|---|---|
| `random` | uniform sampling | any | — | smoke-testing the loop, or a baseline to beat |
| `cmaes` | CMA-ES | **1 only** | `optuna`, `cmaes` | default for continuous single-objective search |
| `tpe` | Tree-structured Parzen | any | `optuna` | trials are expensive, budget is small |
| `nsga2` | NSGA-II | any (built for 2–3) | `optuna` | 2–3 objectives; returns a Pareto front |
| `nsga3` | NSGA-III | any (built for 3+) | `optuna` | more than 3, where NSGA-II degrades |

```bash
pip install -e ".[optimization]"      # everything except 'random' needs Optuna
```

Without it, those names still appear in `available()` and raise an `ImportError` naming the
install command. Restart the kernel after installing.

**A single-objective backend refuses a multi-objective spec.** `cmaes` raises rather than
collapsing objectives into a weighted sum, because those weights would be an invented scientific
judgement. Use `nsga2`/`nsga3` and read the Pareto front.

Budget: at most `pop_size × max_generations` executions, plus one baseline. With a simulation
costing seconds, that product is the first number to think about. `options` is forwarded verbatim
to the underlying sampler — `AlgorithmConfig(name="cmaes", options={"sigma0": 0.2})`.

---

## Several objectives, and units

Each objective reports **how far it missed, in its own unit** — 14.1 Hz past a rate target,
8.2 ms past an interval target. **Never add them.** The sum has no unit, no meaning, and quietly
favours whichever objective uses bigger numbers.

The one comparable figure is the worst objective's miss divided by the width of its own target
range:

```
gen 1, trial 1/4 (#1): exc.amp=129.5 nA  exc.w=7.542 pA
  ->  network_rate=25.91 Hz (off 14.09 Hz), probe_isi=11.79 ms (off 8.212 ms)   <- best so far
      furthest from target: network_rate, 1.41x its target range
```

`1.41x` means the miss is 1.41 times as wide as the range that was asked for; `0` means every
target is met. Stored as `target_ranges_off` in the ledger, alongside the raw per-objective
`fitness`. It is used for **ranking and reporting only** — what the optimizer is told stays raw
and per-objective, and Pareto dominance is scale-free anyway.

When explaining this to a user: give the arithmetic with their numbers ("your target is 40–50 Hz,
that range is 10 Hz wide, the trial missed by 14.1 Hz, so 1.4"). Do not say "normalized",
"achievement" or "band width" — see the wording rule at the end.

A multi-objective run does not stop at the first candidate that satisfies everything, because the
useful answer is the **front**, not one point:

```python
for row in result.pareto_front:
    print(row["trial"], row["measured"], row["params"], row["target_ranges_off"])
```

Choosing among them is a scientific judgement, not something the search can make.
`apply_best()` falls back to the member whose worst objective missed by the least — a tie-break
and nothing more.

---

## Where you add value, and where you do not

**For the numerical search you add nothing.** Do not insert yourself into every generation by
reflex: that is latency and tokens for no gain. Your value is at the edges.

1. **Setup.** Decide what to tune, propose biologically grounded ranges, map each target to the
   right measured value when several populations exist, and **verify the target is reachable**
   (above). A script cannot do this. *Highest value.*
2. **Rejecting bad dynamics.** A scalar is blind to *how* a target was met — see `reject_fn`
   below. *The strongest reason a neuroscientist is in this loop.*
3. **Watching a long run.** Read the ledger and tell the user the *trend* while it is still
   running: "gen 40/100, the worst objective is now 0.4x its target range, converging in maybe
   15 more generations" or "stalled for 8 generations — the ranges may be too narrow". Turns an
   opaque run into a steerable one. *High value at scale.*
4. **Interpretation.** Explain the front, recommend a configuration, name the trade-off.

If a run is small, routine and fully specified, set it going and read the result.

---

## Rejecting degenerate dynamics

A configuration can hit "10 Hz" while the network is silent except a few bursting cells, or
synchronized in an epileptic-like regime that merely *averages* to 10 Hz. Fitness says perfect;
biology says garbage. `reject_fn` is where that judgement goes:

```python
def reject(workflow, measured):
    """Return a reason to reject this trial, or None to accept it."""
    cv = measured.get("ana.isi_stats.exc.cv")
    if cv is not None and cv < 0.1:
        return f"clock-like firing (ISI CV {cv:.2f}) — not a plausible regime"
    return None

result = optimize(workflow, spec=spec, results_path="./results/opt", reject_fn=reject)
```

Signature: `reject_fn(workflow, measured) -> str | None`, where `measured` is the same flat
address→value mapping `discover_measurables()` returns. A rejected trial is recorded as
`status: "rejected"` with the reason, and reported as `null` fitness rather than a large penalty,
so the sampler marks it failed instead of letting an invented number distort its model.

`rejected` is first-class in the ledger, not a footnote — it is the whole reason a human or an
agent is in the loop.

---

## Watching and steering a running search

Everything a monitor needs is written to `<results_path>/<run_id>/`:

| file | contents |
|---|---|
| `run.json` | the manifest: dimensions, objectives, algorithm, baseline. Written once |
| `trials.jsonl` | one line per evaluation, append-only |
| `status.json` | replaced each generation: `state`, counts, `best`, `progress.best_target_ranges_off_by_gen` |
| `control.json` | **written by you** to steer the run |

Read `status.json` for a summary; `progress.best_target_ranges_off_by_gen` draws a convergence
curve without parsing every trial.

To steer, write `control.json` into the run directory:

```json
{"seq": 1, "command": "stop", "issued_by": "agent", "args": {}}
```

Commands: `stop`, `pause`, `resume`, and `inject` (with
`args={"candidates": [{"exc.nest_params.I_e": 390.0, "conn.syn_weight": 12.0}]}`). The loop reads
it **between generations** and applies a command at most once — only when `seq` is higher than
the last acknowledged one, which it records in `status.control_ack`. So increment `seq` for each
new command and confirm delivery by reading the same file you already watch.

There is deliberately **no retarget command**: changing an objective mid-run invalidates every
fitness already recorded. Stop, and start a new run.

---

## The locked evaluation protocol

Everything not in the exploration space is **frozen for the whole run**: simulation length, `dt`,
the analysis window and method, and the network topology. Only `optimizable` parameters may vary.
If anything else drifts, fitnesses stop being comparable and the search is invalid.

Trial-to-trial RNG variation is **acceptable and need not be controlled** — objectives are ranges,
not points, so seed noise is absorbed. Do not spend budget averaging seeds unless asked.

---

## When a search stalls

Read the ledger first, then work through this in order:

1. **Check reachability again** (above). A flat objective across every trial almost always means
   the target cannot be produced, not that the algorithm is weak.
2. **Combine near-misses** — inject an interpolation of the best candidates so far via
   `control.json`.
3. **Probe the boundaries.** If the best points sit against a range edge, the optimum is probably
   outside it. Propose widening — and **ask the user** before expanding any range.
4. **Drop dead dimensions.** If a dimension shows no effect on the objective across the log, say
   so and propose removing it so the search concentrates where it matters.

---

## Reporting

- **Single objective**: the best parameters, the measured value, and whether it is inside the
  target range — in the objective's own unit.
- **Several**: the Pareto front, with each member's measured values; recommend one and name the
  trade-off it represents.
- Always: the baseline-versus-best comparison, the number of evaluations, and the stopping reason.
- Always: a ready-to-use snippet — `result.configure_snippet()` produces it, correctly handling
  dict keys (`nest_params={**exc._parameters["nest_params"], "I_e": 390.0}`).

**Not reaching the target is a result, not a failure.** The run answered "nothing in these ranges
hits that band" — which is information about the model. Only a run where no trial produced a
usable measurement (`result.best is None`) has actually failed.

The search leaves the workflow holding the **last** trial's values, and its `results_path`
pointing at the last trial's directory. Adopting the winner is a separate, deliberate step
(`apply_best`). Say clearly what was applied.

---

## Expensive artifacts: do not rebuild what did not change

A search re-runs the whole graph every trial, including anything expensive it builds. Nodes can
declare what survives a trial unchanged:

```python
class NW_SimConfig(Node):
    REUSABLE_PATHS = ("network",)     # relative to results_path
```

The engine copies those directories in from the baseline before each trial; a node that
recognises its own earlier work skips redoing it. For the `NW_*` builders that means the SONATA
network is built once instead of once per trial, controlled by `rebuild_network`
(`auto` / `always` / `never`).

Know which parameters are **structural** and which are **runtime**, because only structural ones
force a rebuild:

| | example | written to |
|---|---|---|
| structural | `syn_weight`, `N`, connection rules | `nodes.h5`, `edge_types.csv` — rebuild required |
| runtime | `nest_params`, synapse `dynamics_params_dict` | a JSON referenced by filename — no rebuild |

If a search tunes only runtime values, the network is built once for the entire run.

---

## From the GUI

The same search is declared on canvas with an **`NW_Optimization`** node: it holds *how* to
search (algorithm, budget, seed, results path), has no ports and no process steps, and takes no
part in the workflow's execution. Its presence is the signal for the code generator to emit an
optimization run instead of a single execution.

```python
opt = NW_Optimization("opt")
opt.configure(algorithm="cmaes", pop_size=16, max_generations=12, seed=1)

spec   = build_spec(workflow, opt.algorithm_config())      # not added to the workflow
spec.add_objective(name="exc_firing_rate", measures="ana.firing_rate_hz.exc",
                   low=40.0, high=50.0, unit="Hz")
result = optimize(workflow, spec=spec, results_path=opt.results_path())
```

**`NW_Optimization` defaults to `algorithm="random"`**, which needs nothing beyond NumPy, so a
run works out of the box. Choosing `cmaes`/`tpe`/`nsga2`/`nsga3` needs Optuna — and the nest kernel
image does **not** have it as of 2026-09-10, so those algorithms fail there until `optuna` and
`cmaes` are added to `Dockerfile.nest`. See `docs/OPTIMIZATION_GUI_HANDOFF.md`.

Treat `random` as a way to check the loop and the addresses, not as a search: it is a baseline to
beat, and a real study should switch to `cmaes` (one objective) or `nsga2` (several).

---

## Checklist

- [ ] The workflow runs cleanly at its current values; baseline recorded
- [ ] **Target reachability checked** — baseline alive, range extremes evaluated, analytic bound
      considered; stated explicitly
- [ ] Every `measures` address taken from `discover_measurables()`, not guessed
- [ ] Ranges inside `constraints`, and grounded in something — literature, the model, or the user
- [ ] Dict parameters expanded to per-key ranges where keys are explored
- [ ] Objective count matches the algorithm (1 → `cmaes`/`tpe`, 2–3 → `nsga2`, more → `nsga3`)
- [ ] `spec.summary()` reviewed **before** spending simulation time
- [ ] Evaluation protocol locked — only `optimizable` parameters vary
- [ ] Dynamics checked via `reject_fn` where a right number could come from wrong dynamics
- [ ] Long runs narrated from the ledger, not only reported at the end
- [ ] Misses reported in their own units; never summed across objectives
- [ ] Stopping reason stated; "target not reached" reported as a result, not a failure
- [ ] Best configuration reported with a ready-to-use `configure()` snippet, and applying it
      described as a separate step

---

## Wording rules for anything a user reads

The vocabulary here was rewritten after real confusion — keep it:

- Say **"target range"**, never "band" or "band width" (*bandwidth* means a frequency range to a
  neuroscientist).
- Say **"1.4x its target range"**, never "achievement", "normalized" or "ratio" — a ratio invites
  "of what to what?".
- Always **name the objective** that is worst: `furthest from target: probe_isi, 1.4x its target
  range`.
- Print a miss **with its unit** (`off 8.2 ms`); print the comparable figure **without one**.
- Define any term in plain words the first time it appears, with the user's own numbers.

---

## Scope

- This skill **optimizes an existing, working workflow**. It does not build nodes — use the
  `create-node` skill for that.
- The search itself is `neuroworkflow.optimization`, wrapping Optuna's samplers. Never
  reimplement a search; add a backend with `register_optimizer()` if one is missing.
- Reference material: `docs/OPTIMIZATION.md` (how the engine works),
  `docs/OPTIMIZATION_GUI_HANDOFF.md` (the GUI half), and five runnable notebooks under
  `notebooks/` — `NW_SingleCell_Optimization`, `NW_Clamp_Weight_Optimization`,
  `NW_Network_Optimization`, and `generated_optimization_example` /
  `generated_multiobjective_example`, which show what generated code looks like.
