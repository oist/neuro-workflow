"""The optimization loop.

The workflow is the objective function; the optimizer is the outer loop. A
workflow is a DAG and runs feed-forward once, so the loop cannot live inside it —
it sits outside and calls the workflow repeatedly:

    ask -> configure nodes -> workflow.execute() -> read measured values
        -> fitness -> tell -> log -> repeat

The workflow is built once and reconfigured between trials; it is never rebuilt.

The loop is fully usable with no agent attached: it narrates to stdout, writes the
ledger, and stops on its own. An agent or a GUI panel participates by reading the
ledger and writing ``control.json`` — nothing here requires that to happen.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .addressing import read_output, set_parameter
from .ledger import Ledger
from .optimizers import create_optimizer
from .spec import AlgorithmConfig, Objective, OptimizationSpec, build_spec

#: A hook that inspects a trial's outputs and returns a reason to reject it, or
#: None to accept. A scalar fitness cannot tell a healthy network from a
#: pathological one that merely averages to the right number — this is where that
#: judgement plugs in.
RejectFn = Callable[[Any, Dict[str, float]], Optional[str]]


@dataclass
class OptimizationResult:
    run_id: str
    run_dir: str
    spec: OptimizationSpec
    trials: List[Dict[str, Any]] = field(default_factory=list)
    best: Optional[Dict[str, Any]] = None
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""

    def configure_snippet(self) -> str:
        """A ready-to-paste snippet applying the best configuration."""
        if not self.best:
            return "# no successful trial"
        by_node: Dict[str, Dict[str, Any]] = {}
        for address, value in self.best["params"].items():
            node, _, rest = address.partition(".")
            by_node.setdefault(node, {})[rest] = value
        lines = []
        for node, params in by_node.items():
            args = ", ".join(f"{k.replace('.', '__')}={v!r}" for k, v in params.items())
            lines.append(f"{node}.configure({args})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def objective_fitness(objective: Objective, value: float) -> float:
    """Distance from the target, always minimized.

    ``in_range`` is zero anywhere inside the band and grows with the distance
    outside it — a band, not a point, so seed-to-seed noise is absorbed rather
    than chased.
    """
    if objective.goal == "minimize":
        return float(value)
    if objective.goal == "maximize":
        return -float(value)
    if value < objective.low:
        return float(objective.low - value)
    if value > objective.high:
        return float(value - objective.high)
    return 0.0


def _dominates(a: List[float], b: List[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = [r for r in rows if r.get("fitness")]
    return [
        r for r in scored
        if not any(_dominates(o["fitness"], r["fitness"])
                   for o in scored if o is not r)
    ]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def optimize(workflow,
             spec: Optional[OptimizationSpec] = None,
             results_path: str = "results/optimization",
             algorithm: Optional[AlgorithmConfig] = None,
             reject_fn: Optional[RejectFn] = None,
             run_id: Optional[str] = None,
             verbose: bool = True) -> OptimizationResult:
    """Optimize a built workflow against the targets declared in its schemas.

    Args:
        workflow: a built workflow that already runs at its current parameters.
        spec: the optimization spec. Built by introspection when omitted — which
            also runs the workflow once to establish the baseline.
        results_path: directory the run directory is created under.
        algorithm: algorithm config, when letting the spec be built here.
        reject_fn: optional dynamics check; return a reason to reject a trial.
        run_id: overrides the generated run id.
        verbose: narrate progress to stdout.
    """
    if spec is None:
        spec = build_spec(workflow, algorithm=algorithm)
    elif algorithm is not None:
        spec.algorithm = algorithm
    spec.validate()

    run_id = run_id or f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ledger = Ledger(f"{results_path}/{run_id}", run_id)
    ledger.write_manifest(spec, extra={
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow_name": getattr(workflow, "name", ""),
    })

    optimizer = create_optimizer(spec.dimensions, len(spec.objectives), spec.algorithm)

    def say(message: str) -> None:
        if verbose:
            print(message, flush=True)

    say(f"[{run_id}] {spec.algorithm.name}: "
        f"{len(spec.dimensions)} dimensions, {len(spec.objectives)} objective(s), "
        f"up to {spec.algorithm.max_generations} generations "
        f"x {spec.algorithm.pop_size} candidates")
    for line in spec.summary().splitlines():
        say("  " + line)

    trials: List[Dict[str, Any]] = []
    best_by_generation: List[Optional[float]] = []
    control_ack: Optional[Dict[str, Any]] = None
    stop_reason = ""
    paused = False
    trial_no = 0

    for generation in range(1, spec.algorithm.max_generations + 1):
        candidates = optimizer.ask()
        fitnesses: List[Optional[List[float]]] = []

        for index, candidate in enumerate(candidates):
            trial_no += 1
            row = _evaluate(workflow, spec, candidate, reject_fn,
                            trial_no, generation, index)
            fitnesses.append(row["fitness"])
            trials.append(row)
            ledger.append_trial(row)

        optimizer.tell(candidates, fitnesses)

        scored = [t for t in trials if t["fitness"] is not None]
        best = min(scored, key=lambda t: sum(t["fitness"]), default=None)
        best_by_generation.append(round(sum(best["fitness"]), 6) if best else None)

        in_range = bool(best and all(f == 0.0 for f in best["fitness"]))
        n_failed = sum(1 for t in trials if t["status"] == "failed")
        n_rejected = sum(1 for t in trials if t["status"] == "rejected")

        # Stop conditions, checked before honouring control so an explicit
        # command always wins over an automatic stop.
        if in_range:
            stop_reason = "a candidate is inside every target band"

        command = ledger.read_control()
        if command:
            control_ack, stop_reason, paused = _apply_control(
                command, optimizer, say, stop_reason, paused)

        ledger.write_status({
            "state": "done" if stop_reason else ("paused" if paused else "running"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "generation": generation,
            "max_generations": spec.algorithm.max_generations,
            "n_evals": len(trials),
            "n_ok": len(scored),
            "n_failed": n_failed,
            "n_rejected": n_rejected,
            "best": best,
            "pareto_front": _pareto_front(trials) if len(spec.objectives) > 1 else [],
            "progress": {
                "best_fitness_by_gen": best_by_generation,
                "gens_since_improvement": _gens_since_improvement(best_by_generation),
            },
            "control_ack": control_ack,
            "stop_reason": stop_reason or None,
            "message": _narrate(generation, spec, best, in_range,
                                n_failed, n_rejected),
        })

        say(f"  gen {generation}/{spec.algorithm.max_generations}: "
            + _narrate(generation, spec, best, in_range, n_failed, n_rejected))

        if stop_reason:
            break
        while paused:
            time.sleep(2)
            command = ledger.read_control()
            if command:
                control_ack, stop_reason, paused = _apply_control(
                    command, optimizer, say, stop_reason, paused)
            if stop_reason:
                break
        if stop_reason:
            break
    else:
        stop_reason = "generation budget exhausted"

    scored = [t for t in trials if t["fitness"] is not None]
    best = min(scored, key=lambda t: sum(t["fitness"]), default=None)
    say(f"[{run_id}] stopped: {stop_reason}")

    return OptimizationResult(
        run_id=run_id,
        run_dir=str(ledger.run_dir),
        spec=spec,
        trials=trials,
        best=best,
        pareto_front=_pareto_front(trials) if len(spec.objectives) > 1 else [],
        stop_reason=stop_reason,
    )


def _evaluate(workflow, spec, candidate, reject_fn,
              trial_no: int, generation: int, index: int) -> Dict[str, Any]:
    """Run one candidate and produce its ledger row."""
    started = time.time()
    row: Dict[str, Any] = {
        "trial": trial_no,
        "generation": generation,
        "candidate": index,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "source": "ask",
        "params": dict(candidate),
        "measured": {},
        "fitness": None,
        "status": "ok",
        "reject_reason": None,
        "error": None,
    }

    try:
        for address, value in candidate.items():
            set_parameter(workflow, address, value)

        if not workflow.execute():
            row.update(status="failed", error="workflow.execute() returned False")
            return _finish(row, started)

        measured = {}
        for objective in spec.objectives:
            value = read_output(workflow, objective.measures)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                row.update(status="failed",
                           error=f"{objective.measures} is not numeric: {value!r}")
                return _finish(row, started)
            measured[objective.name] = float(value)
        row["measured"] = measured

        if reject_fn is not None:
            reason = reject_fn(workflow, measured)
            if reason:
                row.update(status="rejected", reject_reason=reason)
                return _finish(row, started)

        row["fitness"] = [objective_fitness(o, measured[o.name])
                          for o in spec.objectives]

    except Exception as exc:  # a bad candidate must not kill the run
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")

    return _finish(row, started)


def _finish(row: Dict[str, Any], started: float) -> Dict[str, Any]:
    row["duration_s"] = round(time.time() - started, 3)
    return row


def _apply_control(command, optimizer, say, stop_reason: str, paused: bool):
    """Apply a control command and build the acknowledgement for status.json."""
    name = command.get("command", "")
    args = command.get("args", {}) or {}
    result = ""

    if name == "stop":
        stop_reason = f"stopped by {command.get('issued_by', 'control.json')}"
        result = "stopping"
    elif name == "pause":
        paused, result = True, "paused"
    elif name == "resume":
        paused, result = False, "resumed"
    elif name == "inject":
        accepted = 0
        for params in args.get("candidates", []):
            try:
                optimizer.enqueue(params)
                accepted += 1
            except NotImplementedError:
                result = f"{type(optimizer).__name__} cannot inject candidates"
                break
        result = result or f"{accepted} candidate(s) injected"
    else:
        result = f"unknown command {name!r}"

    say(f"  control: {name} -> {result}")
    return (
        {
            "seq": command.get("seq"),
            "command": name,
            "issued_by": command.get("issued_by"),
            "reason": command.get("reason"),
            "applied_at": datetime.now().isoformat(timespec="seconds"),
            "result": result,
        },
        stop_reason,
        paused,
    )


def _gens_since_improvement(history: List[Optional[float]]) -> int:
    scored = [h for h in history if h is not None]
    if not scored:
        return 0
    best = min(scored)
    for count, value in enumerate(reversed(history)):
        if value is not None and value <= best:
            return count
    return len(history)


def _narrate(generation, spec, best, in_range, n_failed, n_rejected) -> str:
    if best is None:
        return "no candidate has produced a usable measurement yet"
    values = ", ".join(
        f"{o.name.split('.')[-1]}={best['measured'][o.name]:.4g}"
        f"{' ' + o.unit if o.unit else ''}"
        for o in spec.objectives
    )
    state = "inside every target band" if in_range else \
        f"distance {sum(best['fitness']):.4g}"
    extra = ""
    if n_failed or n_rejected:
        extra = f" ({n_failed} failed, {n_rejected} rejected)"
    return f"best {values} — {state}{extra}"
