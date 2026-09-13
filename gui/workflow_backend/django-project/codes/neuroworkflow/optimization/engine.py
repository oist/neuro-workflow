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

import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from .addressing import (
    _is_number,
    clear_output_ports,
    discover_measurables,
    get_parameter,
    read_output,
    set_parameter,
)
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

    def apply_best(self, workflow, execute: bool = False) -> bool:
        """Set the workflow's parameters to those of the best trial.

        The search leaves the workflow configured with whatever the *last* trial
        used, which is rarely the best one. This applies the winner instead.

        Deliberately opt-in: parameters define a run, so the loop never writes a
        result back on its own. With ``execute=True`` the workflow is re-run
        afterwards, so its output ports match its parameters again — without it,
        the ports still hold the last trial's values.
        """
        if not self.best:
            return False
        for address, value in self.best["params"].items():
            set_parameter(workflow, address, value)
        return workflow.execute() if execute else True

    def configure_snippet(self) -> str:
        """A ready-to-paste snippet applying the best configuration.

        A dict key cannot be passed as a keyword argument, so those are emitted as
        a copy of the current dict with the tuned keys replaced — the same
        read-modify-write a notebook does by hand.
        """
        if not self.best:
            return "# no successful trial"

        scalars: Dict[str, Dict[str, Any]] = {}
        nested: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for address, value in self.best["params"].items():
            node, _, rest = address.partition(".")
            param, _, key = rest.partition(".")
            if key:
                nested.setdefault(node, {}).setdefault(param, {})[key] = value
            else:
                scalars.setdefault(node, {})[param] = value

        lines = []
        for node in dict.fromkeys(list(scalars) + list(nested)):
            args = [f"{k}={v!r}" for k, v in scalars.get(node, {}).items()]
            for param, keys in nested.get(node, {}).items():
                updates = ", ".join(f'"{k}": {v!r}' for k, v in keys.items())
                args.append(f'{param}={{**{node}._parameters["{param}"], {updates}}}')
            lines.append(f"{node}.configure({', '.join(args)})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


def objective_fitness(objective: Objective, value: float) -> float:
    """Distance from the target, always minimized.

    ``in_range`` is zero anywhere inside the band and grows with the distance
    outside it — a band, not a point, so seed-to-seed noise is absorbed rather
    than chased. Non-finite values are not a distance of zero; the caller
    must treat them as a failed trial.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite measurement {value!r} cannot be scored")
    if objective.goal == "minimize":
        return number
    if objective.goal == "maximize":
        return -number
    if number < objective.low:
        return float(objective.low - number)
    if number > objective.high:
        return float(number - objective.high)
    return 0.0


def objective_scales(spec) -> List[float]:
    """The scale each objective's distance is measured against, one per objective.

    Misses live in each objective's own unit — Hz for one, ms for another — so they
    cannot be added or compared as they stand. Dividing each by its own scale makes them
    comparable, and for a target range the natural scale is the width of that range: by
    asking for 40-50 Hz the user already said a 10 Hz spread is acceptable. It also
    avoids the usual difficulty of scaling by the nadir point, which needs knowledge of
    the Pareto front we do not have.

    Falls back to the magnitude of the target when a range has zero width, and to the
    baseline measurement for a minimize/maximize goal, which has no range at all.
    """
    scales = []
    for objective in spec.objectives:
        scale = None
        if (
            objective.goal == "in_range"
            and objective.low is not None
            and objective.high is not None
        ):
            width = float(objective.high) - float(objective.low)
            if width > 0:
                scale = width
            else:
                scale = abs(float(objective.high)) or None
        else:
            baseline = (spec.baseline.get("measured") or {}).get(objective.name)
            if _is_number(baseline) and baseline:
                scale = abs(float(baseline))
        scales.append(scale or 1.0)
    return scales


def target_ranges_off(spec, fitness: Optional[Sequence[float]]) -> Optional[float]:
    """How far the worst objective missed, counted in its own target range.

    One objective, targeted at 40-50 Hz, is a target range 10 Hz wide. A trial measuring
    25.9 Hz missed it by 14.1 Hz, so 14.1 / 10 = 1.4: the miss is 1.4 times the size of
    the range that was asked for. 0 means the value landed inside the range.

    Dividing is what makes two objectives comparable at all: a 14.1 Hz miss and an 8.2 ms
    miss cannot be ranked as they stand, but "1.4 times its range" and "0.4 times its
    range" can. The divisor comes from the user, who said by choosing 40-50 that a 10 Hz
    spread is acceptable.

    The value reported is the LARGEST of the objectives, not their sum: a trial is only
    as good as whatever it is doing worst, so 0 requires every objective to be inside its
    range. (Adding them would need a unit they do not share, and would quietly favour
    whichever objective uses bigger numbers.)

    Only ever used for reporting and for picking a representative. What the optimizer is
    told stays raw and per-objective, so the search itself is unaffected.
    """
    worst = worst_objective(spec, fitness)
    return None if worst is None else worst[1]


def worst_objective(spec, fitness: Optional[Sequence[float]]):
    """The objective furthest from its target, as ``(name, target ranges missed by)``."""
    if fitness is None:
        return None
    scales = objective_scales(spec)
    ranked = [(o.name, abs(f) / s) for o, f, s in zip(spec.objectives, fitness, scales)]
    return max(ranked, key=lambda pair: pair[1], default=None)


def decode_candidate(spec, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Map a proposal onto the real parameter space, one axis at a time.

    The samplers work in a continuous space and know nothing about whole numbers:
    CMA-ES proposes ``N = 2500.37`` for a count of neurons. Nothing is gained by
    teaching the algorithm about integers — sampling continuously and mapping here
    is the same search — but the mapping has to happen in ONE place, before
    ``configure()``, so that the value which runs is the value the ledger records.
    Left to each node's own ``int()`` call it was invisible: the ledger said
    2500.37 for a network of 2500.

    Rounding is clamped inside the declared range, so a proposal at the edge cannot
    be rounded out of bounds and rejected by ``configure()``.
    """
    decoded = dict(candidate)
    for dimension in spec.dimensions:
        if not dimension.integer or dimension.address not in decoded:
            continue
        value = decoded[dimension.address]
        try:
            nearest = int(round(float(value)))
        except (TypeError, ValueError):
            continue  # not a number; let the trial fail where it is reported
        low, high = math.ceil(dimension.low), math.floor(dimension.high)
        decoded[dimension.address] = max(low, min(high, nearest))
    return decoded


def reusable_paths(workflow) -> List[str]:
    """Artifacts the workflow's nodes say a later run can reuse.

    A node opts in with a class attribute, relative to ``results_path``::

        class MyBuilder(Node):
            REUSABLE_PATHS = ("network",)

    The engine neither knows nor cares what they contain — only that copying them
    into a trial lets a node recognise its own earlier work as still current and
    skip redoing it. A node that declares nothing simply gets no help, which is
    the correct default for any node whose outputs are cheap or always differ.
    """
    names = set()
    for node in workflow.nodes.values():
        for name in getattr(type(node), "REUSABLE_PATHS", ()) or ():
            name = str(name).strip()
            # Relative names only: a declaration must not reach outside the trial.
            if name and not os.path.isabs(name) and ".." not in name.split(os.sep):
                names.add(name)
    return sorted(names)


def _point_at_trial_dir(
    workflow,
    run_dir: str,
    trial_no: int,
    seed_root: Optional[str] = None,
    seed_names: Sequence[str] = (),
) -> str:
    """Give a trial its own results directory and return it.

    Nodes that write files — a network builder, a simulator config — otherwise
    rewrite the same paths on every trial. Beyond losing each trial's outputs,
    rebuilding a file in place can fail outright while a handle from the previous
    trial is still open. This is the same thing a notebook does by hand when it
    re-runs a workflow into a fresh directory.

    Anything the nodes declared reusable is copied in from ``seed_root``, so a
    trial that changes nothing relevant can skip rebuilding it. Sharing one
    directory between trials would save the copy, but a trial that *does* change
    the artifact would then rewrite it in place — the failure this isolation
    exists to prevent.
    """
    trial_dir = os.path.join(run_dir, "trials", f"{trial_no:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    if seed_root:
        for name in seed_names:
            source = os.path.join(seed_root, name)
            target = os.path.join(trial_dir, name)
            if os.path.isdir(source) and not os.path.exists(target):
                shutil.copytree(source, target)

    workflow.context["results_path"] = trial_dir
    for node in workflow.nodes.values():
        node._context["results_path"] = trial_dir
    return trial_dir


def _dominates(a: List[float], b: List[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def _pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scored = [r for r in rows if r.get("fitness")]
    return [
        r
        for r in scored
        if not any(_dominates(o["fitness"], r["fitness"]) for o in scored if o is not r)
    ]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def optimize(
    workflow,
    spec: Optional[OptimizationSpec] = None,
    results_path: str = "results/optimization",
    algorithm: Optional[AlgorithmConfig] = None,
    reject_fn: Optional[RejectFn] = None,
    run_id: Optional[str] = None,
    per_trial_results: bool = True,
    stop_when_reached: Optional[bool] = None,
    verbose: bool = True,
) -> OptimizationResult:
    """Optimize a built workflow against the targets declared in its schemas.

    Args:
        workflow: a built workflow that already runs at its current parameters.
        spec: the optimization spec. Built by introspection when omitted — which
            also runs the workflow once to establish the baseline.
        results_path: directory the run directory is created under.
        algorithm: algorithm config, when letting the spec be built here.
        reject_fn: optional dynamics check, called as ``reject_fn(workflow, measured)``
            — return a reason string to reject the trial, or None to accept it. A
            scalar cannot tell a healthy network from a pathological one that merely
            averages to the right number, so ``measured`` holds **every** numeric
            value the trial produced, addressed as ``node.port[.key]`` (the same
            mapping ``discover_measurables()`` returns), plus each objective under
            its declared name. A rejected trial is reported to the optimizer as a
            failure rather than with an invented penalty.
        run_id: overrides the generated run id.
        per_trial_results: give each trial its own results directory under the
            run (default). Any workflow whose nodes write files needs this:
            rewriting the same paths every trial loses each trial's outputs, and
            rebuilding a file in place can fail outright while a handle from the
            previous trial is still open. A workflow that writes nothing loses
            nothing by having it on, so set it False only to keep every trial in
            one directory on purpose. The workflow's results_path is left
            pointing at the last trial when the run ends.
        stop_when_reached: end the run as soon as one candidate satisfies every
            target. Defaults to True for a single objective — the question is
            answered, so more simulations are waste — and to False for several,
            where the useful answer is the Pareto front: the set of trade-offs
            between objectives. Stopping at the first candidate that satisfies
            everything would return a front of one, which says nothing about the
            trade-off. Only targets expressed as a band can be "reached" at all;
            a minimize or maximize goal has nothing to hit, so it never triggers.
        verbose: narrate progress to stdout.
    """
    if spec is None:
        spec = build_spec(workflow, algorithm=algorithm)
    elif algorithm is not None:
        spec.algorithm = algorithm
    spec.validate()

    if stop_when_reached is None:
        stop_when_reached = len(spec.objectives) == 1

    if run_id is None:
        # Second-resolution timestamps collide when runs start in the same second,
        # which would silently append one run's trials to another's ledger.
        run_id = base_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        attempt = 2
        while os.path.exists(os.path.join(results_path, run_id)):
            run_id = f"{base_id}_{attempt}"
            attempt += 1

    ledger = Ledger(os.path.join(results_path, run_id), run_id)
    ledger.write_manifest(
        spec,
        extra={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "workflow_name": getattr(workflow, "name", ""),
        },
    )

    optimizer = create_optimizer(spec.dimensions, len(spec.objectives), spec.algorithm)

    def say(message: str) -> None:
        if verbose:
            print(message, flush=True)

    say(
        f"[{run_id}] {spec.algorithm.name}: "
        f"{len(spec.dimensions)} dimensions, {len(spec.objectives)} objective(s), "
        f"up to {spec.algorithm.max_generations} generations "
        f"x {spec.algorithm.pop_size} candidates"
    )
    for line in spec.summary().splitlines():
        say("  " + line)
    if len(spec.objectives) > 1:
        say(
            "  each objective reports how far it missed, in its own unit. To compare "
            "objectives measured in different units, that miss is also given as a "
            "multiple of the target range asked for: '1.4x its target range' means the "
            "miss is 1.4 times as wide as the range. 0 = the target was met."
        )

    # Captured before the first rotation: where the baseline ran, and whatever the
    # nodes declared as reusable from it.
    seed_root = workflow.context.get("results_path") if per_trial_results else None
    seed_names = reusable_paths(workflow) if per_trial_results else []
    if seed_root and seed_names:
        say(f"  reusing {', '.join(seed_names)} from {seed_root}")

    reported_reached = False
    trials: List[Dict[str, Any]] = []
    best_by_generation: List[Optional[float]] = []
    control_ack: Optional[Dict[str, Any]] = None
    stop_reason = ""
    paused = False
    trial_no = 0

    for generation in range(1, spec.algorithm.max_generations + 1):
        candidates = optimizer.ask()
        sources = list(getattr(optimizer, "last_ask_sources", []) or [])
        fitnesses: List[Optional[List[float]]] = []

        for index, candidate in enumerate(candidates):
            trial_no += 1
            trial_dir = (
                _point_at_trial_dir(
                    workflow, str(ledger.run_dir), trial_no, seed_root, seed_names
                )
                if per_trial_results
                else None
            )
            row = _evaluate(
                workflow,
                spec,
                candidate,
                reject_fn,
                trial_no,
                generation,
                index,
                trial_dir,
                source=sources[index] if index < len(sources) else "ask",
            )
            fitnesses.append(row["fitness"])
            trials.append(row)
            ledger.append_trial(row)

            previous = [t for t in trials[:-1] if t["target_ranges_off"] is not None]
            is_best = row["target_ranges_off"] is not None and (
                not previous
                or row["target_ranges_off"]
                < min(t["target_ranges_off"] for t in previous)
            )
            try:
                say(
                    "  "
                    + _narrate_trial(
                        spec, row, generation, index, spec.algorithm.pop_size, is_best
                    )
                )
            except Exception as exc:
                say(f"  (could not narrate trial #{row['trial']}: {exc})")

        optimizer.tell(candidates, fitnesses)

        scored = [t for t in trials if t["fitness"] is not None]
        best = min(scored, key=lambda t: t["target_ranges_off"], default=None)
        best_by_generation.append(round(best["target_ranges_off"], 6) if best else None)

        # "Reached" only means something for a target expressed as a band: a
        # minimize or maximize goal has no value that counts as arrival.
        bands = [i for i, o in enumerate(spec.objectives) if o.goal == "in_range"]
        in_range = bool(
            best
            and bands
            and len(bands) == len(spec.objectives)
            and all(best["fitness"][i] == 0.0 for i in bands)
        )
        n_failed = sum(1 for t in trials if t["status"] == "failed")
        n_rejected = sum(1 for t in trials if t["status"] == "rejected")

        # Stop conditions, checked before honouring control so an explicit
        # command always wins over an automatic stop.
        if in_range and stop_when_reached:
            stop_reason = "a candidate is inside every target band"
        elif in_range and not reported_reached:
            reported_reached = True
            say(
                "  a candidate satisfies every target; continuing to develop "
                "the Pareto front"
            )

        command = ledger.read_control()
        if command:
            control_ack, stop_reason, paused = _apply_control(
                command,
                optimizer,
                say,
                stop_reason,
                paused,
                workflow=workflow,
                spec=spec,
            )

        ledger.write_status(
            {
                "state": "done" if stop_reason else ("paused" if paused else "running"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "generation": generation,
                "max_generations": spec.algorithm.max_generations,
                "n_evals": len(trials),
                "n_ok": len(scored),
                "n_failed": n_failed,
                "n_rejected": n_rejected,
                "best": best,
                "pareto_front": (
                    _pareto_front(trials) if len(spec.objectives) > 1 else []
                ),
                "progress": {
                    "best_target_ranges_off_by_gen": best_by_generation,
                    "gens_since_improvement": _gens_since_improvement(
                        best_by_generation
                    ),
                },
                "control_ack": control_ack,
                "stop_reason": stop_reason or None,
                "message": _narrate(
                    generation, spec, best, in_range, n_failed, n_rejected
                ),
            }
        )

        say(
            f"  gen {generation}/{spec.algorithm.max_generations}: "
            + _narrate(generation, spec, best, in_range, n_failed, n_rejected)
        )

        if stop_reason:
            break
        while paused:
            time.sleep(2)
            command = ledger.read_control()
            if command:
                control_ack, stop_reason, paused = _apply_control(
                    command,
                    optimizer,
                    say,
                    stop_reason,
                    paused,
                    workflow=workflow,
                    spec=spec,
                )
            if stop_reason:
                break
        if stop_reason:
            break
    else:
        stop_reason = "generation budget exhausted"

    scored = [t for t in trials if t["fitness"] is not None]
    best = min(scored, key=lambda t: t["target_ranges_off"], default=None)
    front = _pareto_front(trials) if len(spec.objectives) > 1 else []

    # A final status write, so a reader sees the run end. The per-generation write
    # happens before the loop decides to stop, and the budget-exhausted case is
    # decided after the loop entirely — without this, status.json would sit at
    # "running" forever.
    ledger.write_status(
        {
            "state": "done",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "generation": generation,
            "max_generations": spec.algorithm.max_generations,
            "n_evals": len(trials),
            "n_ok": len(scored),
            "n_failed": sum(1 for t in trials if t["status"] == "failed"),
            "n_rejected": sum(1 for t in trials if t["status"] == "rejected"),
            "best": best,
            "pareto_front": front,
            "progress": {
                "best_target_ranges_off_by_gen": best_by_generation,
                "gens_since_improvement": _gens_since_improvement(best_by_generation),
            },
            "control_ack": control_ack,
            "stop_reason": stop_reason,
            "message": f"finished after {len(trials)} evaluations: {stop_reason}",
        }
    )

    say(f"[{run_id}] stopped: {stop_reason}")
    if best:
        for objective, distance in zip(spec.objectives, best["fitness"]):
            unit = f" {objective.unit}" if objective.unit else ""
            band = (
                f"target {objective.low}-{objective.high}{unit}"
                if objective.goal == "in_range"
                else objective.goal
            )
            state = "in target" if distance == 0.0 else f"off {distance:.4g}{unit}"
            say(
                f"  {objective.name} = {best['measured'][objective.name]:.4g}{unit}"
                f"   ({band}, {state})"
            )
        if len(spec.objectives) > 1:
            furthest = worst_objective(spec, best["fitness"])
            if furthest and furthest[1] > 0:
                say(
                    f"  furthest from target: {furthest[0]}, "
                    f"{furthest[1]:.3g}x its target range"
                )

    return OptimizationResult(
        run_id=run_id,
        run_dir=str(ledger.run_dir),
        spec=spec,
        trials=trials,
        best=best,
        pareto_front=front,
        stop_reason=stop_reason,
    )


def _evaluate(
    workflow,
    spec,
    candidate,
    reject_fn,
    trial_no: int,
    generation: int,
    index: int,
    trial_dir: Optional[str] = None,
    source: str = "ask",
) -> Dict[str, Any]:
    """Run one candidate and produce its ledger row."""
    started = time.time()
    row: Dict[str, Any] = {
        "trial": trial_no,
        "generation": generation,
        "candidate": index,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "params": dict(candidate),
        "results_path": trial_dir,
        "measured": {},
        "fitness": None,
        "status": "ok",
        "reject_reason": None,
        "error": None,
    }

    try:
        # Mapped before anything is configured, and written back into the row, so
        # params/ledger/configure_snippet() all report the values that really ran.
        applied = decode_candidate(spec, candidate)
        row["params"] = dict(applied)
        for address, value in applied.items():
            set_parameter(workflow, address, value)

        clear_output_ports(workflow)

        if not workflow.execute():
            row.update(status="failed", error="workflow.execute() returned False")
            return _finish(row, started)

        measured = {}
        for objective in spec.objectives:
            value = read_output(workflow, objective.measures)
            if not _is_number(value):
                row.update(
                    status="failed",
                    error=f"{objective.measures} is not numeric: {value!r}",
                )
                return _finish(row, started)
            measured[objective.name] = float(value)
        row["measured"] = measured

        if reject_fn is not None:
            # Everything the trial produced, not only the objectives: the point of a
            # dynamics check is to look at what the objective does NOT capture — the
            # ISI regularity behind a correct-looking rate, say. Objective values are
            # merged in last, under their declared names, so a check written against
            # those keeps working.
            reason = reject_fn(workflow, {**discover_measurables(workflow), **measured})
            if reason:
                row.update(status="rejected", reject_reason=reason)
                return _finish(row, started)

        row["fitness"] = [
            objective_fitness(o, measured[o.name]) for o in spec.objectives
        ]

    except Exception as exc:  # a bad candidate must not kill the run
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")

    # Normalised, unit-free, for ranking and reporting only — the raw per-objective
    # fitness above is what the optimizer was told.
    row["target_ranges_off"] = target_ranges_off(spec, row["fitness"])
    return _finish(row, started)


def _finish(row: Dict[str, Any], started: float) -> Dict[str, Any]:
    row.setdefault("target_ranges_off", None)
    row["duration_s"] = round(time.time() - started, 3)
    return row


def _complete_candidate(workflow, spec, params: Dict[str, Any]) -> Dict[str, float]:
    """Fill missing dimension keys from the workflow's current parameter values."""
    complete: Dict[str, float] = {}
    for dimension in spec.dimensions:
        if dimension.address in params:
            complete[dimension.address] = params[dimension.address]
        else:
            complete[dimension.address] = get_parameter(workflow, dimension.address)
    return decode_candidate(spec, complete)


def _apply_control(
    command, optimizer, say, stop_reason: str, paused: bool, workflow=None, spec=None
):
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
        errors = []
        for params in args.get("candidates", []):
            try:
                complete = (
                    _complete_candidate(workflow, spec, params)
                    if workflow is not None and spec is not None
                    else dict(params)
                )
                optimizer.enqueue(complete)
                accepted += 1
            except NotImplementedError:
                result = f"{type(optimizer).__name__} cannot inject candidates"
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if not result:
            result = f"{accepted} candidate(s) injected"
            if errors:
                result += f" ({len(errors)} rejected: {errors[0]})"
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


def _narrate_trial(spec, row, generation, index, pop_size, is_best) -> str:
    """One line per trial: which trial, what it tried, what came out.

    A generation line alone leaves long stretches of simulator logging with nothing
    readable in them, and hides the fact that the search is exploring rather than
    stuck.
    """
    label = f"gen {generation}, trial {index + 1}/{pop_size} (#{row['trial']})"

    tried = "  ".join(
        (
            f"{d.address}={row['params'][d.address]:.4g}"
            f"{' ' + d.unit if d.unit else ''}"
            if d.address in row["params"]
            else f"{d.address}=?"
        )
        for d in spec.dimensions
    )

    if row["status"] != "ok":
        why = row.get("reject_reason") or row.get("error") or ""
        return f"{label}: {tried}  ->  {row['status']}: {why}"

    # Each distance in its own unit. Adding them would mean adding, say, Hz to ms —
    # a number with no meaning, dominated by whichever objective has the larger range.
    got = []
    for objective, distance in zip(spec.objectives, row["fitness"]):
        unit = f" {objective.unit}" if objective.unit else ""
        state = "in target" if distance == 0.0 else f"off {distance:.4g}{unit}"
        got.append(
            f"{objective.name.split('.')[-1]}="
            f"{row['measured'][objective.name]:.4g}{unit} ({state})"
        )

    # With one objective the miss above says everything. With several, name the one
    # doing worst and size its miss against its own target range, the only way to
    # compare a miss in Hz with a miss in ms.
    worst = ""
    if len(spec.objectives) > 1:
        furthest = worst_objective(spec, row["fitness"])
        if furthest and furthest[1] > 0:
            worst = (
                f"\n      furthest from target: {furthest[0]}, "
                f"{furthest[1]:.3g}x its target range"
            )
        elif furthest:
            worst = "\n      every target met"

    # The marker ranks this trial against the others, so it belongs on the line of
    # measurements — not after the indented line below, which compares this trial's
    # own objectives against each other.
    marker = "   <- best so far" if is_best else ""
    return f"{label}: {tried}  ->  {', '.join(got)}{marker}{worst}"


def _narrate(generation, spec, best, in_range, n_failed, n_rejected) -> str:
    """One line per generation: what the best candidate measured, against the goal.

    The goal is repeated every generation on purpose. A long run scrolls the spec far
    out of view, and "46 Hz" means nothing without "target 40-50 Hz" beside it.
    """
    if best is None:
        return "no candidate has produced a usable measurement yet"

    # The parameter values that produced it, so a generation line answers both
    # "how good" and "with what".
    values = "  ".join(
        f"{d.address}={best['params'][d.address]:.4g}"
        f"{' ' + d.unit if d.unit else ''}"
        for d in spec.dimensions
    )

    parts = []
    for objective, distance in zip(spec.objectives, best["fitness"]):
        unit = f" {objective.unit}" if objective.unit else ""
        goal = (
            f"target {objective.low}-{objective.high}{unit}"
            if objective.goal == "in_range"
            else objective.goal
        )
        state = "in target" if distance == 0.0 else f"off {distance:.4g}{unit}"
        parts.append(
            f"{objective.name.split('.')[-1]}="
            f"{best['measured'][objective.name]:.4g}{unit} [{goal}, {state}]"
        )

    if in_range:
        state = " — all targets met"
    elif len(spec.objectives) > 1:
        # "its" on purpose: this line is about the best trial, and the objective named
        # is the weakest one inside that trial — not a comparison against other trials.
        furthest = worst_objective(spec, best["fitness"])
        state = (
            (
                f" — its furthest objective from target: {furthest[0]}, "
                f"{furthest[1]:.3g}x its target range"
            )
            if furthest
            else ""
        )
    else:
        state = ""
    extra = ""
    if n_failed or n_rejected:
        extra = f" ({n_failed} failed, {n_rejected} rejected)"
    return f"best {values}  ->  {'; '.join(parts)}{state}{extra}"
