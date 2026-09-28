"""The optimization spec — what to tune, what to hit, and with which algorithm.

The spec is plain data. It is built from the node schemas by introspection, then
handed to a human or an agent to review and edit, then consumed by the engine and
frozen into ``run.json``. Nothing in it is Python-specific, so an agent can change
the algorithm, widen a range or retarget a band without touching code.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from neuroworkflow.core.schema import ParameterDefinition

from .addressing import _is_number, clear_output_ports, discover_measurables


@dataclass
class Dimension:
    """One axis of the search space."""

    address: str  # "Model.nest_params.C_m"
    low: float
    high: float
    unit: str = ""
    description: str = ""
    source: str = ""  # where the range came from
    note: str = ""  # e.g. why it was clipped
    # A count of neurons is not a continuous quantity. The samplers stay continuous
    # anyway — the engine maps a proposal onto this axis before configure() sees it,
    # so the value that runs is the value the ledger records.
    integer: bool = False


@dataclass
class Objective:
    """One target the search is steering toward."""

    name: str  # "Population_exc.target_rate_hz"
    measures: str  # "Analysis.firing_rate_hz.exc"
    low: Optional[float] = None  # target band, for goal="in_range"
    high: Optional[float] = None
    goal: str = "in_range"  # in_range | minimize | maximize
    unit: str = ""
    description: str = ""
    source: str = ""


@dataclass
class AlgorithmConfig:
    """Which optimizer to run and how. Editable by a human or an agent."""

    # CMA-ES, not random: someone who omits the algorithm has not asked for a
    # baseline, they just did not specify. 'random' is uniform sampling, so a
    # silent default of it would look like a search without being one. This needs
    # Optuna; the ImportError names the install command.
    name: str = "cmaes"  # see optimizers.available()
    pop_size: int = 16
    max_generations: int = 20
    seed: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSpec:
    dimensions: List[Dimension] = field(default_factory=list)
    objectives: List[Objective] = field(default_factory=list)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    locked_protocol: Dict[str, Any] = field(default_factory=dict)
    baseline: Dict[str, Any] = field(default_factory=dict)
    skipped: List[Dict[str, str]] = field(default_factory=list)

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationSpec":
        return cls(
            dimensions=[Dimension(**d) for d in data.get("dimensions", [])],
            objectives=[Objective(**o) for o in data.get("objectives", [])],
            algorithm=AlgorithmConfig(**data.get("algorithm", {})),
            locked_protocol=data.get("locked_protocol", {}),
            baseline=data.get("baseline", {}),
            skipped=data.get("skipped", []),
        )

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        """Raise if the spec cannot be run. Called by the engine before starting."""
        if not self.dimensions:
            raise ValueError(
                "No search dimensions — mark at least one parameter "
                "optimizable=True with a range, or add a Dimension by hand."
            )
        if not self.objectives:
            raise ValueError(
                "No objectives — optimization without a target has no direction. "
                "Declare is_objective=True with objective_range and measures, or "
                "add an Objective by hand."
            )
        if self.algorithm.pop_size < 1 or self.algorithm.max_generations < 1:
            raise ValueError(
                f"Algorithm budget: pop_size {self.algorithm.pop_size} and "
                f"max_generations {self.algorithm.max_generations} must both be >= 1"
            )
        for d in self.dimensions:
            if d.low >= d.high:
                raise ValueError(f"Dimension {d.address}: low {d.low} >= high {d.high}")
            # collect_dimensions() skips such an axis, but a spec written by hand,
            # reloaded from run.json, or edited by an agent never passes through it.
            # Every proposal would round to the same invalid value.
            if d.integer and not _has_integer_inside(d.low, d.high):
                raise ValueError(
                    f"Dimension {d.address} is a whole-number quantity but no "
                    f"integer lies inside [{d.low}, {d.high}]"
                )
        for o in self.objectives:
            if o.goal not in ("in_range", "minimize", "maximize"):
                raise ValueError(f"Objective {o.name}: unknown goal {o.goal!r}")
            if o.goal == "in_range" and (o.low is None or o.high is None):
                raise ValueError(
                    f"Objective {o.name}: goal 'in_range' needs both low and high "
                    f"(set objective_range on the parameter)"
                )
            # A zero-width band (low == high) is a point target and is allowed.
            if o.goal == "in_range" and o.low > o.high:
                raise ValueError(f"Objective {o.name}: low {o.low} > high {o.high}")

    def add_objective(
        self,
        name: str,
        measures: str,
        low: Optional[float] = None,
        high: Optional[float] = None,
        goal: str = "in_range",
        unit: str = "",
        description: str = "",
    ) -> "OptimizationSpec":
        """Declare a target that no node parameter carries.

        An objective is a label, a measurement address and a band — nothing about it
        requires a parameter to hang it on. Declaring it here keeps a study's target
        out of the model, and avoids inventing a parameter the simulation never reads
        purely as somewhere to store a number.

        The address is checked against the baseline run, so a typo fails here, naming
        what *is* available, rather than at the first trial.
        """
        if any(o.name == name for o in self.objectives):
            raise ValueError(f"An objective named {name!r} is already declared")

        available = self.baseline.get("measurables") or {}
        if available and measures not in available:
            node = measures.split(".")[0]
            if node not in {a.split(".")[0] for a in available}:
                raise ValueError(
                    f"measures {measures!r}: no node named {node!r} produced anything "
                    f"measurable. Nodes that did: "
                    f"{', '.join(sorted({a.split('.')[0] for a in available}))}"
                )
            near = sorted(a for a in available if a.startswith(node + "."))
            raise ValueError(
                f"measures {measures!r} is not a number produced by the baseline run. "
                f"On {node!r}: {', '.join(near)}"
            )

        self.objectives.append(
            Objective(
                name=name,
                measures=measures,
                low=low,
                high=high,
                goal=goal,
                unit=unit,
                description=description,
                source="study",
            )
        )

        # build_spec() records a baseline value for every objective it discovers.
        # A target added afterwards needs the same entry: objective_scales() reads
        # it to normalize a minimize/maximize fitness, and without it that
        # objective silently falls back to a scale of 1.0 and is weighted wrongly
        # against the others.
        if available and isinstance(self.baseline.get("measured"), dict):
            self.baseline["measured"][name] = available.get(measures)

        return self

    def summary(self) -> str:
        lines = [
            f"algorithm : {self.algorithm.name} "
            f"(pop_size={self.algorithm.pop_size}, "
            f"max_generations={self.algorithm.max_generations})",
            f"dimensions: {len(self.dimensions)}",
        ]
        for d in self.dimensions:
            unit = f" {d.unit}" if d.unit else ""
            note = f"   [{d.note}]" if d.note else ""
            kind = " integer" if d.integer else ""
            lines.append(f"    {d.address:<40} [{d.low}, {d.high}]{unit}{kind}{note}")
        lines.append(f"objectives: {len(self.objectives)}")
        for o in self.objectives:
            band = f"in [{o.low}, {o.high}]" if o.goal == "in_range" else o.goal
            unit = f" {o.unit}" if o.unit else ""
            lines.append(f"    {o.name:<40} {band}{unit}  <- {o.measures}")
        if self.skipped:
            lines.append(f"skipped   : {len(self.skipped)}")
            for s in self.skipped:
                lines.append(f"    {s['address']:<40} {s['reason']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Building a spec from the node schemas
# ---------------------------------------------------------------------------


def _is_integer_axis(pdef: ParameterDefinition, live_value: Any, key: str = "") -> bool:
    """Is this axis a whole-number quantity — a count of neurons, a number of trials?

    Read from the node author's **declared default**, not from whatever the node
    currently holds: ``N = 20`` says the parameter counts things, while a current
    value of ``200`` for a parameter declared as ``0.15`` nA says only that someone
    typed an int literal for a continuous quantity. Inferring from the live value
    would silently restrict that search to whole nanoamps.

    ``constraints={"integer": True/False}`` overrides the inference, and for a dict
    parameter may be given per key, e.g. ``{"integer": {"n_syn": True}}``.
    """
    declared = pdef.constraints.get("integer")
    if isinstance(declared, dict) and key:
        declared = declared.get(key)
    if isinstance(declared, bool):
        return declared

    reference = pdef.default_value
    if key:
        reference = reference.get(key) if isinstance(reference, dict) else None
    if reference is None:
        # No declared default to judge by — a key added through configure(), or a
        # parameter whose default is None. The live value is all there is.
        reference = live_value
    # bool is an int subclass, and a flag is not a quantity to search.
    return isinstance(reference, int) and not isinstance(reference, bool)


def _has_integer_inside(low: float, high: float) -> bool:
    """True when at least one whole number lies within [low, high]."""
    import math

    return math.ceil(low) <= math.floor(high)


def _unusable_value(value: Any) -> str:
    """Why this current value cannot be a numeric search axis, or '' if it can.

    A declared ``optimization_range`` says a parameter is tunable, but the value
    the node actually holds decides whether that is true. Text (a NEST model
    name), a flag, or NaN cannot be searched: the sampler would propose a float,
    ``configure()`` would store it, and the simulation would run something the
    author never meant. ``None`` is allowed through — a parameter that was never
    configured has nothing to contradict its declared range.
    """
    if value is None or _is_number(value):
        return ""
    return f"current value is {type(value).__name__} {value!r}, not a number"


def _numeric_pair(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return [float(value[0]), float(value[1])]
    return None


def collect_dimensions(workflow) -> tuple:
    """Collect search dimensions from every ``optimizable=True`` parameter.

    Returns ``(dimensions, skipped)``. A parameter is skipped rather than
    guessed at when no usable range can be derived — the engine reports those so
    a human or agent can supply ranges deliberately.
    """
    dimensions: List[Dimension] = []
    skipped: List[Dict[str, str]] = []

    for node_name, node in workflow.nodes.items():
        params = getattr(node, "NODE_DEFINITION", None)
        if params is None:
            continue
        for pname, pdef in params.parameters.items():
            if not isinstance(pdef, ParameterDefinition) or not pdef.optimizable:
                continue

            address = f"{node_name}.{pname}"

            # The node's live value, not the schema default: a key added through
            # configure() (a NEST parameter the default dict does not list, say)
            # is just as tunable as one that was declared.
            value = node._parameters.get(pname)

            # A dict-valued parameter declares one range per key. Each becomes its
            # own dimension, addressed as Node.parameter.key — the address the
            # engine reassembles into the whole dict before calling configure().
            if isinstance(pdef.optimization_range, dict):
                for key, pair in pdef.optimization_range.items():
                    key_address = f"{address}.{key}"
                    rng = _numeric_pair(pair)
                    if rng is None:
                        skipped.append(
                            {
                                "address": key_address,
                                "reason": f"range is not [min, max]: {pair}",
                            }
                        )
                        continue
                    if isinstance(value, dict) and key not in value:
                        skipped.append(
                            {
                                "address": key_address,
                                "reason": "key is not present in the parameter's "
                                "current value — configure() it first",
                            }
                        )
                        continue
                    key_value = value.get(key) if isinstance(value, dict) else None
                    problem = _unusable_value(key_value)
                    if problem:
                        skipped.append({"address": key_address, "reason": problem})
                        continue
                    integer = _is_integer_axis(pdef, key_value, key=key)
                    if integer and not _has_integer_inside(rng[0], rng[1]):
                        skipped.append(
                            {
                                "address": key_address,
                                "reason": f"integer value but no whole number inside "
                                f"[{rng[0]}, {rng[1]}]",
                            }
                        )
                        continue
                    dimensions.append(
                        Dimension(
                            address=key_address,
                            low=rng[0],
                            high=rng[1],
                            unit=pdef.unit,
                            description=pdef.description,
                            source="schema.optimization_range",
                            integer=integer,
                        )
                    )
                continue

            if isinstance(value, dict):
                skipped.append(
                    {
                        "address": address,
                        "reason": "dict parameter with no per-key optimization_range — "
                        'declare one, e.g. {"V_th": [-60.0, -45.0]}',
                    }
                )
                continue

            problem = _unusable_value(value)
            if problem:
                skipped.append({"address": address, "reason": problem})
                continue

            c_min = pdef.constraints.get("min")
            c_max = pdef.constraints.get("max")
            rng = _numeric_pair(pdef.optimization_range)
            source, note = "schema.optimization_range", ""

            if rng is None:
                if isinstance(c_min, (int, float)) and isinstance(c_max, (int, float)):
                    rng, source = [float(c_min), float(c_max)], "schema.constraints"
                else:
                    skipped.append(
                        {
                            "address": address,
                            "reason": "optimizable but no optimization_range and no "
                            "min/max constraints to fall back on",
                        }
                    )
                    continue

            # Constraints are the authority: a range reaching past them describes
            # points configure() would reject anyway, so clip rather than fail.
            low, high = rng
            clipped = []
            if isinstance(c_min, (int, float)) and low < c_min:
                clipped.append(f"low raised to {c_min}")
                low = float(c_min)
            if isinstance(c_max, (int, float)) and high > c_max:
                clipped.append(f"high lowered to {c_max}")
                high = float(c_max)
            if clipped:
                source, note = "clipped", "; ".join(clipped) + " (constraints)"

            if low >= high:
                skipped.append(
                    {
                        "address": address,
                        "reason": f"empty range after clipping to constraints: [{low}, {high}]",
                    }
                )
                continue

            integer = _is_integer_axis(pdef, value)
            if integer and not _has_integer_inside(low, high):
                skipped.append(
                    {
                        "address": address,
                        "reason": f"integer parameter but no whole number inside "
                        f"[{low}, {high}]",
                    }
                )
                continue

            dimensions.append(
                Dimension(
                    address=address,
                    low=low,
                    high=high,
                    unit=pdef.unit,
                    description=pdef.description,
                    source=source,
                    note=note,
                    integer=integer,
                )
            )

    return dimensions, skipped


def collect_objectives(workflow, measurables: Dict[str, float]) -> tuple:
    """Collect objectives from every ``is_objective=True`` parameter.

    ``measures`` is resolved against the measurables discovered in a baseline
    run, so a broken address fails here — before any search budget is spent —
    rather than midway through the loop.
    """
    objectives: List[Objective] = []
    skipped: List[Dict[str, str]] = []

    for node_name, node in workflow.nodes.items():
        definition = getattr(node, "NODE_DEFINITION", None)
        if definition is None:
            continue
        for pname, pdef in definition.parameters.items():
            if not isinstance(pdef, ParameterDefinition) or not pdef.is_objective:
                continue

            name = f"{node_name}.{pname}"

            if not pdef.measures:
                skipped.append(
                    {
                        "address": name,
                        "reason": "is_objective=True but no measures address — set it to "
                        "the output that holds the measured value",
                    }
                )
                continue

            if pdef.measures not in measurables:
                near = [
                    m for m in measurables if m.startswith(pdef.measures.split(".")[0])
                ]
                skipped.append(
                    {
                        "address": name,
                        "reason": f"measures {pdef.measures!r} did not resolve to a number "
                        f"in the baseline run"
                        + (
                            f" (available on that node: {', '.join(sorted(near)[:6])})"
                            if near
                            else ""
                        ),
                    }
                )
                continue

            band = _numeric_pair(pdef.objective_range)
            if band is None:
                skipped.append(
                    {
                        "address": name,
                        "reason": "is_objective=True but objective_range is not [min, max]",
                    }
                )
                continue

            # The schema can only express a target band. minimize/maximize are set
            # by editing the spec, which is also where an agent would change them.
            objectives.append(
                Objective(
                    name=name,
                    measures=pdef.measures,
                    low=band[0],
                    high=band[1],
                    goal="in_range",
                    unit=pdef.unit,
                    description=pdef.description,
                    source="schema.objective_range",
                )
            )

    return objectives, skipped


def build_spec(
    workflow, algorithm: Optional[AlgorithmConfig] = None, run_baseline: bool = True
) -> OptimizationSpec:
    """Introspect a built workflow and produce a spec ready for review.

    With ``run_baseline=True`` the workflow is executed once at its current
    parameters. That run does two jobs: it is the reference point every result is
    compared against, and its outputs are what ``measures`` addresses resolve
    against.
    """
    baseline: Dict[str, Any] = {}
    measurables: Dict[str, float] = {}

    if run_baseline:
        # Trials clear the ports first; the baseline must too. A workflow is
        # normally executed once in the notebook before anyone optimizes it, and
        # Node.process() swallows exceptions — so without this a node that fails
        # here leaves its previous value in place and the baseline records it as
        # a fresh measurement. Every result is compared against that number, and
        # objective_scales() uses it to normalize minimize/maximize fitness.
        clear_output_ports(workflow)
        ok = workflow.execute()
        measurables = discover_measurables(workflow)
        baseline = {
            "executed": bool(ok),
            "measurables": measurables,
        }
        if not ok:
            raise RuntimeError(
                "The workflow failed at its current parameters — fix that before "
                "optimizing (check the printed node errors)."
            )

    dimensions, skipped_dims = collect_dimensions(workflow)
    objectives, skipped_objs = collect_objectives(workflow, measurables)

    if baseline:
        baseline["params"] = {
            d.address: _read_param(workflow, d.address) for d in dimensions
        }
        baseline["measured"] = {o.name: measurables.get(o.measures) for o in objectives}

    return OptimizationSpec(
        dimensions=dimensions,
        objectives=objectives,
        algorithm=algorithm or AlgorithmConfig(),
        baseline=baseline,
        skipped=skipped_dims + skipped_objs,
    )


def _read_param(workflow, address: str) -> Any:
    from .addressing import split_address

    node_name, param, keys = split_address(address)
    value = workflow.nodes[node_name]._parameters[param]
    for k in keys:
        value = value[k]
    return value
