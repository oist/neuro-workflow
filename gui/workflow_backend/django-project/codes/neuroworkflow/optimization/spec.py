"""The optimization spec — what to tune, what to hit, and with which algorithm.

The spec is plain data. It is built from a study — the ``explore`` and
``objectives`` entries an ``NW_Optimization`` node holds — or, absent one, from the
``optimizable`` flags in the node schemas; then handed to a human or an agent to
review and edit, then consumed by the engine and frozen into ``run.json``. Nothing in it is Python-specific, so an agent can change
the algorithm, widen a range or retarget a band without touching code.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from neuroworkflow.core.schema import ParameterDefinition

from .addressing import _is_number, discover_measurables, split_address


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
                "No search dimensions — list entries in NW_Optimization.explore "
                "(or build_spec(explore=...)), mark a parameter optimizable=True "
                "with a range, or add a Dimension by hand."
            )
        if not self.objectives:
            raise ValueError(
                "No objectives — optimization without a target has no direction. "
                "List entries in NW_Optimization.objectives (or "
                "build_spec(objectives=...)), or call spec.add_objective()."
            )
        if self.algorithm.pop_size < 1 or self.algorithm.max_generations < 1:
            raise ValueError(
                f"Algorithm budget: pop_size {self.algorithm.pop_size} and "
                f"max_generations {self.algorithm.max_generations} must both be >= 1"
            )
        for d in self.dimensions:
            if d.low >= d.high:
                raise ValueError(f"Dimension {d.address}: low {d.low} >= high {d.high}")
        for o in self.objectives:
            if o.goal not in ("in_range", "minimize", "maximize"):
                raise ValueError(f"Objective {o.name}: unknown goal {o.goal!r}")
            if o.goal == "in_range" and (o.low is None or o.high is None):
                raise ValueError(
                    f"Objective {o.name}: goal 'in_range' needs both low and high"
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
        """Declare a target.

        An objective is a label, a measurement address and a band — nothing about it
        requires a parameter to hang it on. It belongs to the study, not to the
        model, so it is never stored on a node parameter the simulation would not
        read.

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


def _numeric_pair(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return [float(value[0]), float(value[1])]
    return None


def _dimension_for(
    node,
    address: str,
    pdef: ParameterDefinition,
    key: str,
    low: float,
    high: float,
    *,
    unit: Optional[str] = None,
    integer: Optional[bool] = None,
    source: str,
) -> Tuple[Optional[Dimension], Optional[str]]:
    """One search axis for ``address``, or the reason there is none.

    ``key`` is the dict key for a ``Node.param.key`` address, empty for a scalar
    parameter. A scalar range is clipped to the parameter's ``constraints``; a
    per-key range is not, since constraints belong to the parameter as a whole.
    ``unit`` and ``integer`` override the schema's unit and the integer inference
    when given.
    """
    _, pname, _ = split_address(address)
    value = node._parameters.get(pname)
    note = ""

    if key:
        if isinstance(value, dict) and key not in value:
            return None, (
                "key is not present in the parameter's current value — "
                "configure() it first"
            )
        live = value.get(key) if isinstance(value, dict) else None
    else:
        if isinstance(value, dict):
            return None, (
                "dict parameter with no per-key optimization_range — "
                'declare one, e.g. {"V_th": [-60.0, -45.0]}'
            )
        live = value

        # Constraints are the authority: a range reaching past them describes
        # points configure() would reject anyway, so clip rather than fail.
        c_min = pdef.constraints.get("min")
        c_max = pdef.constraints.get("max")
        clipped = []
        if isinstance(c_min, (int, float)) and low < c_min:
            clipped.append(f"low raised to {c_min}")
            low = float(c_min)
        if isinstance(c_max, (int, float)) and high > c_max:
            clipped.append(f"high lowered to {c_max}")
            high = float(c_max)
        if clipped:
            source, note = "clipped", "; ".join(clipped) + " (constraints)"

    if live is not None and not _is_number(live):
        return None, f"current value is not a number: {live!r}"

    if low >= high:
        return None, f"empty range after clipping to constraints: [{low}, {high}]"

    if not isinstance(integer, bool):
        integer = _is_integer_axis(pdef, live, key=key)
    if integer and not _has_integer_inside(low, high):
        what = "integer value" if key else "integer parameter"
        return None, f"{what} but no whole number inside [{low}, {high}]"

    return (
        Dimension(
            address=address,
            low=low,
            high=high,
            unit=pdef.unit if unit is None else unit,
            description=pdef.description,
            source=source,
            note=note,
            integer=integer,
        ),
        None,
    )


def collect_dimensions(workflow) -> tuple:
    """Collect search dimensions from every ``optimizable=True`` parameter.

    Returns ``(dimensions, skipped)``. A parameter is skipped rather than
    guessed at when no usable range can be derived — the engine reports those so
    a human or agent can supply ranges deliberately.
    """
    dimensions: List[Dimension] = []
    skipped: List[Dict[str, str]] = []

    def keep(address: str, dim: Optional[Dimension], reason: Optional[str]) -> None:
        if dim is not None:
            dimensions.append(dim)
        else:
            skipped.append({"address": address, "reason": reason or ""})

    for node_name, node in workflow.nodes.items():
        params = getattr(node, "NODE_DEFINITION", None)
        if params is None:
            continue
        for pname, pdef in params.parameters.items():
            if not isinstance(pdef, ParameterDefinition) or not pdef.optimizable:
                continue

            address = f"{node_name}.{pname}"

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
                    keep(
                        key_address,
                        *_dimension_for(
                            node,
                            key_address,
                            pdef,
                            key,
                            rng[0],
                            rng[1],
                            source="schema.optimization_range",
                        ),
                    )
                continue

            c_min = pdef.constraints.get("min")
            c_max = pdef.constraints.get("max")
            rng = _numeric_pair(pdef.optimization_range)
            source = "schema.optimization_range"

            if rng is None:
                if isinstance(c_min, (int, float)) and isinstance(c_max, (int, float)):
                    rng, source = [float(c_min), float(c_max)], "schema.constraints"
                elif isinstance(node._parameters.get(pname), dict):
                    # Reported by _dimension_for, with the per-key hint.
                    rng = [0.0, 0.0]
                else:
                    skipped.append(
                        {
                            "address": address,
                            "reason": "optimizable but no optimization_range and no "
                            "min/max constraints to fall back on",
                        }
                    )
                    continue

            keep(
                address,
                *_dimension_for(node, address, pdef, "", rng[0], rng[1], source=source),
            )

    return dimensions, skipped


def dimensions_from_study(workflow, explore: Sequence[Mapping[str, Any]]) -> tuple:
    """Search dimensions from explicit study entries.

    Each entry is a mapping with ``address`` (``"Node.param"`` or
    ``"Node.param.key"``), ``low`` and ``high``, and optionally ``unit`` and
    ``integer``. Anything else in an entry — an editor's ``node_id``, say — is
    not read. Returns ``(dimensions, skipped)``; an entry that cannot be searched
    is reported with the reason rather than guessed at.
    """
    dimensions: List[Dimension] = []
    skipped: List[Dict[str, str]] = []
    seen = set()

    for index, entry in enumerate(explore):
        address = entry.get("address") if isinstance(entry, Mapping) else None
        label = address if isinstance(address, str) and address else f"explore[{index}]"

        def skip(reason: str) -> None:
            skipped.append({"address": label, "reason": reason})

        if not isinstance(entry, Mapping):
            skip(f"entry must be a dict, got {type(entry).__name__}")
            continue
        if not isinstance(address, str) or address.count(".") < 1:
            skip("address must be 'Node.param' or 'Node.param.key'")
            continue
        node_name, pname, keys = split_address(address)
        if len(keys) > 1:
            skip("nested keys deeper than one level are not searchable")
            continue
        if address in seen:
            skip("declared twice")
            continue
        seen.add(address)

        node = workflow.nodes.get(node_name)
        if node is None:
            skip(
                f"no node named {node_name!r} "
                f"(have: {', '.join(sorted(workflow.nodes))})"
            )
            continue
        if pname not in node._parameters:
            skip(
                f"node {node_name!r} has no parameter {pname!r} "
                f"(have: {', '.join(sorted(node._parameters))})"
            )
            continue

        rng = _numeric_pair([entry.get("low"), entry.get("high")])
        if rng is None:
            skip("low and high must both be numbers")
            continue

        definition = getattr(node, "NODE_DEFINITION", None)
        pdef = definition.parameters.get(pname) if definition is not None else None
        if not isinstance(pdef, ParameterDefinition):
            # A dict-format legacy definition: judge integer-ness and unit from
            # what the node holds now.
            pdef = ParameterDefinition(default_value=node._parameters[pname])

        integer = entry.get("integer")
        unit = entry.get("unit")
        dim, reason = _dimension_for(
            node,
            address,
            pdef,
            keys[0] if keys else "",
            rng[0],
            rng[1],
            unit=unit if isinstance(unit, str) else None,
            integer=integer if isinstance(integer, bool) else None,
            source="study",
        )
        if dim is None:
            skip(reason or "")
        else:
            dimensions.append(dim)

    return dimensions, skipped


def build_spec(
    workflow,
    algorithm: Optional[AlgorithmConfig] = None,
    run_baseline: bool = True,
    explore: Optional[Sequence[Mapping[str, Any]]] = None,
    objectives: Optional[Sequence[Mapping[str, Any]]] = None,
) -> OptimizationSpec:
    """Produce a spec ready for review from a study, or from the node schemas.

    ``explore`` and ``objectives`` are the study's entries — the lists an
    ``NW_Optimization`` node holds. With ``explore`` given, the ``optimizable``
    flags on the nodes are ignored; without it (``None``) every ``optimizable=True``
    parameter is searched over its declared range, which is how a notebook
    without an optimization node declares a search. Objectives only ever come
    from the study: each entry goes through ``add_objective()``, so a
    ``measures`` address that the baseline did not produce fails here, naming
    what is available.

    With ``run_baseline=True`` the workflow is executed once at its current
    parameters. That run does two jobs: it is the reference point every result is
    compared against, and its outputs are what ``measures`` addresses resolve
    against.
    """
    baseline: Dict[str, Any] = {}
    measurables: Dict[str, float] = {}

    if run_baseline:
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

    if explore is None:
        dimensions, skipped = collect_dimensions(workflow)
    else:
        dimensions, skipped = dimensions_from_study(workflow, explore)

    spec = OptimizationSpec(
        dimensions=dimensions,
        algorithm=algorithm or AlgorithmConfig(),
        baseline=baseline,
        skipped=skipped,
    )

    for index, entry in enumerate(objectives or []):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"objectives[{index}] must be a dict, got {type(entry).__name__}"
            )
        missing = [k for k in ("name", "measures") if not entry.get(k)]
        if missing:
            raise ValueError(
                f"objectives[{index}] needs {' and '.join(missing)}: {dict(entry)}"
            )
        spec.add_objective(
            name=str(entry["name"]),
            measures=str(entry["measures"]),
            low=entry.get("low"),
            high=entry.get("high"),
            goal=str(entry.get("goal") or "in_range"),
            unit=str(entry.get("unit") or ""),
            description=str(entry.get("description") or ""),
        )

    if baseline:
        baseline["params"] = {
            d.address: _read_param(workflow, d.address) for d in spec.dimensions
        }
        baseline["measured"] = {
            o.name: measurables.get(o.measures) for o in spec.objectives
        }

    return spec


def _read_param(workflow, address: str) -> Any:
    from .addressing import split_address

    node_name, param, keys = split_address(address)
    value = workflow.nodes[node_name]._parameters[param]
    for k in keys:
        value = value[k]
    return value
