import json
from typing import Any, Dict, List

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, ParameterDefinition


class NW_Optimization(Node):
    """Declares a study: how to search a workflow's parameter space, over what,
    and toward which targets.

    All three parts of a study live here — the algorithm and its budget, the
    parameters to explore with their ranges (``explore``), and the targets to hit
    with the outputs they are measured from (``objectives``). A person reviews
    the whole study in one place, an editor edits one node, and two studies over
    the same model are two of these nodes. The ``optimizable`` /
    ``optimization_range`` / ``unit`` fields a node author may put on a parameter
    are defaults an editor prefills a range from; the engine reads this node.

    It is not a step in the workflow. A workflow is a DAG that runs once; a
    search is a loop that runs it many times, so the loop cannot be a node inside
    the graph. Placing this node marks a workflow as one to optimize, and the code
    generator emits an optimization run instead of a single execution. It has no
    ports and no process steps, so adding it to a workflow changes nothing about
    how that workflow executes.

    Typical use, once the workflow is built::

        opt = NW_Optimization("opt")
        opt.configure(
            algorithm="cmaes", pop_size=16, max_generations=12,
            explore=[
                {"address": "clamp.amp_na", "low": 100.0, "high": 1000.0, "unit": "nA"},
                {"address": "exc.nest_params.I_e", "low": 0.0, "high": 400.0},
            ],
            objectives=[
                {"name": "exc_rate", "measures": "ana.firing_rate_hz.exc",
                 "low": 40.0, "high": 50.0, "unit": "Hz"},
            ],
        )
        spec   = opt.build_spec(workflow)     # baseline run, validation, clipping
        result = optimize(workflow, spec=spec, results_path=opt.results_path())
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_optimization",
        stage="optimization",
        tool="custom",
        model_source="https://github.com/oist/neuro-workflow",
        description=(
            "Declares a study over a workflow's parameter space: which algorithm "
            "with what budget, which parameters to explore over which ranges, and "
            "which measured outputs to steer into which target ranges. Its presence "
            "marks the workflow as one to optimize; it takes no part in the "
            "workflow's own execution."
        ),
        parameters={
            "algorithm": ParameterDefinition(
                default_value="cmaes",
                description=(
                    "Search algorithm. 'cmaes' is the default for a single objective and "
                    "continuous parameters; 'tpe' when each trial is expensive; 'nsga2' "
                    "or 'nsga3' for several objectives, returning a Pareto front. "
                    "'random' is uniform sampling — a baseline to beat and a way to "
                    "smoke-test the loop, not a search, so do not leave it selected for "
                    "a real study. Everything except 'random' needs Optuna: "
                    'pip install -e ".[optimization]".'
                ),
                constraints={
                    "allowed_values": [
                        "random",
                        "cmaes",
                        "tpe",
                        "nsga2",
                        "nsga3",
                        "optuna_random",
                    ]
                },
            ),
            "pop_size": ParameterDefinition(
                default_value=16,
                description=(
                    "Candidates proposed per generation, i.e. workflow runs before the "
                    "algorithm learns from them and proposes the next batch."
                ),
                constraints={"min": 1},
            ),
            "max_generations": ParameterDefinition(
                default_value=20,
                description=(
                    "Generation budget. At most pop_size x max_generations runs, plus one "
                    "baseline. For a single in_range objective the search also stops once a "
                    "candidate lands inside the target range; with several objectives it "
                    "keeps going to develop the Pareto front (stop_when_reached defaults "
                    "to False)."
                ),
                constraints={"min": 1},
            ),
            "seed": ParameterDefinition(
                default_value=None,
                description=(
                    "Random seed, making a run reproducible. Leave empty for an unseeded run."
                ),
            ),
            "options": ParameterDefinition(
                default_value={},
                description=(
                    "Extra keyword arguments passed straight to the underlying sampler, e.g. "
                    '{"sigma0": 0.2} for CMA-ES or {"n_startup_trials": 20} for TPE. '
                    "Names are the sampler's own."
                ),
            ),
            "explore": ParameterDefinition(
                default_value=[],
                description=(
                    "Parameters to search, one entry per axis. Each entry is a dict with "
                    "'address' ('Node.parameter' or 'Node.parameter.key' for one key of "
                    "a dict parameter, Node being the node's instance name), 'low' and "
                    "'high' (the range to search; clipped to the parameter's min/max "
                    "constraints), and optionally 'unit' (display only; defaults to the "
                    "parameter's unit) and 'integer' (true to round proposals to whole "
                    "numbers; inferred from the declared default when omitted). Prefill a "
                    "range from the parameter's optimization_range, constraints and unit."
                ),
            ),
            "objectives": ParameterDefinition(
                default_value=[],
                description=(
                    "Targets to steer toward, one entry per objective. Each entry is a "
                    "dict with 'name' (a label for reports), 'measures' (the output the "
                    "achieved value is read from, as 'Node.output_port' or "
                    "'Node.output_port.key' — take it from discover_measurables() after a "
                    "run), 'goal' ('in_range', the default, or 'minimize' / 'maximize'), "
                    "'low' and 'high' (the target range, for in_range) and optionally "
                    "'unit' (display only). Two or more objectives need 'nsga2' or 'nsga3'."
                ),
            ),
            "results_path": ParameterDefinition(
                default_value="results/optimization",
                description=(
                    "Directory the run is written under. Each run creates its own "
                    "subdirectory holding the manifest, the per-trial log, live status and "
                    "the per-trial results."
                ),
            ),
        },
        inputs={},
        outputs={},
        methods={},
    )

    def __init__(self, name: str):
        super().__init__(name)

    def algorithm_config(self):
        """This node's settings as an ``AlgorithmConfig`` for ``build_spec()``."""
        from neuroworkflow.optimization import AlgorithmConfig

        p = self._parameters
        seed = p["seed"]
        return AlgorithmConfig(
            name=str(p["algorithm"]),
            pop_size=int(p["pop_size"]),
            max_generations=int(p["max_generations"]),
            seed=None if seed in (None, "") else int(seed),
            options=dict(p["options"] or {}),
        )

    @staticmethod
    def _entries(value: Any, what: str) -> List[Dict[str, Any]]:
        """A study list as a list of dicts.

        Accepts the list itself, or its JSON text — the form a value typed into
        an editor's text field arrives in. An empty string is an empty list.
        """
        if isinstance(value, str):
            value = json.loads(value) if value.strip() else []
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"{what} must be a list of dicts, got {type(value).__name__}"
            )
        entries = list(value)
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"{what}[{index}] must be a dict, got {type(entry).__name__}"
                )
        return entries

    def explore_entries(self) -> List[Dict[str, Any]]:
        """The parameters to explore, as the entries ``build_spec()`` reads."""
        return self._entries(self._parameters["explore"], "explore")

    def objective_entries(self) -> List[Dict[str, Any]]:
        """The targets, as the entries ``build_spec()`` reads."""
        return self._entries(self._parameters["objectives"], "objectives")

    def build_spec(self, workflow, run_baseline: bool = True, strict: bool = True):
        """The study as an ``OptimizationSpec``, resolved against ``workflow``.

        Runs the baseline (unless ``run_baseline=False``), clips ranges to the
        parameters' constraints and checks every ``measures`` address against
        what the baseline produced. With ``strict=True`` an ``explore`` entry that
        cannot be searched is an error naming the entry and the reason; with
        ``strict=False`` it is only reported in ``spec.skipped``.
        """
        from neuroworkflow.optimization import build_spec

        spec = build_spec(
            workflow,
            algorithm=self.algorithm_config(),
            run_baseline=run_baseline,
            explore=self.explore_entries(),
            objectives=self.objective_entries(),
        )
        if strict and spec.skipped:
            problems = "\n".join(
                f"  {s['address']}: {s['reason']}" for s in spec.skipped
            )
            raise ValueError(
                f"{self.name}: these explore entries cannot be searched:\n{problems}"
            )
        return spec

    def results_path(self) -> str:
        """Where the run's ledger and per-trial results are written."""
        return str(self._parameters["results_path"])

    def summary(self) -> Dict[str, Any]:
        """The declared settings, for printing or logging."""
        return dict(self._parameters)
