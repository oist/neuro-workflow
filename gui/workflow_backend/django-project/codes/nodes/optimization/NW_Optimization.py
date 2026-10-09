from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, ParameterDefinition,
)


class NW_Optimization(Node):
    """Declares how to search a workflow's parameter space.

    This node holds *how* to search — algorithm and budget. It deliberately does not
    hold *what* to search: that lives on the parameters themselves, as
    ``optimizable`` / ``optimization_range``, the fields the engine reads, so there
    is one source of truth. *What to hit* is an output port of some node: either a
    node author declares it on a parameter (``is_objective`` / ``objective_range`` /
    ``measures``), or the GUI stores the study's objectives on this node's FlowNode
    data (``data.study.objectives``) and the code generator emits them as
    ``spec.add_objective()`` calls.

    It is also not a step in the workflow. A workflow is a DAG that runs once; a
    search is a loop that runs it many times, so the loop cannot be a node inside
    the graph. Placing this node marks a workflow as one to optimize, and the code
    generator emits an optimization run instead of a single execution. It has no
    ports and no process steps, so adding it to a workflow changes nothing about
    how that workflow executes.

    Typical use, once the workflow is built::

        opt    = NW_Optimization("opt")
        opt.configure(algorithm="cmaes", pop_size=16, max_generations=12)

        spec   = build_spec(workflow, opt.algorithm_config())
        result = optimize(workflow, spec=spec, results_path=opt.results_path())
    """

    NODE_DEFINITION = NodeDefinitionSchema(
        type="nw_optimization",
        stage="optimization",
        tool="custom",
        model_source="https://github.com/oist/neuro-workflow",
        description=(
            "Declares how to search a workflow's parameter space: which algorithm and "
            "how large a budget. What to explore and what to hit are declared on the "
            "parameters themselves, not here. Its presence marks the workflow as one "
            "to optimize; it takes no part in the workflow's own execution."
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
                    "a real study. 'optuna_random' is the same uniform sampling run "
                    "through Optuna instead of the built-in sampler; prefer plain "
                    "'random', which needs no dependency, unless you specifically want "
                    "the trials recorded in an Optuna study. Everything except 'random' "
                    "needs Optuna: pip install -e \".[optimization]\". "
                    "\n\nOnly 'cmaes', 'nsga2' and 'nsga3' maintain a population, so for "
                    "them pop_size is part of the algorithm. 'tpe', 'random' and "
                    "'optuna_random' propose points one at a time, and for those "
                    "pop_size x max_generations is simply the number of runs - see those "
                    "two parameters."
                ),
                constraints={"allowed_values": ["random", "cmaes", "tpe",
                                                "nsga2", "nsga3", "optuna_random"]},
            ),
            "pop_size": ParameterDefinition(
                default_value=16,
                description=(
                    "Candidates proposed per generation, i.e. workflow runs before the "
                    "algorithm learns from them and proposes the next batch. "
                    "\n\nWhat it means depends on the algorithm. For 'cmaes', 'nsga2' and "
                    "'nsga3' this IS the algorithm's own population: it is handed to the "
                    "sampler, and it decides how many candidates are compared before the "
                    "search distribution (cmaes) or the Pareto front (nsga2/nsga3) is "
                    "updated. Changing it changes the search, not only its cost. For "
                    "'tpe', 'random' and 'optuna_random' there is no population - the "
                    "sampler proposes one point at a time - so this is only how many are "
                    "asked for before results are fed back, and 6x8 searches the same as "
                    "8x6. "
                    "\n\nRough guidance: 'nsga2'/'nsga3' need 16 or more, or the front is "
                    "too sparse to read; 'cmaes' works well at 6-12 for a handful of "
                    "dimensions; for 'tpe' and 'random' pick whatever total you can afford."
                ),
                constraints={"min": 1},
            ),
            "max_generations": ParameterDefinition(
                default_value=20,
                description=(
                    "Generation budget. At most pop_size x max_generations runs, plus one "
                    "baseline. For a single in_range objective the search also stops once a "
                    "candidate lands inside the target band; with several objectives it keeps "
                    "going to develop the Pareto front (stop_when_reached defaults to False). "
                    "\n\nFor 'cmaes', 'nsga2' and 'nsga3' a generation is a real step: the "
                    "population is scored, then the next one is drawn from what was learned, "
                    "so too few generations stops the search before it has adapted. For "
                    "'tpe' and 'random' only the product with pop_size matters."
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
                    "{\"sigma0\": 0.2} for CMA-ES or {\"n_startup_trials\": 20} for TPE. "
                    "Names are the sampler's own."
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

    def results_path(self) -> str:
        """Where the run's ledger and per-trial results are written."""
        return str(self._parameters["results_path"])

    def summary(self) -> Dict[str, Any]:
        """The declared settings, for printing or logging."""
        return dict(self._parameters)
