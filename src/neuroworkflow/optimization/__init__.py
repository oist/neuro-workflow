"""Parameter optimization for NeuroWorkflow workflows.

The workflow is the objective function; the optimizer wraps it. Typical use::

    from neuroworkflow.optimization import build_spec, optimize, AlgorithmConfig

    spec = build_spec(workflow, AlgorithmConfig(name="cmaes", pop_size=12))
    print(spec.summary())          # review, edit, or hand to an agent
    result = optimize(workflow, spec=spec)
    print(result.configure_snippet())
"""

from .addressing import discover_measurables, read_output, set_parameter
from .engine import OptimizationResult, objective_fitness, optimize
from .ledger import Ledger
from .optimizers import Optimizer, available, register_optimizer
from .spec import (
    AlgorithmConfig,
    Dimension,
    Objective,
    OptimizationSpec,
    build_spec,
    collect_dimensions,
    collect_objectives,
)

__all__ = [
    "AlgorithmConfig",
    "Dimension",
    "Ledger",
    "Objective",
    "OptimizationResult",
    "OptimizationSpec",
    "Optimizer",
    "available",
    "build_spec",
    "collect_dimensions",
    "collect_objectives",
    "discover_measurables",
    "objective_fitness",
    "optimize",
    "read_output",
    "register_optimizer",
    "set_parameter",
]
