"""Optimizer backends behind one ask / tell interface.

The engine never talks to a search library directly. It asks for a generation of
candidates, evaluates them, and tells the fitnesses back. Everything else —
which algorithm, how it adapts, whether it is population-based — lives behind
this interface.

Adding an algorithm:

* if the library is Optuna, add one entry to ``_OPTUNA_SAMPLERS``
* otherwise, subclass ``Optimizer`` and call ``register_optimizer``

Fitness is always a **list**, one entry per objective, and always **minimized**
(the engine converts targets into distances). Single-objective backends refuse a
multi-objective spec rather than silently collapsing it into a weighted sum.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from .spec import AlgorithmConfig, Dimension


class Optimizer(ABC):
    """Proposes parameter vectors and learns from their fitnesses."""

    #: Can this backend handle more than one objective at a time?
    supports_multi_objective: bool = False

    def __init__(
        self,
        dimensions: Sequence[Dimension],
        n_objectives: int,
        config: AlgorithmConfig,
    ):
        if n_objectives > 1 and not self.supports_multi_objective:
            raise ValueError(
                f"{type(self).__name__} is single-objective but the spec declares "
                f"{n_objectives} objectives. Choose a multi-objective algorithm "
                f"(e.g. 'nsga2') — the engine will not collapse them into a "
                f"weighted sum on its own."
            )
        self.dimensions = list(dimensions)
        self.n_objectives = n_objectives
        self.config = config
        self.last_ask_sources: List[str] = []

    @abstractmethod
    def ask(self) -> List[Dict[str, float]]:
        """Propose the next generation as a list of {address: value} dicts."""

    @abstractmethod
    def tell(
        self, candidates: List[Dict[str, float]], fitnesses: List[List[float]]
    ) -> None:
        """Report the measured fitness of each candidate from the last ``ask``."""

    def enqueue(self, params: Dict[str, float]) -> None:
        """Seed a chosen candidate into the next generation.

        Backends that cannot honour this should leave it as a no-op; the engine
        logs whether an injection was accepted.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support injecting candidates"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Callable[..., Optimizer]] = {}


def register_optimizer(name: str, factory: Callable[..., Optimizer]) -> None:
    _REGISTRY[name] = factory


def available() -> List[str]:
    return sorted(_REGISTRY)


def create_optimizer(
    dimensions: Sequence[Dimension], n_objectives: int, config: AlgorithmConfig
) -> Optimizer:
    if config.name not in _REGISTRY:
        raise ValueError(
            f"Unknown algorithm {config.name!r}. Available: {', '.join(available())}"
        )
    return _REGISTRY[config.name](dimensions, n_objectives, config)


# ---------------------------------------------------------------------------
# RandomSearch — no dependencies, so the engine is always runnable
# ---------------------------------------------------------------------------


class RandomSearch(Optimizer):
    """Uniform sampling inside the bounds. A baseline, and the smoke test."""

    supports_multi_objective = True

    def __init__(self, dimensions, n_objectives, config):
        super().__init__(dimensions, n_objectives, config)
        self._rng = random.Random(config.seed)
        self._queued: List[Dict[str, float]] = []

    def ask(self) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        sources: List[str] = []
        while self._queued and len(out) < self.config.pop_size:
            out.append(self._queued.pop(0))
            sources.append("inject")
        while len(out) < self.config.pop_size:
            out.append(
                {d.address: self._rng.uniform(d.low, d.high) for d in self.dimensions}
            )
            sources.append("ask")
        self.last_ask_sources = sources
        return out

    def tell(self, candidates, fitnesses) -> None:
        pass  # memoryless by design

    def enqueue(self, params: Dict[str, float]) -> None:
        self._queued.append(dict(params))


register_optimizer("random", RandomSearch)


# ---------------------------------------------------------------------------
# Optuna backends — CMA-ES, NSGA-II, NSGA-III, TPE, ...
# ---------------------------------------------------------------------------

#: name -> (optuna sampler class attribute, multi-objective?, extra sampler kwargs)
_OPTUNA_SAMPLERS: Dict[str, tuple] = {
    "cmaes": ("CmaEsSampler", False, {}),
    "nsga2": ("NSGAIISampler", True, {}),
    "nsga3": ("NSGAIIISampler", True, {}),
    "tpe": ("TPESampler", True, {}),
    "optuna_random": ("RandomSampler", True, {}),
}


def sampler_init_kwargs(
    sampler_name: str,
    config: AlgorithmConfig,
    sampler_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Keyword arguments passed to the Optuna sampler constructor.

    NSGA-II/III take ``population_size``; CMA-ES takes ``popsize``. Passing the
    wrong name is silently ignored, so CMA-ES would otherwise run at Optuna's
    default population (~4+3*ln(n)) and throw away the rest of each generation.
    """
    kwargs = dict(sampler_kwargs or {})
    kwargs.update(config.options)
    if config.seed is not None:
        kwargs.setdefault("seed", config.seed)
    if sampler_name.startswith("NSGA"):
        kwargs.setdefault("population_size", config.pop_size)
    elif sampler_name == "CmaEsSampler":
        kwargs.setdefault("popsize", config.pop_size)
    return kwargs


class OptunaOptimizer(Optimizer):
    """Adapter over Optuna's ask/tell API.

    Optuna hands out one trial at a time; a generation is simply ``pop_size``
    consecutive asks. Population-based samplers (NSGA-II/III) manage their own
    generations internally, so this stays correct for them too.
    """

    def __init__(
        self,
        dimensions,
        n_objectives,
        config,
        sampler_name: str,
        multi_objective: bool,
        sampler_kwargs: Dict[str, Any],
    ):
        self.supports_multi_objective = multi_objective
        super().__init__(dimensions, n_objectives, config)

        try:
            import optuna
            from optuna.distributions import FloatDistribution
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                f"The {config.name!r} algorithm needs Optuna. "
                f'Install it with: pip install -e ".[optimization]"'
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler_cls = getattr(optuna.samplers, sampler_name, None)
        if sampler_cls is None:  # pragma: no cover - depends on optuna version
            raise ImportError(
                f"This Optuna version has no {sampler_name} "
                f"(needed for algorithm {config.name!r})"
            )

        kwargs = sampler_init_kwargs(sampler_name, config, sampler_kwargs)

        self._optuna = optuna
        self._distributions = {
            d.address: FloatDistribution(d.low, d.high) for d in self.dimensions
        }
        self._study = optuna.create_study(
            directions=["minimize"] * n_objectives,
            sampler=sampler_cls(**kwargs),
        )
        self._pending: List[Any] = []
        self._inject_remaining = 0

    def ask(self) -> List[Dict[str, float]]:
        self._pending = [
            self._study.ask(self._distributions) for _ in range(self.config.pop_size)
        ]
        sources = []
        for _ in self._pending:
            if self._inject_remaining > 0:
                sources.append("inject")
                self._inject_remaining -= 1
            else:
                sources.append("ask")
        self.last_ask_sources = sources
        return [dict(t.params) for t in self._pending]

    def tell(self, candidates, fitnesses) -> None:
        for trial, fitness in zip(self._pending, fitnesses):
            if fitness is None:
                self._study.tell(trial, state=self._optuna.trial.TrialState.FAIL)
            else:
                self._study.tell(trial, list(fitness))
        self._pending = []

    def enqueue(self, params: Dict[str, float]) -> None:
        self._study.enqueue_trial(dict(params), skip_if_exists=False)
        self._inject_remaining += 1


def _make_optuna_factory(sampler_name: str, multi: bool, kwargs: Dict[str, Any]):
    def factory(dimensions, n_objectives, config):
        return OptunaOptimizer(
            dimensions, n_objectives, config, sampler_name, multi, kwargs
        )

    return factory


for _name, (_sampler, _multi, _kwargs) in _OPTUNA_SAMPLERS.items():
    register_optimizer(_name, _make_optuna_factory(_sampler, _multi, _kwargs))
