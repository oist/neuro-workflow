#!/usr/bin/env python3
"""Outer-loop parameter search (JointOptimizationNode was removed).

A workflow is a DAG that runs once. Search sits outside it:

    from neuroworkflow.optimization import AlgorithmConfig, build_spec, optimize

    spec = build_spec(workflow, AlgorithmConfig(name="random", pop_size=8, seed=1))
    result = optimize(workflow, spec=spec, results_path="./results/optimization")
    print(result.configure_snippet())

``random`` needs no extra packages. Optuna algorithms need
``pip install -e ".[optimization]"``.

A generated example of what the GUI should emit:

    notebooks/generated_optimization_example.py
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
