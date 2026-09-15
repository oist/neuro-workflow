#!/usr/bin/env python3
"""
Generated workflow for project: Clamp and weight optimization

EXAMPLE of what the code generator should emit when an NW_Optimization node is present
on the canvas. Everything above `workflow_builder.build()` is what the generator already
produces today — node creation, configure() with the edited parameters, add_node, connect.
Only the tail differs: instead of a single `workflow.execute()`, it declares what to
explore and what to hit, then runs the search.

Run it directly:

    python generated_optimization_example.py

Needs Optuna for the default algorithm:  pip install optuna cmaes
"""
import sys
import os
import numpy as np

# Add paths for JupyterLab environment
sys.path.append('../../')

from neuroworkflow.core.workflow import WorkflowBuilder

from neuroworkflow.nodes.stimulus.NW_IClamp import NW_IClamp
from neuroworkflow.nodes.network.NW_Population import NW_Population
from neuroworkflow.nodes.network.NW_Connectivity import NW_Connectivity
from neuroworkflow.nodes.simulation.NW_SimConfig import NW_SimConfig
from neuroworkflow.nodes.analysis.NW_Analysis import NW_Analysis
from neuroworkflow.nodes.optimization.NW_Optimization import NW_Optimization

from neuroworkflow.optimization import build_spec, optimize


def main():
    """Optimize a neural simulation workflow."""

    # workflow_builder creation
    workflow_builder = WorkflowBuilder(
        "Clamp_and_weight",
        context={
            "results_path": "./results/generated_example"
        }
    )

    # Create nodes

    # Stimulus
    clamp = NW_IClamp("clamp")
    clamp.configure(
        amp_na=200.0,
        delay_ms=50.0,
        duration_ms=450.0
    )

    # Network
    exc = NW_Population("exc")
    exc.configure(
        pop_name='exc',
        N=20,
        model_type='point_neuron',
        model_template='nest:iaf_psc_alpha',
        ei_type='exc',
        location='VISp',
        layer='L4',
        nest_params={'C_m': 250.0, 'tau_m': 10.0, 't_ref': 2.0, 'V_th': -55.0, 'V_reset': -70.0, 'E_L': -70.0, 'I_e': 0.0}
    )

    conn = NW_Connectivity("conn")
    conn.configure(
        connection_rule=1,
        syn_weight=5.0,
        connections=[{'source': 'exc', 'target': 'exc'}]
    )

    # Simulation
    sim = NW_SimConfig("sim")
    sim.configure(
        simulator='pointnet',
        config_file='config_generated_example.json',
        tstop_ms=500.0,
        dt_ms=0.1
    )

    # Analysis
    ana = NW_Analysis("ana")
    ana.configure(
        plot_raster=False,
        plot_traces=False
    )

    # Optimization
    # Not added to the workflow: it declares how to search, and takes no part in the
    # workflow's own execution.
    opt = NW_Optimization("opt")
    opt.configure(
        algorithm='cmaes',
        pop_size=16,
        max_generations=12,
        seed=1,
        results_path='./results/generated_example/optimization'
    )

    # workflow_builder_ready
    workflow_builder.add_node(clamp)
    workflow_builder.add_node(exc)
    workflow_builder.add_node(conn)
    workflow_builder.add_node(sim)
    workflow_builder.add_node(ana)

    workflow_builder.connect("clamp", "iclamp", "exc", "iclamp")
    workflow_builder.connect("exc", "population", "conn", "populations")
    workflow_builder.connect("conn", "network", "sim", "populations")
    workflow_builder.connect("sim", "results", "ana", "results")

    workflow = workflow_builder.build()

    # Print workflow information
    print(workflow)

    # Parameters marked optimizable in the editor
    clamp.NODE_DEFINITION.parameters["amp_na"].optimizable = True
    clamp.NODE_DEFINITION.parameters["amp_na"].optimization_range = [100.0, 1000.0]
    clamp.NODE_DEFINITION.parameters["amp_na"].unit = "nA"

    conn.NODE_DEFINITION.parameters["syn_weight"].optimizable = True
    conn.NODE_DEFINITION.parameters["syn_weight"].optimization_range = [1.0, 100.0]
    conn.NODE_DEFINITION.parameters["syn_weight"].unit = "pA"

    # Execute optimization
    print("\nOptimizing workflow...")
    spec = build_spec(workflow, opt.algorithm_config())

    # Objectives declared by the study.
    #
    # An objective is a label, a measurement address and a band - nothing about it
    # needs a parameter to hang it on. Declaring it here keeps the target out of the
    # model, so no node has to carry a parameter the simulation never reads, and a
    # workflow can be optimized towards different targets without editing its nodes.
    # This is the form the GUI will produce, where the study lives on the
    # NW_Optimization node rather than being scattered across the model.
    spec.add_objective(
        name="exc_firing_rate",
        measures="ana.firing_rate_hz.exc",
        low=40.0,
        high=50.0,
        unit="Hz"
    )

    # The other way, still supported: declare the target on a node parameter and let
    # build_spec() discover it. Useful when a node author ships a sensible default
    # target with the node, but it needs a parameter to exist for the purpose.
    #
    # exc.NODE_DEFINITION.parameters["mean_firing_rate"].is_objective = True
    # exc.NODE_DEFINITION.parameters["mean_firing_rate"].objective_range = [40.0, 50.0]
    # exc.NODE_DEFINITION.parameters["mean_firing_rate"].unit = "Hz"
    # exc.NODE_DEFINITION.parameters["mean_firing_rate"].measures = "ana.firing_rate_hz.exc"
    result = optimize(workflow, spec=spec, results_path=opt.results_path())

    # Not reaching the target is a result, not an error: the search ran and reported
    # that nothing in the declared ranges hits the band. Only a run that produced no
    # usable measurement at all has failed.
    if result.best is None:
        print("Optimization failed: no trial produced a usable measurement!")
        return 1

    print(f"Optimization finished: {result.stop_reason}")

    # The search leaves the workflow holding the LAST trial's values, not the best
    # ones. Adopting the winner is a separate, deliberate step.
    result.apply_best(workflow)

    print("\nApplied the following parameters to the workflow:")
    print(result.configure_snippet())
    print(f"\nResults for this configuration: {result.best['results_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
