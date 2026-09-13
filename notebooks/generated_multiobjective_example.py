#!/usr/bin/env python3
"""
Generated workflow for project: Network and probe, two objectives

Same shape as generated_optimization_example.py, with two differences that matter:

  * a second population, `probe`: a SINGLE inhibitory neuron that presses down on the
    network and receives no connection from it - inhibition, not a read-out.

    It is NOT independent of the drive, though: BMTK applies the current clamp to the
    whole network, so `clamp.amp_na` reaches the probe as well as `exc`. Measured at
    probe I_e = 100 pA: amp 300 -> probe ISI 29.8 ms, amp 450 -> 13.5 ms. That shared
    current is what makes the two objectives pull against each other, and it is why
    `probe.tau_m` matters - it moves the probe's own threshold, which is the only way to
    change the probe's rate without changing the drive to the network.

        clamp --> exc --> exc      drive and recurrent excitation, both fixed
                   ^
                   +---- probe     inhibition, negative weight

  * TWO objectives, so there is no single best answer. NSGA-II returns a Pareto front:
    the set of configurations where improving one objective costs the other.

    1. the network's firing rate             ana.firing_rate_hz.exc
    2. the probe's mean inter-spike interval  ana.isi_stats.probe.mean_ms

Why these two compete: the probe's own I_e and tau_m decide how fast it fires, which
decides how much inhibition the network receives, which decides the network's rate.
Two knobs, two targets, one causal path between them.

Four dimensions: the drive, the recurrent excitatory weight, and the probe's I_e and
tau_m. The probe's two are read at simulation time, but the recurrent weight is written
into the SONATA edge files - so every trial rebuilds the network.

The probe's inhibition of the network is fixed at -60 pA, so what the search controls
is how hard the network is driven and how fast the probe fires against it.

Run it directly:

    python generated_multiobjective_example.py

Needs Optuna:  pip install optuna cmaes
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
    """Optimize a neural simulation workflow against two objectives."""

    # workflow_builder creation
    workflow_builder = WorkflowBuilder(
        "Network_and_probe",
        context={
            "results_path": "./results/multiobjective_example"
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

    # A single inhibitory neuron. It receives nothing from the network: its firing comes
    # from its own injected current, and it inhibits the network. I_e and tau_m are the
    # two tuned values, so they set both how fast it fires and how hard it inhibits.
    probe = NW_Population("probe")
    probe.configure(
        pop_name='probe',
        N=1,
        model_type='point_neuron',
        model_template='nest:iaf_psc_alpha',
        ei_type='inh',
        location='VISp',
        layer='L4',
        nest_params={'C_m': 250.0, 'tau_m': 10.0, 't_ref': 2.0, 'V_th': -55.0, 'V_reset': -70.0, 'E_L': -70.0, 'I_e': 200.0}
    )

    # E->E does not set its own weight, so it inherits the node-level syn_weight - the
    # value the search tunes, i.e. the strength of recurrent excitation. probe->exc
    # carries its own, fixed: a negative weight is what makes a synapse inhibitory in
    # NEST; ei_type is metadata carried into SONATA, not what inhibits.
    conn = NW_Connectivity("conn")
    conn.configure(
        connection_rule=1,
        syn_weight=5.0,
        connections=[
            {'source': 'exc', 'target': 'exc'},
            {'source': 'probe', 'target': 'exc', 'syn_weight': -60.0}
        ]
    )

    # Simulation
    sim = NW_SimConfig("sim")
    sim.configure(
        simulator='pointnet',
        config_file='config_multiobjective_example.json',
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
    # workflow's own execution. Two objectives require a multi-objective algorithm;
    # a single-objective one refuses rather than inventing weights between them.
    opt = NW_Optimization("opt")
    opt.configure(
        algorithm='nsga2',
        pop_size=12,
        max_generations=15,
        seed=1,
        results_path='./results/multiobjective_example/optimization'
    )

    # workflow_builder_ready
    workflow_builder.add_node(clamp)
    workflow_builder.add_node(exc)
    workflow_builder.add_node(probe)
    workflow_builder.add_node(conn)
    workflow_builder.add_node(sim)
    workflow_builder.add_node(ana)

    workflow_builder.connect("clamp", "iclamp", "exc", "iclamp")
    workflow_builder.connect("exc", "population", "conn", "populations")
    workflow_builder.connect("probe", "population", "conn", "populations")
    workflow_builder.connect("conn", "network", "sim", "populations")
    workflow_builder.connect("sim", "results", "ana", "results")

    workflow = workflow_builder.build()

    # Print workflow information
    print(workflow)

    # Parameters marked optimizable in the editor
    # [300, 600] rather than [100, 1000]: the network is silent below ~390 pA and
    # saturating above ~500, so most of the wider range carried no information. Measured
    # while checking this example: 380 -> 0 Hz, 400 -> 48 Hz, 500 -> 98 Hz.
    clamp.NODE_DEFINITION.parameters["amp_na"].optimizable = True
    clamp.NODE_DEFINITION.parameters["amp_na"].optimization_range = [300.0, 600.0]
    clamp.NODE_DEFINITION.parameters["amp_na"].unit = "nA"

    # Reaches E->E, the only connection that does not set its own weight.
    conn.NODE_DEFINITION.parameters["syn_weight"].optimizable = True
    conn.NODE_DEFINITION.parameters["syn_weight"].optimization_range = [1.0, 50.0]
    conn.NODE_DEFINITION.parameters["syn_weight"].unit = "pA"

    # Keys inside a dict parameter get one range each. These belong to `probe` alone:
    # each node instance carries its own definition, so `exc` is unaffected.
    probe.NODE_DEFINITION.parameters["nest_params"].optimizable = True
    # tau_m sets the probe's own threshold current (C_m/tau_m x 15 mV), which is the only
    # lever that moves the probe's rate independently of the clamp. Both ranges are kept
    # tight around the region where the two targets can hold together: measured at
    # amp_na = 400, I_e = 100, tau_m = 8 the network runs at 48 Hz with a probe ISI of
    # 24.2 ms, inside both bands. Widen them and the search spends its budget in the part
    # of the space where the network is either silent or saturated.
    probe.NODE_DEFINITION.parameters["nest_params"].optimization_range = {
        'I_e': [0.0, 250.0],
        'tau_m': [6.0, 14.0]
    }

    # Execute optimization
    print("\nOptimizing workflow...")
    spec = build_spec(workflow, opt.algorithm_config())

    # Objectives declared by the study. Two of them, so the result is a Pareto front
    # rather than a single answer.
    spec.add_objective(
        name="network_rate",
        measures="ana.firing_rate_hz.exc",
        low=40.0,
        high=50.0,
        unit="Hz"
    )
    spec.add_objective(
        name="probe_isi",
        measures="ana.isi_stats.probe.mean_ms",
        low=20.0,
        high=40.0,
        unit="ms"
    )

    result = optimize(workflow, spec=spec, results_path=opt.results_path())

    if result.best is None:
        print("Optimization failed: no trial produced a usable measurement!")
        return 1

    print(f"Optimization finished: {result.stop_reason}")

    # With two objectives there is no single winner. The Pareto front is the set of
    # configurations where improving one objective would cost the other.
    # Measured values are given as measured, each in its own unit. The trailing figure
    # is the only derived one: the objective that missed by most, sized against the
    # target range asked for - "1.4x" meaning the miss is 1.4 times as wide as the range.
    units = {o.name: o.unit for o in spec.objectives}
    print(f"\nPareto front: {len(result.pareto_front)} configuration(s)")
    for row in result.pareto_front:
        measured = ", ".join(f"{k}={v:.4g} {units.get(k, '')}".rstrip()
                             for k, v in row["measured"].items())
        tuned = ", ".join(f"{k.split('.', 1)[1]}={v:.4g}" for k, v in row["params"].items())
        missed = row["target_ranges_off"]
        state = ("every target met" if not missed
                 else f"worst objective off by {missed:.3g}x its target range")
        print(f"  trial {row['trial']:>4}: {measured}   <-   {tuned}   [{state}]")

    # Picking one of them is a scientific judgement, not something the search can make.
    # apply_best() takes the configuration whose WORST objective missed by the least,
    # each miss sized against its own target range - the only way to weigh a miss in Hz
    # against a miss in ms. That is a reasonable default and nothing more: any member of
    # the front above may suit the question better, and applying a different one is one
    # configure() call.
    result.apply_best(workflow)

    measured = ", ".join(
        f"{o.name}={result.best['measured'][o.name]:.4g}"
        f"{' ' + o.unit if o.unit else ''} (target {o.low}-{o.high})"
        for o in spec.objectives
    )
    missed = result.best["target_ranges_off"]
    state = ("meets every target" if not missed
             else f"misses by {missed:.3g}x its target range on its worst objective")
    print(f"\nApplied one configuration of the {len(result.pareto_front)} on the front - "
          f"trial {result.best['trial']}, which {state}:")
    print(f"  it gives {measured}")
    print()
    print(result.configure_snippet())
    print(f"\nResults for this configuration: {result.best['results_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
