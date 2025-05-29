"""
Example script demonstrating a properly connected workflow in NeuroWorkflow.

This script shows how to create a workflow with connected nodes for neuron
parameter optimization, where the optimization loop is at the workflow level
and nodes are properly connected using the workflow builder.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from neuroworkflow.core.workflow import WorkflowBuilder, Workflow
from neuroworkflow.nodes.optimization import NeuronOptimizationNode
from neuroworkflow.nodes.simulation import NeuronSimulationNode
from neuroworkflow.nodes.stimulus import StimulusGeneratorNode
from neuroworkflow.nodes.network import NESTNeuronSetupNode




def run_optimization_workflow(workflow: Workflow, setup_node: NESTNeuronSetupNode, 
                            stimulus_node: StimulusGeneratorNode,
                            simulation_node: NeuronSimulationNode, 
                            optimization_node: NeuronOptimizationNode,
                            simulation_time: float, dt: float,
                            objective_target: float, max_iterations: int = 20) -> Dict[str, Any]:
    """Run the optimization workflow.
    
    Args:
        workflow: The workflow to run
        setup_node: NeuronSetupNode
        stimulus_node: StimulusGeneratorNode
        simulation_node: NeuronSimulationNode
        optimization_node: NeuronOptimizationNode
        simulation_time: Simulation time
        dt: Time step
        objective_target: Target value for the objective function
        max_iterations: Maximum number of iterations
        
    Returns:
        Optimization results
    """
    # Reset optimization state
    optimization_node.reset_optimization()
    
    # Initial parameters - start with default values
    # For the first iteration, we need to provide parameters directly
    # After that, the optimization node will feed parameters to the setup node
    initial_parameters = {}
    
    # Run optimization loop
    iteration = 0
    continue_optimization = True
    
    while continue_optimization and iteration < max_iterations:
        print(f"\nIteration {iteration}:")
        
        #if iteration == 0:
            # For the first iteration, we need to manually set up the neuron and stimulus
            # since the optimization node hasn't provided parameters yet
        #    setup_result = setup_node.create_neuron(initial_parameters)
            
            # Generate the initial stimulus
        #    stimulus_result = stimulus_node.generate_stimulus()
                #simulation_time=simulation_time,
                #dt=dt
            #)
        #else:
            # For subsequent iterations, the parameters will flow through the connected workflow
            # The optimization node's output is connected to the setup node's input
        #    pass
            
        # Run simulation - the neuron model and input current will be automatically passed
        # from the setup node and stimulus node respectively
        #sim_result = simulation_node.simulate(
        #    simulation_time=simulation_time
        #)
        
        # Evaluate results with the objective target
        # The simulation results will be automatically passed from the simulation node
        eval_result = optimization_node.evaluate(
            objective_target=objective_target, 
            iteration=iteration
        )
        
        error = eval_result['error']
        print(f"  Error: {error}")
        
        # The optimization node will automatically suggest new parameters
        # and pass them to the setup node through the connected workflow
        
        # Check if we should continue
        if iteration >= max_iterations - 1:
            print("Reached maximum iterations")
            continue_optimization = False
        else:
            iteration += 1
    
    # Get final optimization results
    results = optimization_node.get_optimization_results()
    
    # Print best parameters
    print("\nOptimization complete!")
    print("Best parameters:")
    for name, value in results['best_parameters'].items():
        print(f"  {name}: {value}")
    print(f"Best error: {results['best_error']}")
    
    return results


def main():
    """Run the properly connected workflow example."""
    # Create nodes
    neuron_node = NESTNeuronSetupNode("neuron_node")
    stimulus_node = StimulusGeneratorNode("stimulus_node")
    simulation_node = NeuronSimulationNode("simulation_node")
    optimization_node = NeuronOptimizationNode("optimization_node")
    
    # Configure nodes
    neuron_node.configure(
        nest_model = 'iaf_psc_alpha',
        threshold = -55.0,
        resting_potential = -70.0,
        time_constant = 20.0,
        refractory_period = 2.0
        )

    optimization_node.configure(optimization_method='grid')

    stimulus_node.configure(
        stimulus_type='step',
        amplitude=100.0,
        start_time=250.0,
        end_time=750.0
    )

    simulation_node.configure(
        dt = 0.1,
        simulation_time = 1000.0
    )

    
    # Create a workflow with connected nodes
    workflow = (
        WorkflowBuilder("neuron_optimization_workflow")
        .add_node(neuron_node)
        .add_node(stimulus_node)
        .add_node(simulation_node)
        .add_node(optimization_node)
        # Connect setup node to simulation node
        .connect("neuron_node", "nest_neuron", "simulation_node", "nest_neuron")
        .connect("neuron_node", "nest_neuron_config", "simulation_node", "nest_neuron_config")
        # Connect stimulus node to simulation node
        .connect("stimulus_node", "input_current", "simulation_node", "input_current")
        # Connect simulation node to optimization node
        .connect("simulation_node", "simulation_results", "optimization_node", "simulation_results")
        # Connect optimization node back to setup node to complete the loop
        .connect("optimization_node", "parameters", "neuron_node", "parameters")
        .build()
    )
    
    # Define simulation parameters
    simulation_time = 200.0  # ms
    dt = 0.1  # ms
    objective_target = 10  # Target number of spikes
    
    # Run optimization workflow
    print("\nRunning optimization workflow...")
    results = run_optimization_workflow(
        workflow=workflow,
        setup_node=neuron_node,
        stimulus_node=stimulus_node,
        simulation_node=simulation_node,
        optimization_node=optimization_node,
        simulation_time=simulation_time,
        dt=dt,
        objective_target=objective_target,
        max_iterations=20
    )
    
    # Get best simulation results
    best_simulation = results['best_simulation']
    best_params = results['best_parameters']
    history = results['history']
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot input current
    plt.subplot(3, 1, 1)
    t = np.arange(0, simulation_time, dt)
    plt.plot(t, input_current)
    plt.title('Input Current')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (nA)')
    
    # Plot membrane potential
    plt.subplot(3, 1, 2)
    plt.plot(best_simulation['time'], best_simulation['membrane_potential'])
    plt.axhline(y=best_simulation['parameters']['threshold'], color='r', linestyle='--', label='Threshold')
    plt.title('Membrane Potential (Optimized)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    
    # Plot optimization history
    plt.subplot(3, 1, 3)
    iterations = [h['iteration'] for h in history]
    errors = [h['error'] for h in history]
    plt.plot(iterations, errors, 'o-')
    plt.title('Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('properly_connected_workflow.png')
    print("\nPlot saved as 'properly_connected_workflow.png'")


if __name__ == "__main__":
    main()