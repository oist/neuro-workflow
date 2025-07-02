#!/usr/bin/env python3
"""
Encapsulated optimization example using the NeuroWorkflow library.

This example demonstrates how to create a workflow for optimizing neuron parameters
without creating cycles in the workflow graph, while respecting node encapsulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the src directory to the Python path to import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from neuroworkflow.core.workflow import WorkflowBuilder
from neuroworkflow.nodes.optimization import JointOptimizationNode
from neuroworkflow.nodes.simulation import NeuronSimulationNode
from neuroworkflow.nodes.stimulus import StimulusGeneratorNode
from neuroworkflow.nodes.network import NESTNeuronSetupNode


def main():
    """Run an encapsulated optimization workflow example."""
    # Create nodes
    neuron_node = NESTNeuronSetupNode("neuron_node")
    stimulus_node = StimulusGeneratorNode("stimulus_node")
    simulation_node = NeuronSimulationNode("simulation_node")
    optimization_node = JointOptimizationNode("optimization_node")
    
    # Configure nodes with initial parameters
    neuron_node.configure(
        nest_model='iaf_psc_alpha',
        threshold=-55.0,
        resting_potential=-70.0,
        time_constant=20.0,
        refractory_period=2.0
    )
    
    stimulus_node.configure(
        stimulus_type='step',
        amplitude=100.0,
        start_time=250.0,
        end_time=750.0
    )
    
    # Define simulation parameters (these are not node configuration parameters)
    simulation_time = 1000.0
    dt = 0.1
    
    simulation_node.configure(
        dt=0.1,
        simulation_time=1000.0
    )
    
    optimization_node.configure(
        optimization_method='grid',
        grid_points=3,  # Use fewer points for faster execution
        max_iterations=120
    )
    
    # Initialize the stimulus generator with the simulation parameters
    # This needs to be done before creating the workflow
    stimulus_result = stimulus_node.generate_stimulus(
        simulation_time=simulation_time,
        dt=dt
    )
    
    # Create a workflow WITHOUT the feedback connections
    # This keeps the workflow acyclic (no cycles)
    workflow = (
        WorkflowBuilder("encapsulated_optimization_workflow")
        .add_node(neuron_node)
        .add_node(stimulus_node)
        .add_node(simulation_node)
        .add_node(optimization_node)
        
        # Connect neuron and stimulus nodes to simulation node
        .connect("neuron_node", "nest_neuron", "simulation_node", "nest_neuron")
        .connect("neuron_node", "nest_neuron_config", "simulation_node", "nest_neuron_config")
        .connect("stimulus_node", "input_current", "simulation_node", "input_current")
        
        # Connect simulation results to optimization node
        .connect("simulation_node", "simulation_results", "optimization_node", "simulation_results")
        
        # Connect parameter metadata to optimization node
        .connect("neuron_node", "parameter_metadata", "optimization_node", "parameter_metadata")
        .connect("stimulus_node", "parameter_metadata", "optimization_node", "parameter_metadata")
        
        # NO connections from optimization node back to setup nodes
        # (These would create cycles)
        
        .build()
    )
    
    # Print workflow information
    print(workflow)
    
    # Define optimization parameters
    objective_target = 10  # Target number of spikes
    max_iterations = 9     # Maximum number of iterations (3^2 grid points)
    
    # Reset optimization state
    optimization_node.reset_optimization()
    
    # Variable to bridge the gap between optimization and setup nodes
    # This breaks the cycle in the workflow
    optimization_parameters = {}
    
    # Simple optimization loop
    print("\nRunning encapsulated optimization workflow...")
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration}:")
        
        # Apply optimization parameters to setup nodes
        # This uses port access methods to get the data and configure to apply it
        if optimization_parameters:
            # Extract parameters for each node using the node names as keys
            neuron_params = optimization_parameters.get(neuron_node.name, {})
            stimulus_params = optimization_parameters.get(stimulus_node.name, {})
            
            # Configure nodes with new parameters
            if neuron_params:
                neuron_node.configure(**neuron_params)
            if stimulus_params:
                stimulus_node.configure(**stimulus_params)
                
            # Regenerate the stimulus with the new parameters
            # This is necessary because the stimulus generator needs to create
            # a new stimulus based on the updated parameters
            stimulus_node.generate_stimulus(
                simulation_time=simulation_time,
                dt=dt
            )
        
        # Execute the workflow (which is now acyclic)
        success = workflow.execute()
        if not success:
            print(f"Workflow execution failed at iteration {iteration}")
            break
        
        # Get optimization parameters for next iteration using port access methods
        # This completes the feedback loop that was removed from the workflow
        # We're using get_output_port to get the port object, then accessing its value
        params_port = optimization_node.get_output_port('parameters')
        optimization_parameters = params_port.value
        
        # Alternatively, we could use the convenience method:
        # optimization_parameters = optimization_node.get_output('parameters')
        
        # Check if we should continue
        # This could be based on error, iteration count, or other criteria
        # Get the error using the port access method
        error_port = optimization_node.get_output_port('error')
        error = error_port.value
        print(f"  Error: {error}")
        
        if error == 0 or iteration == max_iterations - 1:
            print("Optimization complete or maximum iterations reached")
            break
    
    # Get final optimization results
    results = optimization_node.get_optimization_results()
    
    # Print best parameters
    print("\nOptimization complete!")
    print("Best parameters:")
    for name, value in results['best_parameters'].items():
        print(f"  {name}: {value}")
    print(f"Best error: {results['best_error']}")
    
    # Plot optimization history
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot optimization history
        history = results['history']
        iterations = [h['iteration'] for h in history]
        errors = [h['error'] for h in history]
        
        plt.plot(iterations, errors, 'o-')
        plt.title('Encapsulated Optimization Progress')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('encapsulated_optimization_results.png')
        print("\nPlot saved as 'encapsulated_optimization_results.png'")
    except Exception as e:
        print(f"Error creating plot: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())