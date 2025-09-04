#!/usr/bin/env python3
"""
Simple simulation example using the NeuroWorkflow library.

This example demonstrates how to create a basic workflow for neural simulation
using the NeuroWorkflow library.
"""

import sys
import os

# Add the src directory to the Python path to import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from neuroworkflow import WorkflowBuilder
from neuroworkflow.nodes.network import BuildSonataNetworkNode
from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode


def main():
    """Run a simple neural simulation workflow."""
    # Create nodes
    build_network = BuildSonataNetworkNode("SonataNetworkBuilder")
    build_network.configure(
        sonata_path="../data/300_pointneurons",
        net_config_file="circuit_config.json",
        sim_config_file="simulation_config.json",
        hdf5_hyperslab_size=1024,
    )

    simulate_network = SimulateSonataNetworkNode("SonataNetworkSimulation")
    simulate_network.configure(
        simulation_time=1000.0, record_from_population="internal", record_n_neurons=40
    )

    # Create workflow
    workflow = (
        WorkflowBuilder("neural_simulation")
        .add_node(build_network)
        .add_node(simulate_network)
        .connect(
            "SonataNetworkBuilder",
            "sonata_net",
            "SonataNetworkSimulation",
            "sonata_net",
        )
        .connect(
            "SonataNetworkBuilder",
            "node_collections",
            "SonataNetworkSimulation",
            "node_collections",
        )
        .build()
    )

    # Print workflow information
    print(workflow)

    # Execute workflow
    print("\nExecuting workflow...")
    success = workflow.execute()

    if success:
        print("Workflow execution completed successfully!")
    else:
        print("Workflow execution failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
