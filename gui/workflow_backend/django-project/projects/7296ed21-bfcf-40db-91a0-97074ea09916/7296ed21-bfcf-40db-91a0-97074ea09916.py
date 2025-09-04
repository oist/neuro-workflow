#!/usr/bin/env python3
"""
Example using the installed NeuroWorkflow library.

This example demonstrates how to use the NeuroWorkflow library after installation.
"""

from neuroworkflow import WorkflowBuilder
from neuroworkflow.nodes.network import BuildSonataNetworkNode
from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode


def main():
    """Run a simple neural simulation workflow."""
    # Create nodes
    build_network = BuildSonataNetworkNode("SonataNetworkBuilder")
    build_network.configure(
        sonata_path="/Users/carlosengutierrez/Downloads/neuro-workflow/data/300_pointneurons/",  # Path to our SONATA configuration
        net_config_file="circuit_config.json",
        sim_config_file="simulation_config.json",
        hdf5_hyperslab_size=1024,
    )

    # Print node information
    #print(build_network)

    simulate_network = SimulateSonataNetworkNode("SonataNetworkSimulation")
    simulate_network.configure(
        simulation_time=1000.0,
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
    import sys

    sys.exit(main())
