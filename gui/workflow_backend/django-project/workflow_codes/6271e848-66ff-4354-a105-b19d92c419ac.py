#!/usr/bin/env python3
"""
Sonata Network Simulation (example from NEST)
"""
import sys
import os
# Add the src directory to the Python path to import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from neuroworkflow import WorkflowBuilder
from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode
from neuroworkflow.nodes.network import BuildSonataNetworkNode

def main():
    """Run a simple neural simulation workflow."""

    calc_1755795881690_xfvp3rea4 = BuildSonataNetworkNode("SonataNetworkBuilder")
    calc_1755795881690_xfvp3rea4.configure(
        sonata_path="../data/300_pointneurons",
        net_config_file="circuit_config.json",
        sim_config_file="simulation_config.json",
        hdf5_hyperslab_size=1024
    )

    calc_1755795886998_6z36vzifg = SimulateSonataNetworkNode("SonataNetworkSimulation")
    calc_1755795886998_6z36vzifg.configure(
        simulation_time=1000.0,
        record_from_population="internal",
        record_n_neurons=40
    )


    workflow = (
        WorkflowBuilder("neural_simulation")
            .add_node(calc_1755795881690_xfvp3rea4)
            .add_node(calc_1755795886998_6z36vzifg)
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
