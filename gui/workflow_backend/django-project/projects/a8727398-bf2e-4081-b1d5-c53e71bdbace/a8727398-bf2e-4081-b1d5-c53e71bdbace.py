#!/usr/bin/env python3
"""
Generated workflow for project: test
"""
import sys
import os
# Add the src directory to the Python path to import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from neuroworkflow import WorkflowBuilder
from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode

def main():
    """Run a simple neural simulation workflow."""

    calc_1756742677918_5nx4yval2 = SimulateSonataNetworkNode("SonataNetworkSimulation")
    calc_1756742677918_5nx4yval2.configure(
        simulation_time=1000.0,
        record_from_population="internal",
        record_n_neurons=40
    )


    workflow = (
        WorkflowBuilder("neural_simulation")
            .add_node(calc_1756742677918_5nx4yval2)
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
