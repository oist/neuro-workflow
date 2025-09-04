#!/usr/bin/env python3
"""
Generated workflow for project: Test Project
"""
import sys
import os

# Add paths for JupyterLab environment
sys.path.append('../neuro/src')
sys.path.append('../upload_nodes')

from neuroworkflow import WorkflowBuilder
from upload_nodes.SampleSonataNetworkNode import SampleSonataNetworkNode
from neuroworkflow.nodes.network import BuildSonataNetworkNode

def main():
    """Run a simple neural simulation workflow."""

    calc_1756949691233_q2v2jsgh4 = SampleSonataNetworkNode("SampleSonataNetwork")

    calc_1756949877162_goi85ondm = BuildSonataNetworkNode("SonataNetworkBuilder")
    calc_1756949877162_goi85ondm.configure(
        sonata_path="../data/300_pointneurons",
        net_config_file="circuit_config.json",
        sim_config_file="simulation_config.json",
        hdf5_hyperslab_size=1024)


    workflow = (
        WorkflowBuilder("neural_simulation")
            .add_node(calc_1756949691233_q2v2jsgh4)
            .add_node(calc_1756949877162_goi85ondm)
            .connect("SonataNetworkBuilder", "calc_1756949877162_goi85ondm-sonata_net-output-object", "Node_calc_1756949691233_q2v2jsgh4", "calc_1756949691233_q2v2jsgh4-sample_sonata_net-input-object")
            .connect("SonataNetworkBuilder", "calc_1756949877162_goi85ondm-node_collections-output-dict", "Node_calc_1756949691233_q2v2jsgh4", "calc_1756949691233_q2v2jsgh4-sample_node_collections-input-dict")
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
