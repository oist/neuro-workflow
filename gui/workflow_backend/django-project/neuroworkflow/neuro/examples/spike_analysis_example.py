#!/usr/bin/env python3
"""
Spike analysis example using the NeuroWorkflow library.

This example demonstrates how to use the SpikeAnalysisNode to analyze spike data.
"""

import sys
import os

# Add the src directory to the Python path to import the library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from neuroworkflow import WorkflowBuilder
from neuroworkflow.nodes.analysis import SpikeAnalysisNode


def main():
    """Run a spike analysis workflow."""
    # Create a spike analysis node
    analysis_node = SpikeAnalysisNode("SpikeAnalyzer")
    analysis_node.configure(
        time_window=[0.0, 2000.0],
        bin_size=5.0,
        metrics=['rate', 'isi']  # Valid values are 'rate', 'isi', 'cv', 'fano'
    )
    
    # Create a simple workflow with just this node
    workflow = WorkflowBuilder("spike_analysis_workflow").add_node(analysis_node).build()
    
    # Set input values directly (in a real workflow, these would come from connections)
    analysis_node.get_input_port("spike_data").set_value({
        "format": "spike_recorder",
        "data": {
            "senders": [1, 2, 3, 1, 2, 1],
            "times": [10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
        }
    })
    
    # Execute the workflow
    print("\nExecuting workflow...")
    success = workflow.execute()
    
    if success:
        # Get the results
        report = analysis_node.get_output_port("analysis_report").value
        firing_rates = analysis_node.get_output_port("firing_rates").value
        isi_histograms = analysis_node.get_output_port("isi_histograms").value
        
        print("\nAnalysis Report Summary:")
        print(f"Number of neurons: {report['summary']['n_neurons']}")
        print(f"Mean firing rate: {report['summary']['mean_firing_rate']:.2f} Hz")
        
        print("\nFiring Rates:")
        for neuron_id, rate in firing_rates.items():
            print(f"  Neuron {neuron_id}: {rate:.2f} Hz")
        
        print("\nWorkflow execution completed successfully!")
    else:
        print("Workflow execution failed!")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())