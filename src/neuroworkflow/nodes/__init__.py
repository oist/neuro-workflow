"""
Pre-built node implementations for the NeuroWorkflow system.

This package provides a collection of ready-to-use nodes for various tasks.
"""

# Import node implementations for easy access
from neuroworkflow.nodes.network import BuildSonataNetworkNode, NESTNeuronSetupNode
from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode, NeuronSimulationNode
from neuroworkflow.nodes.analysis import SpikeAnalysisNode
from neuroworkflow.nodes.optimization import NeuronOptimizationNode
from neuroworkflow.nodes.stimulus import StimulusGeneratorNode


