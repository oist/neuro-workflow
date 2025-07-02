"""
Node for loading and visualizing the structural connectivity matrix that represents 
the set of all existing anatomical connections between brain areas in the virtual brain (TVB)
"""

from typing import Dict, Any, Optional
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType

#%matplotlib inline
# Import a bunch of stuff for TVB
from tvb.simulator.lab import *
from tvb.simulator.models.epileptor_rs import EpileptorRestingState
from tvb.datatypes.time_series import TimeSeriesRegion
import time as tm
import matplotlib.pyplot as plt 
import sys


class TVBConnectivitySetUpNode(Node):
    """Node for loading and visualizing the structural connectivity matrix that represents the set of all existing anatomical connections between brain areas"""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='network_builder',
        description='Initialise the Connectivity. TVB structural connectome is read from a zip file.',
        
        parameters={
            'connectivity_file': ParameterDefinition(
                default_value='sc.zip',
                description='structural connectivity matrix, from MRI',
                constraints={},
                optimizable=False,
                optimization_range=[]
            ),
        },
        
        inputs={
        },
        
        outputs={
            'tvb_connectivity': PortDefinition(
                type=PortType.OBJECT,
                description='Configured connectivity matrix object in TVB'
            ),
        },
        methods={
            'sc_initialization': MethodDefinition(
                description='Initialize the connectivity',
                inputs=[],
                outputs=['tvb_connectivity']
            ),
            'sc_visualization': MethodDefinition(
                description='Visualization of connectivity matrix',
                inputs=['tvb_connectivity'],
                outputs=['visualization completed']
            )
        }
    )
    def __init__(self, name: str):
        """Initialize the TVBConnectivitySetUpNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "sc_initialization",
            self.sc_initialization,
            method_key="sc_initialization"
        )
        self.add_process_step(
            "sc_visualization",
            self.sc_visualization,
            method_key="sc_visualization"
        )
        
    def sc_initialization(self) -> Dict[str, Any]:
        """read a zip file and setup connectivity matrix as TVB object
            
        Returns:
            connectivity matrix object in TVB format
        """
        con = connectivity.Connectivity.from_file(self._parameters['connectivity_file'])      
        nregions = len(con.region_labels)                               #number of regions
        con.weights = con.weights - con.weights * np.eye((nregions))    #remove self-connection
        con.speed = np.array([sys.float_info.max])                      #set conduction speed (here we neglect it)
        con.configure()
        return {
            'tvb_connectivity': con,
        }

    
    def sc_visualization(self, tvb_connectivity: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize connectivity matrix object in TVB format.
        
        Args:
            tvb_connectivity: connectivity matrix object in TVB format.
            
        Returns:
            a flag indicating the visualization is completed
        """
        
        # Visualization.
        plt.figure(figsize=(12,12))
        plt.imshow(tvb_connectivity.weights, interpolation='nearest', aspect='equal', cmap='jet')
        plt.title('TVB Structural Connectome', fontsize=20)
        plt.xticks(range(0, len(tvb_connectivity.region_labels)), tvb_connectivity.region_labels, fontsize=10, rotation=90)
        plt.yticks(range(0, len(tvb_connectivity.region_labels)), tvb_connectivity.region_labels, fontsize=10)
        cb=plt.colorbar(shrink=0.8)
        cb.set_label('weights', fontsize=14)
        plt.show()
        
        return {
            'visualization completed': True
        }