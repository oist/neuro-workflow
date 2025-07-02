"""
SONATA network building nodes.

This module provides nodes for loading and building networks from SONATA format.
"""

import os
import json
from typing import Dict, Any, List, Tuple
from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import nest 


class BuildSonataNetworkNode(Node):
    """SONATA Data loader and Network Building node."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='network_builder',
        description='Loads SONATA configuration and builds a network in NEST',
        parameters={
            'sonata_path': ParameterDefinition(
                default_value='',
                description='Path to SONATA configuration files'
            ),
            'net_config_file': ParameterDefinition(
                default_value='circuit_config.json',
                description='Network configuration file name'
            ),
            'sim_config_file': ParameterDefinition(
                default_value='simulation_config.json',
                description='Simulation configuration file name'
            ),
            'hdf5_hyperslab_size': ParameterDefinition(
                default_value=1,
                description='Size of HDF5 hyperslab for efficient loading',
                constraints={'min': 1}
            ),
        },
        outputs={
            'sonata_net': PortDefinition(
                type=PortType.OBJECT,
                description='SONATA network object'
            ),
            'node_collections': PortDefinition(
                type=PortType.DICT,
                description='NEST node collections'
            )
        },
        methods={
            'initialize_environment': MethodDefinition(
                description='Initialize NEST environment',
                inputs=[],
                outputs=['environment_initialized']
            ),
            'initialize_sonata': MethodDefinition(
                description='Initialize SONATA network',
                inputs=[],
                outputs=['sonata_net']
            ),
            'build_network': MethodDefinition(
                description='Build network in NEST',
                inputs=['sonata_net'],
                outputs=['node_collections']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize a BuildSonataNetworkNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
            
        
    def _define_process_steps(self) -> None:
        """Define process steps for this node."""
        # With the new schema, we can use method_key to link directly to NODE_DEFINITION methods
        self.add_process_step(
            "initialize_environment",
            self.initialize_environment,
            method_key="initialize_environment"
        )
        
        self.add_process_step(
            "initialize_sonata",
            self.initialize_sonata,
            method_key="initialize_sonata"
        )
        
        self.add_process_step(
            "build_network",
            self.build_network,
            method_key="build_network"
        )
        
    def initialize_environment(self) -> Dict[str, Any]:
        """Initialize NEST environment."""
        print("Initializing NEST environment")

        nest.set_verbosity("M_ERROR")
        nest.ResetKernel()
        return {"environment_initialized": True}
        
    def initialize_sonata(self) -> Dict[str, Any]:
        """Initialize SONATA network.

           Load JSON configuration files.
            
        Returns:
            Dict containing the sonata network object
        """
    
        sonata_path = self._parameters["sonata_path"]
        print(f"Initializing SONATA network from {sonata_path}")

        net_config_file = self._parameters["net_config_file"]
        sim_config_file = self._parameters["sim_config_file"]
        
        # Get absolute paths to config files
        net_config_path = os.path.join(sonata_path, net_config_file)
        sim_config_path = os.path.join(sonata_path, sim_config_file)

        # Verify files exist
        if not os.path.exists(net_config_path):
            raise FileNotFoundError(f"Network config file not found: {net_config_path}")
    
        if not os.path.exists(sim_config_path):
            raise FileNotFoundError(f"Simulation config file not found: {sim_config_path}")
   
        # Create a SONATA network object
        sonata_net = nest.SonataNetwork(net_config_path, sim_config=sim_config_path)

        return {"sonata_net": sonata_net}
        
    def build_network(self, sonata_net: Dict[str, Any]) -> Dict[str, Any]:
        """Build network in NEST."""
            
        hdf5_hyperslab_size = self._parameters["hdf5_hyperslab_size"]
        print(f"Building network using hdf5_hyperslab_size = {hdf5_hyperslab_size}")

        node_collections = sonata_net.BuildNetwork(hdf5_hyperslab_size=2**20)
        
        return {"node_collections": node_collections}