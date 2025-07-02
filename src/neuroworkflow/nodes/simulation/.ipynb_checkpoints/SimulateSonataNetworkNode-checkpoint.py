"""
NEST simulation nodes.

This module provides nodes for simulating neural networks using NEST.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import random
import numpy as np
from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import nest


class SimulateSonataNetworkNode(Node):
    """Simulation of a NEST network built from SONATA."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='simulation_node',
        description='Simulates a NEST network built from SONATA',
        parameters={
            'simulation_time': ParameterDefinition(
                default_value=1000.0,
                description='Simulation time in milliseconds',
                constraints={'min': 0.0}
            ),
            'record_from_population':ParameterDefinition(
                default_value='internal',
                description='Any population of interest'
            ),
            'record_n_neurons':ParameterDefinition(
                default_value='100',
                description='Number of neurons of record_from_population from where spikes are recorded',
                constraints={'min':1}
            )
        },
        inputs={
            'sonata_net': PortDefinition(
                type=PortType.OBJECT,
                description='SONATA network object'
            ),
            'node_collections': PortDefinition(
                type=PortType.DICT,
                description='NEST node collections'
            )
        },
        outputs={
            'simulation_completed': PortDefinition(
                type=PortType.BOOL,
                description='Whether simulation completed successfully'
            ),
            'spike_recorder': PortDefinition(
                type=PortType.OBJECT,
                description='NEST spike recorder device'
            )
        },
        methods={
            'initialize_recordables': MethodDefinition(
                description='Setup spike recorders',
                inputs=['node_collections'],
                outputs=['spike_recorder']
            ),
            'simulate_sonata': MethodDefinition(
                description='Run simulation',
                inputs=['sonata_net', 'node_collections', 'spike_recorder'],
                outputs=['simulation_completed']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize a SimulateSonataNetworkNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    
    def _define_process_steps(self) -> None:
        """Define process steps for this node."""
        self.add_process_step(
            "initialize_recordables",
            self.initialize_recordables,
            method_key="initialize_recordables"
        )
        
        self.add_process_step(
            "simulate_sonata",
            self.simulate_sonata,
            method_key="simulate_sonata"
        )
    
        
    def initialize_recordables(self, node_collections: Dict[str, Any]) -> Dict[str, Any]:
        """Setup spike recorders."""
        if node_collections is None:
            raise ValueError("Node collections not set")
        
        print("Creating spike recorder")
        
        spike_recorder = nest.Create("spike_recorder")
        pop_name = self._parameters["record_from_population"]
        record_node_ids = node_collections[pop_name][:self._parameters["record_n_neurons"]] #[1, 80, 160, 240, 270]
        nest.Connect(record_node_ids, spike_recorder)

        #nest.Connect(node_collections[pop_name][record_node_ids], spike_recorder)

        return {"spike_recorder": spike_recorder}
        
    def simulate_sonata(self, sonata_net: Dict[str, Any], node_collections: Dict[str, Any], 
                        spike_recorder: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate SONATA network."""
        # Validate inputs
        if sonata_net is None:
            raise ValueError("SONATA network not set")
        if node_collections is None:
            raise ValueError("Node collections not set")
        if spike_recorder is None:
            raise ValueError("Spike recorder not set")
            
        # Use simulation time from parameters
        simulation_time = self._parameters["simulation_time"]
        print(f"SONATA network simulation for {simulation_time} ms")
        
        
        # Get the start time for measuring simulation duration
        start_time = time.time()
        
        # Run the simulation
        print(f"Running NEST simulation for {simulation_time} ms...")
        nest.Simulate(simulation_time)

        # Calculate simulation duration
        simulation_duration = time.time() - start_time
        print(f"Simulation completed in {simulation_duration:.2f} seconds")
        
        import matplotlib.pyplot as plt
        nest.raster_plot.from_device(spike_recorder)
        plt.show()
          
        return {"simulation_completed": True}
    
