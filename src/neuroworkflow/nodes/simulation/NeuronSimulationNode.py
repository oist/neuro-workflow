"""
Neuron simulation node for parameter optimization example.

This module provides a node for simulating a neuron model.
"""

from typing import Dict, Any, List
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import nest


class NeuronSimulationNode(Node):
    """Node for simulating a neuron model."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='neuron_simulation',
        description='Simulates a neuron in NEST with given parameters',
        
        parameters={
            'dt': ParameterDefinition(
                default_value=0.1,
                description='Time step (ms), simulation resolution',
                constraints={'min': 0.1, 'max': 1.0},
                
            ),
            'simulation_time': ParameterDefinition(
                default_value=1000.0,
                description='Simulation time in milliseconds',
                constraints={'min': 1.0}
            ),
        },
        
        inputs={
            'nest_neuron': PortDefinition(
                type=PortType.OBJECT,
                description='Neuron object in NEST to simulate'
            ),
            'input_current': PortDefinition(
                type=PortType.OBJECT,
                description='Step input current in NEST over time (pA) from stimulus generator'
            ),
            'nest_neuron_config': PortDefinition(
                type=PortType.DICT,
                description='Neuron configuration parameters'
            )
            
        },
        
        outputs={
            'membrane_potential': PortDefinition(
                type=PortType.LIST,
                description='Membrane potential over time (mV)'
            ),
            'spike_times': PortDefinition(
                type=PortType.LIST,
                description='Times of action potentials (ms)'
            ),
            'simulation_results': PortDefinition(
                type=PortType.DICT,
                description='Complete simulation results'
            )
        },
        
        methods={
            'simulate': MethodDefinition(
                description='Run the neuron simulation',
                inputs=['nest_neuron', 'input_current','nest_neuron_config'],
                outputs=['membrane_potential', 'spike_times', 'simulation_results']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the NeuronSimulationNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "simulate",
            self.simulate,
            method_key="simulate"
        )
    
    def simulate(self, nest_neuron: Dict[str, Any], input_current: Dict[str, Any], nest_neuron_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the neuron simulation.
        
        Args:
            nest_neuron: Neuron model object to simulate
            input_current: Input current over time (nA)
            nest_neuron_config: 'Neuron configuration parameters'
            
        Returns:
            Dictionary with simulation results
        """
        # Get simulation parameters
        nest.resolution = self._parameters['dt']

        # Get neuron parameters for reporting        
        print(f"Running neuron simulation with parameters:")
        for key, value in nest_neuron_config.items():
            print(f"  {key}: {value}")
        print(f"  Time step: {dt} ms")
        
        # Apply stimuli
        nest.Connect(input_current, nest_neuron)

        #create devices for recordings
        #multimeter
        mul = nest.Create("multimeter",params={"interval": self._parameters['dt'], "record_from": ["V_m"]})
        #spikes recorder
        spr = nest.Create("spike_recorder")
        
        # Simulate using the neuron model object
        nest.Simulate(self._parameters['simulation_time'])
        
        # Extract simulation data
        v = mul.events["V_m"]
        spike_times = spr.events['times']

        # Create complete results dictionary
        simulation_results = {
            'membrane_potential': v.tolist(),
            'spike_times': spike_times,
            'spike_count': len(spike_times),
        }
        
        return {
            'membrane_potential': v.tolist(),
            'spike_times': spike_times,
            'simulation_results': simulation_results
        }