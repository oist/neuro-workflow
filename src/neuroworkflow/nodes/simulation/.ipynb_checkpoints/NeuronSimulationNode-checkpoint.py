"""
Simplified neuron simulation node for parameter optimization example.

This module provides a node for simulating a neuron model with a clean,
focused design that separates simulation from parameter tracking.
"""

from typing import Dict, Any, List
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import nest


class NeuronSimulationNode(Node):
    """Enhanced node for simulating a neuron model with a clean, focused design."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='enhanced_neuron_simulation',
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
            'time': PortDefinition(
                type=PortType.LIST,
                description='Time points for the simulation (ms)'
            ),
            'simulation_results': PortDefinition(
                type=PortType.DICT,
                description='Complete simulation results'
            )
        },
        
        methods={
            'simulate': MethodDefinition(
                description='Run the neuron simulation',
                inputs=['nest_neuron', 'input_current', 'nest_neuron_config'],
                outputs=['membrane_potential', 'spike_times', 'time', 'simulation_results']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the EnhancedNeuronSimulationNode.
        
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
    
    def simulate(self, nest_neuron: Any, input_current: Any, nest_neuron_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the neuron simulation.
        
        Args:
            nest_neuron: Neuron model object to simulate
            input_current: Input current over time (nA)
            nest_neuron_config: Neuron configuration parameters
            
        Returns:
            Dictionary with simulation results
        """
        # Get simulation parameters
        dt = self._parameters['dt']
        simulation_time = self._parameters['simulation_time']
        
        # Set NEST resolution
        #nest.resolution = dt check error <--

        # Print neuron parameters for reporting        
        print(f"Running neuron simulation with parameters:")
        for key, value in nest_neuron_config.items():
            print(f"  {key}: {value}")
        print(f"  Time step: {dt} ms")
        
        # Apply stimuli
        #nest.Connect(input_current, nest_neuron)

        # Create devices for recordings
        # Multimeter
        mul = nest.Create("multimeter", params={"interval": dt, "record_from": ["V_m"]})
        nest.Connect(mul, nest_neuron)
        
        # Spike recorder
        spr = nest.Create("spike_recorder")
        nest.Connect(nest_neuron, spr)
        
        # Simulate using the neuron model object
        nest.Simulate(simulation_time)
        
        # Extract simulation data
        events = mul.events
        times = events["times"].tolist()
        v_m = events["V_m"].tolist()
        spike_times = spr.events['times'].tolist()

        # Create simple simulation results without parameter tracking
        simulation_results = {
            'spike_times': spike_times,
            'membrane_potential': v_m,
            'time': times,
            'spike_count': len(spike_times)
        }
        
        return {
            'membrane_potential': v_m,
            'spike_times': spike_times,
            'time': times,
            'simulation_results': simulation_results
        }