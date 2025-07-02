"""
Enhanced stimulus generator node for neural simulations.

This module provides a node for generating various types of input stimuli
for neural simulations, with added support for exposing optimization metadata.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import nest


class StimulusGeneratorNode(Node):
    """Enhanced node for generating input stimuli for neural simulations."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='enhanced_stimulus_generator',
        description='Generates input stimuli for neural simulations with optimization metadata',
        
        parameters={
            'stimulus_type': ParameterDefinition(
                default_value='step',
                description='Type of stimulus to generate (step, sine, ramp, noise)',
                constraints={'allowed_values': ['step', 'sine', 'ramp', 'noise']}
            ),
            'amplitude': ParameterDefinition(
                default_value=2.0,
                description='Amplitude of the stimulus (nA)',
                constraints={'min': 0.0, 'max': 1000.0},
                optimizable=True,
                optimization_range=[0.5, 5.0]
            ),
            'start_time': ParameterDefinition(
                default_value=50.0,
                description='Start time of the stimulus (ms)',
                constraints={'min': 0.0, 'max': 1000.0},
                optimizable=True,
                optimization_range=[10.0, 100.0]
            ),
            'end_time': ParameterDefinition(
                default_value=150.0,
                description='End time of the stimulus (ms)',
                constraints={'min': 0.0, 'max': 1000.0},
                optimizable=True,
                optimization_range=[100.0, 200.0]
            ),
            'frequency': ParameterDefinition(
                default_value=10.0,
                description='Frequency for oscillatory stimuli (Hz)',
                constraints={'min': 0.1, 'max': 100.0},
                optimizable=True,
                optimization_range=[1.0, 50.0]
            ),
            'noise_sigma': ParameterDefinition(
                default_value=0.5,
                description='Standard deviation for noise stimulus',
                constraints={'min': 0.0, 'max': 5.0},
                optimizable=True,
                optimization_range=[0.1, 2.0]
            )
        },
        
        inputs={
            'simulation_time': PortDefinition(
                type=PortType.FLOAT,
                description='Total simulation time (ms)'
            ),
            'dt': PortDefinition(
                type=PortType.FLOAT,
                description='Time step (ms)',
                optional=True
            ),
            'parameters': PortDefinition(
                type=PortType.DICT,
                description='Parameters to configure the stimulus',
                optional=True
            )
        },
        
        outputs={
            'input_current': PortDefinition(
                type=PortType.OBJECT,
                description='Generated input current over time (nA)'
            ),
            'parameter_metadata': PortDefinition(
                type=PortType.DICT,
                description='Metadata about optimizable parameters'
            )
        },
        
        methods={
            'generate_stimulus': MethodDefinition(
                description='Generate a stimulus based on parameters',
                inputs=['simulation_time', 'dt', 'parameters'],
                outputs=['input_current', 'parameter_metadata']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the EnhancedStimulusGeneratorNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "generate_stimulus",
            self.generate_stimulus,
            method_key="generate_stimulus"
        )
    
    def generate_stimulus(self, simulation_time: float, dt: float = 0.1, 
                         parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a stimulus based on parameters.
        
        Args:
            simulation_time: Total simulation time (ms)
            dt: Time step (ms)
            parameters: Optional parameters to override defaults. Can be a flat dictionary
                       or a structured dictionary with node names as keys.
            
        Returns:
            Dictionary with generated stimulus and optimization metadata
        """
        # If parameters are provided, configure the node
        if parameters:
            # Check if parameters is a structured dictionary with node names as keys
            if self.name in parameters:
                # Extract parameters for this node
                node_params = parameters[self.name]
                self.configure(**node_params)
            else:
                # Assume flat dictionary of parameters
                self.configure(**parameters)
            
        # Get parameters
        stimulus_type = self._parameters['stimulus_type']
        amplitude = self._parameters['amplitude']
        start_time = self._parameters['start_time']
        end_time = self._parameters['end_time']
        frequency = self._parameters['frequency']
        noise_sigma = self._parameters['noise_sigma']
        
        # Generate stimulus based on type in nest
        if stimulus_type == 'step':
            input_current = nest.Create("step_current_generator", params={
                "amplitude_values": [amplitude],
                "amplitude_times": [start_time],
                "start": start_time,
                "stop": end_time,
            })
        elif stimulus_type == 'sine':
            # Create sine wave stimulus
            input_current = nest.Create("ac_generator", params={
                "amplitude": amplitude,
                "frequency": frequency,
                "start": start_time,
                "stop": end_time
            })
        elif stimulus_type == 'noise':
            # Create noise stimulus
            input_current = nest.Create("noise_generator", params={
                "mean": amplitude,
                "std": noise_sigma,
                "start": start_time,
                "stop": end_time
            })
        else:
            # Default to step current
            input_current = nest.Create("step_current_generator", params={
                "amplitude_values": [amplitude],
                "amplitude_times": [start_time],
                "start": start_time,
                "stop": end_time,
            })
        
        # Get optimization metadata
        optimization_metadata = self.get_optimizable_parameters()
        
        print(f"Generated {stimulus_type} stimulus with amplitude {amplitude} nA")
        
        return {
            'input_current': input_current,
            'parameter_metadata': {self.name: optimization_metadata}
        }