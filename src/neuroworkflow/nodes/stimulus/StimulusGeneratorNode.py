"""
Stimulus generator node for neural simulations.

This module provides a node for generating various types of input stimuli
for neural simulations.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import nest


class StimulusGeneratorNode(Node):
    """Node for generating input stimuli for neural simulations."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='stimulus_generator',
        description='Generates input stimuli for neural simulations',
        
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
            )
        },
        
        outputs={
            'input_current': PortDefinition(
                type=PortType.OBJECT,
                description='Generated input current over time (nA)'
            ),
          #  'time_array': PortDefinition(
          #      type=PortType.LIST,
          #      description='Time points corresponding to the input current'
          #  ),
          #  'stimulus_info': PortDefinition(
          #      type=PortType.DICT,
          #      description='Information about the generated stimulus'
          #  )
        },
        
        methods={
            'generate_stimulus': MethodDefinition(
                description='Generate a stimulus based on parameters',
                inputs=['simulation_time', 'dt'],
                outputs=['input_current']#, 'time_array', 'stimulus_info']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the StimulusGeneratorNode.
        
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
    
    def generate_stimulus(self, simulation_time: Dict[str, Any], dt: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a stimulus based on parameters.
        
        Args:
            simulation_time: Total simulation time (ms)
            dt: Time step (ms)
            
        Returns:
            Dictionary with generated stimulus
        """
        # Get parameters
        stimulus_type = self._parameters['stimulus_type']
        amplitude = self._parameters['amplitude']
        start_time = self._parameters['start_time']
        end_time = self._parameters['end_time']
        frequency = self._parameters['frequency']
        noise_sigma = self._parameters['noise_sigma']
        
        # Create time array
        #t = np.arange(0, simulation_time, dt)
        
        # Initialize input current
        #input_current = np.zeros_like(t)
        
        # Generate stimulus based on type in nest
        if stimulus_type == 'step':
            # Step stimulus
            #mask = (t >= start_time) & (t <= end_time)
            #input_current[mask] = amplitude
            input_current = nest.Create("step_current_generator",params={
                "amplitude_values": [self._parameters['amplitude'],],
                "amplitude_times": [self._parameters['start_time'],],
                "start": self._parameters['start_time'],
                "stop": self._parameters['end_time'],
                },
                )
            
        elif stimulus_type == 'sine':
            # Sinusoidal stimulus
            mask = (t >= start_time) & (t <= end_time)
            # Convert frequency from Hz to rad/ms
            angular_freq = 2 * np.pi * frequency / 1000
            input_current[mask] = amplitude * np.sin(angular_freq * (t[mask] - start_time))
            
        elif stimulus_type == 'ramp':
            # Ramp stimulus
            mask = (t >= start_time) & (t <= end_time)
            duration = end_time - start_time
            if duration > 0:
                input_current[mask] = amplitude * (t[mask] - start_time) / duration
                
        elif stimulus_type == 'noise':
            # Noise stimulus
            mask = (t >= start_time) & (t <= end_time)
            input_current[mask] = amplitude + np.random.normal(0, noise_sigma, size=np.sum(mask))
        
        # Create stimulus info
        stimulus_info = {
            'type': stimulus_type,
            'amplitude': amplitude,
            'start_time': start_time,
            'end_time': end_time,
            'frequency': frequency,
            'noise_sigma': noise_sigma,
            'simulation_time': simulation_time,
            'dt': dt
        }
        
        print(f"Generated {stimulus_type} stimulus with amplitude {amplitude} nA")
        
        return {
            'input_current': input_current,
            'stimulus_info': stimulus_info
        }