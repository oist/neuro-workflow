"""
Neuron setup node for parameter optimization example.

This module provides a node for creating and configuring neuron models.
"""

from typing import Dict, Any, Optional
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import nest


class NESTNeuronSetupNode(Node):
    """Node for creating and configuring neuron models.
    This class represents a neuron model with configurable parameters.
    In a real implementation, this could be a more complex model or
    interface with external neural simulators."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='neuron_setup',
        description='Creates and configures a neuron model in NEST',
        
        parameters={
            'nest_model': ParameterDefinition(
                default_value='iaf_psc_alpha',
                description='neuron model name in NEST'
            ),
            'threshold': ParameterDefinition(
                default_value=-55.0,
                description='Firing threshold (mV)',
                constraints={'min': -70.0, 'max': -40.0},
                optimizable=True,
                optimization_range=[-65.0, -45.0]
            ),
            'resting_potential': ParameterDefinition(
                default_value=-70.0,
                description='Resting membrane potential (mV)',
                constraints={'min': -80.0, 'max': -60.0},
            ),
            'time_constant': ParameterDefinition(
                default_value=20.0,
                description='Membrane time constant (ms)',
                constraints={'min': 5.0, 'max': 50.0},
                optimizable=True,
                optimization_range=[10.0, 30.0]
            ),
            'refractory_period': ParameterDefinition(
                default_value=2.0,
                description='Refractory period (ms)',
                constraints={'min': 1.0, 'max': 5.0}
            )
        },
        
        inputs={
            'parameters': PortDefinition(
                type=PortType.DICT,
                description='Parameters to configure the neuron',
                optional=True
            )
        },
        
        outputs={
            'nest_neuron': PortDefinition(
                type=PortType.OBJECT,
                description='Configured neuron model object in NEST'
            ),
            'nest_neuron_config': PortDefinition(
                type=PortType.DICT,
                description='Neuron configuration parameters'
            )
        },
        
        methods={
            'create_neuron': MethodDefinition(
                description='Create and configure a neuron in NEST',
                inputs=['parameters'],
                outputs=['nest_neuron', 'nest_neuron_config']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the NeuronSetupNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "create_neuron",
            self.create_neuron,
            method_key="create_neuron"
        )
    
    def create_neuron(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create and configure a neuron model.
        
        Args:
            parameters: Optional parameters to override defaults
            
        Returns:
            Dictionary with neuron model and configuration
        """
        # If parameters are provided, configure the node
        if parameters:
            self.configure(**parameters)
        
        # Create nest neuron object
        neuro_params = {
            "V_th": self._parameters['threshold'],
            "E_L": self._parameters['resting_potential'],
            "tau_m": self._parameters['time_constant'],
            "t_ref": self._parameters['refractory_period'],
        }

        nest.set_verbosity("M_ERROR")
        nest.ResetKernel()
        neuron = nest.Create(self._parameters['nest_model'], params=neuro_params)

        # Create configuration dictionary
        config = {
            'threshold': self._parameters['threshold'],
            'resting_potential': self._parameters['resting_potential'],
            'time_constant': self._parameters['time_constant'],
            'refractory_period': self._parameters['refractory_period'],
        }
        
        print(f"Created neuron model with parameters:")
        for key, value in config.items():
            print(f"  {key}: {value}")
            
        return {
            'nest_neuron': neuron,
            'nest_neuron_config': config
        }