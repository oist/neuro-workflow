"""
Node for select an integration scheme. While TVB supports a number of schemes, for most purposes you should use either HeunDeterministic or HeunStochastic in the virtual brain (TVB). This node adds noise as well to the Epileptor model equations x2 and xrs
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


class TVBIntegratorNode(Node):
    """Node for defining the integration scheme. While TVB supports a number of schemes, for most purposes you should use either HeunDeterministic or HeunStochastic."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='network_builder',
        description='Definition of the integration scheme',
        
        parameters={
            'integrator': ParameterDefinition(
                default_value='HeunStochastic', # HeunDeterministic or HeunStochastic
                description='While TVB supports a number of schemes, for most purposes you should use either HeunDeterministic or HeunStochastic',
                constraints={},
                optimizable=False,
                optimization_range=[]
            ),
            'dt': ParameterDefinition(
                default_value=0.1, 
                description='integration steps [ms] (step size that is small enough for the integration to be numerically stable)',
                constraints={},
            ),
            'nsigma': ParameterDefinition(
                default_value=[0.00025, 0.001], 
                description='list of additive white Gaussian noise (standard deviation of the noise).',
                constraints={},
            ),
            
        },
        
        inputs={
        },
        
        outputs={
            'tvb_integrator': PortDefinition(
                type=PortType.OBJECT,
                description='Configured integration scheme in TVB'
            ),
        },
        methods={
            'int_initialization': MethodDefinition(
                description='Initialize the integration scheme',
                inputs=[],
                outputs=['tvb_integrator']
            ),
        }
    )
    def __init__(self, name: str):
        """Initialize the TVBIntegratorNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "int_initialization",
            self.int_initialization,
            method_key="int_initialization"
        )
        
    def int_initialization(self) -> Dict[str, Any]:
        """Node for select an integration scheme and add noises to the model. Noise is introduced in the state-variables x2x2 and y2y2 with mean 0 and variance 0.00025, and in the state-variable xrsxrs with mean 0 and variance 0.001. Other variables experience no noise due to their high sensitivity.
        Returns:
            integration scheme in TVB
        """
        # Initialise an Integrator scheme.
        dt = self._parameters['dt']   #integration steps [ms]
        nsigma1 = self._parameters['nsigma'][0]   #standard deviation of the noise
        nsigma2 = self._parameters['nsigma'][1] 
        hiss = noise.Additive(nsig=np.array([0., 0., 0., nsigma1, nsigma1, 0.,nsigma2, 0.])) # 
        
        if self._parameters['integrator']=='HeunStochastic':
            heunint = integrators.HeunStochastic(dt=dt, noise=hiss)
            print('passed HeunStochastic')
        if self._parameters['integrator']=='HeunDeterministic':
            heunint = integrators.HeunDeterministic(dt=dt, noise=hiss)
            print('passed HeunDeterministic')
            
        return {
            'tvb_integrator': heunint,
        }