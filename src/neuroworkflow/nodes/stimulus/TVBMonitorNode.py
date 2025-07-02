"""
Monitors. Although there are Monitors which apply a biophysical measurement process to the simulated neural activity, such as EEG, MEG, etc, There are a set of simple monitors.
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


class TVBMonitorNode(Node):
    """Node for defining the monitor scheme for the TVB simulation."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='stimulus_or_monitor', 
        description='Definition of the monitor',
        
        parameters={
            'monitor_type': ParameterDefinition(
                default_value='TemporalAverage', # there are many types.
                description='While TVB supports a number of monitor, a simple one is TemporalAverage monitor which averages over a time window of length period returning one time point every period ms.',
                constraints={},
            ),
            'temp_avg_period': ParameterDefinition(
                default_value=1, 
                description='this parameter is applied to TemporalAverage monitor. It defines the length period of the time window.',
                constraints={},
            ),   
        },
        
        inputs={
        },
        
        outputs={
            'tvb_monitor': PortDefinition(
                type=PortType.OBJECT,
                description='Configured monitor in TVB'
            ),
        },
        methods={
            'mon_initialization': MethodDefinition(
                description='Initialize the monitor',
                inputs=[],
                outputs=['tvb_monitor']
            ),
        }
    )
    def __init__(self, name: str):
        """Initialize the TVBMonitorNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "mon_initialization",
            self.mon_initialization,
            method_key="mon_initialization"
        )
        
    def mon_initialization(self) -> Dict[str, Any]:
        """Node for defining the monitor scheme for the TVB simulation.
        Returns:
            monitor configured in TVB
        """
        # Initialise some Monitors with period in physical time.
        if self._parameters['monitor_type']=='TemporalAverage':
            mon = monitors.TemporalAverage(period=self._parameters['temp_avg_period']) #in ms  

        # Bundle them
        what_to_watch = (mon)
            
        return {
            'tvb_monitor': what_to_watch,
        }