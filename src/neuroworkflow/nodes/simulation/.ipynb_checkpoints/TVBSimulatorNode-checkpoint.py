"""
Node for set up and run the simulation. The simulator is created as an iterable object, so all we need to do is iterate for some length, which we provide in ms, and collect the output.
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


class TVBSimulatorNode(Node):
    """Node for defining and running the simulation."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='simulation_node',
        description='Definition and execution of the simulation',
        
        parameters={
            'simulation_length': ParameterDefinition(
                default_value='50000', 
                description='length for the iteration of the simulation object',
                constraints={},
                optimizable=False,
                optimization_range=[]
            ),
        },
        
        inputs={
            'tvb_connectivity': PortDefinition(
                type=PortType.OBJECT,
                description='Configured connectivity matrix object in TVB'
            ),
            'tvb_model': PortDefinition(
                type=PortType.OBJECT,
                description='The local neural (hybrid version of Epileptor Model) dynamics of each brain area'
            ),
            'tvb_coupling': PortDefinition(
                type=PortType.OBJECT,
                description='It is a function that is used to join the local Model dynamics at distinct spatial locations over the connections described in tvb_connectivity. '
            ),
            'tvb_integrator': PortDefinition(
                type=PortType.OBJECT,
                description='Configured integration scheme in TVB'
            ),
            'tvb_monitor': PortDefinition(
                type=PortType.OBJECT,
                description='Configured monitor in TVB'
            ),
        },
        
        outputs={
            'tvb_simdata': PortDefinition(
                type=PortType.OBJECT,
                description='simulation data series'
            ),
            'tvb_simtime': PortDefinition(
                type=PortType.OBJECT,
                description='simulation time series'
            ),
        },
        
        methods={
            'sim_config_and_run': MethodDefinition(
                description='Initialize the simulation object and run the simulation.',
                inputs=['tvb_connectivity','tvb_model','tvb_coupling','tvb_integrator','tvb_monitor'],
                outputs=['tvb_simdata', 'tvb_simtime']
            ),
        }
    )
    def __init__(self, name: str):
        """Initialize the TVBSimulatorNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "sim_config_and_run",
            self.sim_config_and_run,
            method_key="sim_config_and_run"
        )
        
    def sim_config_and_run(self,tvb_connectivity: Dict[str, Any],tvb_model: Dict[str, Any],
                          tvb_coupling: Dict[str, Any],tvb_integrator: Dict[str, Any],
                          tvb_monitor: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the simulation object and run the simulation.
        Returns:
            Flag whether simulation was completed successfully
        """
        # Initialise the Simulator.
        sim = simulator.Simulator(model=tvb_model,
                          connectivity=tvb_connectivity,
                          conduction_speed=np.float64(tvb_connectivity.speed),
                          coupling=tvb_coupling,
                          integrator=tvb_integrator,
                          monitors=[tvb_monitor])
        sim.configure()

        # Perform simulation.
        tic = tm.time()

        tavg_time, tavg_data = [], []
        for tavg in sim(simulation_length=self._parameters['simulation_length']):
            if not tavg is None:
                tavg_time.append(tavg[0][0])
                tavg_data.append(tavg[0][1])
    
        print('simulation required %0.3f seconds.' % (tm.time()-tic))

        return {
            'tvb_simdata': tavg_data,
            'tvb_simtime': tavg_time,
        }