"""
Node for defining the local neural (or Model) dynamics of each brain area as a set of differential equations. 
There are a number of predefined models available in TVB. For our purpose here we will use
in this node the hybrid version of Epileptor.
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


class TVBEpileptorNode(Node):
    """Node for defining the local neural (or Model) dynamics for the hybrid version of Epileptor in TVB"""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='network_builder', #or model_builder?
        description='defining the local neural (or Model) dynamics and parameters for the hybrid version of Epileptor in TVB',
        
        parameters={
            'Ks': ParameterDefinition(
                default_value=-1.0,
                description='global coupling of the 1st population (x1,y1),(Jirsa et al., 2014)',
                constraints={},
            ),
            'K_rs': ParameterDefinition(
                default_value=1.0,
                description='global coupling of the 3rd population (xrs,yrs),(Jirsa et al., 2014)',
                constraints={},
            ),
            'tau': ParameterDefinition(
                default_value=1000,
                description='is this tau_0 ? time scale of.',
                constraints={},
            ),
            'r': ParameterDefinition(
                default_value=0.000015,
                description='I need to check what is this parameter about',
                constraints={},
            ),
            'a_rs': ParameterDefinition(
                default_value=1.7402,
                description='I need to check what is this parameter about',
                constraints={},
            ),
            'NIZ': ParameterDefinition(
                default_value= {'regions_id':'all',
                                'x0':-2.3,
                                'bb':4,
                                'p':0.1
                                },
                description='heatmap of parameters for the regions corresponding to NIZ: Non Involved Zone. At the beginning, all regions are NIZ. Parameters: regions_id, x0: epileptogenicity value which quantifies each network node i ability to trigger a seizure, bb: to check, p: to check.',
                constraints={},
            ),
            'EZ': ParameterDefinition(
                default_value={'regions_id':[40,47,62],
                                'x0':[-1.4,-1.6,-1.6],
                                'bb':1,
                                'p':0.9
                               },
                description='heatmap of parameters for the regions corresponding to the EZ: epileptic zone. Parameters: regions_id, x0: epileptogenicity value which quantifies each network node i ability to trigger a seizure, bb: to check, p: to check.',
                constraints={},
            ),
            'PZ': ParameterDefinition(
                default_value={'regions_id':[69,72],
                                'x0':[-1.7,-1.8],
                                'bb':2,
                                'p':0.7
                               },
                description='heatmap of parameters for the regions corresponding to the PZ: propagation zone. Parameters: regions_id, x0: epileptogenicity value which quantifies each network node i ability to trigger a seizure, bb: to check, p: to check.',
                constraints={},
            ),
            'coupl': ParameterDefinition(
                default_value=1,
                description='Coupling function parameter',
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
        },
        outputs={
            'tvb_model': PortDefinition(
                type=PortType.OBJECT,
                description='The local neural (hybrid version of Epileptor Model) dynamics of each brain area'
            ),
            'tvb_coupling': PortDefinition(
                type=PortType.OBJECT,
                description='It is a function that is used to join the local Model dynamics at distinct spatial locations over the connections described in tvb_connectivity. '
            ),
        },
        methods={
            'model_initialization': MethodDefinition(
                description='Initialize the Epileptor Model based on Heatmap of epileptogenicity and related parameters',
                inputs=['tvb_connectivity'],
                outputs=['tvb_model']
            ),
            'coupling_initialization': MethodDefinition(
                description='Note that the global coupling parameter value for each submodel is already set in the initialisation of the model (see variables Ks and K_rs above), so here we set the value of coupl',
                inputs=[],
                outputs=['tvb_coupling']
            )
        }
    )
    def __init__(self, name: str):
        """Initialize the TVBConnectivitySetUpNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "model_initialization",
            self.model_initialization,
            method_key="model_initialization"
        )
        self.add_process_step(
            "coupling_initialization",
            self.coupling_initialization,
            method_key="coupling_initialization"
        )
        
    def model_initialization(self, tvb_connectivity: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the Epileptor Model based on Heatmap of epileptogenicity and related parameters
            
        Returns:
            Epileptor model configured in TVB format
        """
        # Initialise the Model.
        mod = EpileptorRestingState(Ks=np.array([self._parameters['Ks']]),
                      r=np.array([self._parameters['r']]),
                      tau=np.array([self._parameters['tau']]),
                      K_rs=np.array([self._parameters['K_rs']]))
        nregions = len(tvb_connectivity.region_labels)
        mod.a_rs = np.ones((nregions)) * (self._parameters['a_rs']) 

        # Heatmap of epileptogenicity and related parameters.
        
        #NIZ
        mod.x0 = np.ones((nregions)) * (self._parameters['NIZ']['x0'])          #NIZ
        mod.bb = np.ones((nregions)) * (self._parameters['NIZ']['bb'])             #NIZ
        mod.p = np.ones((nregions)) * (self._parameters['NIZ']['p'])            #NIZ
        
        #EZ
        if len(self._parameters['EZ']['regions_id'])==len(self._parameters['EZ']['x0']):
            for i in range(len(self._parameters['EZ']['regions_id'])):
                mod.x0[[self._parameters['EZ']['regions_id'][i]]] = np.ones((1)) * (self._parameters['EZ']['x0'][i]) 
        else:
            print('error EZ regions and x0 parameters have different lengths')
        
        mod.bb[self._parameters['EZ']['regions_id']] = np.ones((len(self._parameters['EZ']['regions_id']))) * (self._parameters['EZ']['bb'])
        mod.p[self._parameters['EZ']['regions_id']] = np.ones((len(self._parameters['EZ']['regions_id']))) * (self._parameters['EZ']['p'])

        #PZ
        if len(self._parameters['PZ']['regions_id'])==len(self._parameters['PZ']['x0']):
            for i in range(len(self._parameters['PZ']['regions_id'])):
                mod.x0[[self._parameters['PZ']['regions_id'][i]]] = np.ones((1)) * (self._parameters['PZ']['x0'][i]) 
        else:
            print('error PZ regions and x0 parameters have different lengths')
        
        mod.bb[self._parameters['PZ']['regions_id']] = np.ones((len(self._parameters['PZ']['regions_id']))) * (self._parameters['PZ']['bb'])
        mod.p[self._parameters['PZ']['regions_id']] = np.ones((len(self._parameters['PZ']['regions_id']))) * (self._parameters['PZ']['p'])
        
        return {
            'tvb_model': mod,
        }

    def coupling_initialization(self) -> Dict[str, Any]:
        """It is a function that is used to join the local Model dynamics at distinct spatial locations over the connections described in Connectivity. we will use the Difference coupling class connected through TVB's default connectivity matrix.
        Returns:
            a coupling function tvb_coupling
        """
        # Initialise a Coupling function.
        coupl = self._parameters['coupl']
        con_coupling = coupling.Difference(a=np.array([coupl]))
        
        return {
            'tvb_coupling': con_coupling
        }