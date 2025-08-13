"""
Node for set the visualization of a TVB simulation. up and run the simulation. The simulator is created as an iterable object, so all we need to do is iterate for some length, which we provide in ms, and collect the output.
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


class TVBVisualizationNode(Node):
    """Node for visualizating TVB simulation results."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='simulation_node',
        description='Visualization of the simulation results',
        
        parameters={
            'plot_type': ParameterDefinition(
                default_value='LFP', 
                description='plot local field potentials',
                constraints={},
                optimizable=False,
                optimization_range=[]
            ),
        },
        
        inputs={
            'data_series': PortDefinition(
                type=PortType.OBJECT,
                description='simulation output data series'
            ),
            'time_series': PortDefinition(
                type=PortType.OBJECT,
                description='simulation output time series'
            ),
            'tvb_model': PortDefinition(
                type=PortType.OBJECT,
                description='The local neural (hybrid version of Epileptor Model) dynamics of each brain area'
            ),
            'tvb_connectivity': PortDefinition(
                type=PortType.OBJECT,
                description='Configured connectivity matrix object in TVB'
            ),
        },
        
        outputs={
            'visualization_completed': PortDefinition(
                type=PortType.BOOL,
                description='visualization confirmation'
            ),
        },
        
        methods={
            'plot_lfp': MethodDefinition(
                description='plottinmg lfp from data and time series',
                inputs=['data_series','time_series','tvb_model','tvb_connectivity'],
                outputs=[]
            ),
        }
    )
    def __init__(self, name: str):
        """Initialize the TVBVisualizationNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "plot_lfp",
            self.plot_lfp,
            method_key="plot_lfp"
        )
        
    def plot_lfp(self,data_series: Dict[str, Any],time_series: Dict[str, Any],
                 tvb_model: Dict[str, Any], tvb_connectivity: Dict[str, Any]) -> Dict[str, Any]:
        """Visualize the simulation results as LFP curves.
        Returns:
            Flag whether simulation was completed successfully
        """
        tavg_data = data_series
        #print('data_series.shape: ',data_series.shape)
        tavg_time = time_series
        mod = tvb_model
        con = tvb_connectivity
        nregions = len(con.region_labels)  
        
        # Normalize time series
        tavg_data /= (np.max(tavg_data, 0) - np.min(tavg_data, 0))
        tavg_data -= np.mean(tavg_data, 0)

        print('tavg_data.shape: ',tavg_data.shape)
        # Make lists numpy.arrays for easier use.  
        TAVG = np.squeeze(np.array(tavg_data))
        TAVG.shape
        print('TAVG.shape: ',TAVG.shape)

        # Compute LFP output model.
        DATA = mod.p[[0]] * TAVG[:, 0, :] + (1 - mod.p[[0]]) * TAVG[:, 2, :]

        EZs = [62, 47, 40]
        for ie, ez in enumerate(EZs):
            DATA[:, EZs[ie]] = mod.p[[EZs[ie]]] * TAVG[:, 0, EZs[ie]] + (1 - mod.p[[EZs[ie]]]) * TAVG[:, 2, EZs[ie]]
    
        PZs = [69, 72]
        for ip, pz in enumerate(PZs):
            DATA[:, PZs[ip]] = mod.p[[PZs[ip]]] * TAVG[:, 0, PZs[ip]] + (1 - mod.p[[PZs[ip]]]) * TAVG[:, 2, PZs[ip]]

        # Plot time series.
        fig1 = plt.figure(figsize=(15,15))
        plt.plot(DATA[:, :] + np.r_[:nregions], 'k', alpha=0.1)
        plt.yticks(np.arange(len(con.region_labels)), con.region_labels, fontsize=10)

        EZs = [62, 47, 40]
        for ie, ez in enumerate(EZs):
            plt.plot(DATA[:, EZs[ie]] + EZs[ie], 'C1')
        PZs = [69, 72]
        for ip, pz in enumerate(PZs):
            plt.plot(DATA[:, PZs[ip]] + PZs[ip], 'C0')
    
        #plt.axvline(x=15000, color='k', linestyle='--')
        #plt.axvline(x=40000, color='k', linestyle='--')
        data_length = DATA.shape[0]
        plt.axvline(x=int(data_length * 0.3), color='k', linestyle='--', alpha=0.7)
        plt.axvline(x=int(data_length * 0.8), color='k', linestyle='--', alpha=0.7)
        plt.xlim(0, data_length)

        plt.title('Resting-state with Interictal Spikes', fontsize=20)
        plt.xlabel('Time [ms]', fontsize=20)

        plt.show()

        # Zoom.
        EN = [62, 40, 47, 69, 72]
        idx = np.arange(len(EN))
        
        zoom_start = max(0, int(data_length * 0.3))  # Start at 30% of data
        zoom_end = min(data_length, int(data_length * 0.8))  # End at 80% of data
        time_range = np.arange(zoom_start, zoom_end)
        
        fig2 = plt.figure(figsize=(15,15))
        for ie, en in enumerate(EN):
            data_slice = DATA[zoom_start:zoom_end, EN[ie]]
            plt.plot(time_range, data_slice + idx[ie], 'k', alpha=0.5)
            #plt.plot(np.arange(15000, 40000), DATA[15000:40000, EN[ie]] + idx[ie], 'k', alpha=0.5)
        plt.title('Epileptogenic Network time series', fontsize=15)
        plt.xlabel('Time [ms]', fontsize=15)
        plt.yticks(np.arange(len(EN)), con.region_labels[EN])

        plt.show()


        return {"visualization_completed": True}
