"""
Spike train analysis node.

This module provides a node for analyzing spike trains from neural simulations.
"""

from typing import Dict, Any, List, Optional
import random
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType


class SpikeAnalysisNode(Node):
    """Node for analyzing spike trains from neural simulations."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='spike_analysis',
        description='Analyzes spike trains from neural simulations',
        
        parameters={
            'time_window': ParameterDefinition(
                default_value=[0.0, 1000.0],
                description='Time window for analysis in milliseconds',
                constraints={'min_length': 2, 'max_length': 2}
            ),
            'bin_size': ParameterDefinition(
                default_value=10.0,
                description='Bin size for spike histograms in milliseconds',
                constraints={'min': 0.1, 'max': 1000.0}
            ),
            'metrics': ParameterDefinition(
                default_value=['rate', 'isi'],
                description='Metrics to calculate (rate, isi, cv, fano)',
                # No constraints for list values - we'll validate in the method
            )
        },
        
        inputs={
            'spike_data': PortDefinition(
                type=PortType.OBJECT,
                description='Spike recorder data from simulation'
            ),
            'neuron_ids': PortDefinition(
                type=PortType.LIST,
                description='List of neuron IDs to analyze',
                optional=True
            )
        },
        
        outputs={
            'firing_rates': PortDefinition(
                type=PortType.DICT,
                description='Firing rates for each neuron'
            ),
            'isi_histograms': PortDefinition(
                type=PortType.DICT,
                description='Inter-spike interval histograms',
                optional=True
            ),
            'analysis_report': PortDefinition(
                type=PortType.DICT,
                description='Complete analysis report'
            )
        },
        
        methods={
            'extract_spikes': MethodDefinition(
                description='Extract spikes from recorder data',
                inputs=['spike_data', 'neuron_ids'],
                outputs=['extracted_spikes']
            ),
            'calculate_metrics': MethodDefinition(
                description='Calculate analysis metrics',
                inputs=['extracted_spikes'],
                outputs=['firing_rates', 'isi_histograms']
            ),
            'generate_report': MethodDefinition(
                description='Generate complete analysis report',
                inputs=['firing_rates', 'isi_histograms'],
                outputs=['analysis_report']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the SpikeAnalysisNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "extract_spikes",
            self.extract_spikes,
            method_key="extract_spikes"
        )
        
        self.add_process_step(
            "calculate_metrics",
            self.calculate_metrics,
            method_key="calculate_metrics"
        )
        
        self.add_process_step(
            "generate_report",
            self.generate_report,
            method_key="generate_report"
        )
    
    def extract_spikes(self, spike_data: Dict[str, Any], neuron_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Extract spikes from recorder data.
        
        Args:
            spike_data: Spike recorder data from simulation
            neuron_ids: Optional list of neuron IDs to analyze
            
        Returns:
            Dictionary with extracted spikes
        """
        print(f"Extracting spikes from recorder data")
        
        # Get time window from parameters
        time_window = self._parameters['time_window']
        start_time, end_time = time_window
        
        # If neuron_ids is not provided, use all neurons in the data
        if neuron_ids is None:
            # In a real implementation, you would extract this from spike_data
            neuron_ids = list(range(1, 11))  # Simulate 10 neurons
            
        print(f"Analyzing {len(neuron_ids)} neurons in time window [{start_time}, {end_time}] ms")
        
        # Simulate extracted spikes (neuron_id -> list of spike times)
        extracted_spikes = {}
        for neuron_id in neuron_ids:
            # Simulate Poisson process with random rate between 5-15 Hz
            rate = random.uniform(5, 15)  # Hz
            # Convert to expected number of spikes in the time window
            expected_spikes = rate * (end_time - start_time) / 1000.0
            # Generate random spike times
            n_spikes = int(random.gauss(expected_spikes, expected_spikes**0.5))
            spike_times = sorted([random.uniform(start_time, end_time) for _ in range(max(0, n_spikes))])
            extracted_spikes[neuron_id] = spike_times
            
        return {'extracted_spikes': extracted_spikes}
    
    def calculate_metrics(self, extracted_spikes: Dict[int, List[float]]) -> Dict[str, Any]:
        """Calculate analysis metrics.
        
        Args:
            extracted_spikes: Dictionary of extracted spikes (neuron_id -> spike times)
            
        Returns:
            Dictionary with calculated metrics
        """
        print(f"Calculating metrics: {self._parameters['metrics']}")
        
        # Get parameters
        time_window = self._parameters['time_window']
        start_time, end_time = time_window
        bin_size = self._parameters['bin_size']
        metrics = self._parameters['metrics']
        
        # Validate metrics
        valid_metrics = ['rate', 'isi', 'cv', 'fano']
        for metric in metrics:
            if metric not in valid_metrics:
                print(f"Warning: Unknown metric '{metric}'. Valid metrics are: {valid_metrics}")
                metrics = [m for m in metrics if m in valid_metrics]
                break
        
        # Calculate firing rates
        firing_rates = {}
        for neuron_id, spike_times in extracted_spikes.items():
            # Calculate rate as spikes per second
            duration_sec = (end_time - start_time) / 1000.0
            rate = len(spike_times) / duration_sec if duration_sec > 0 else 0
            firing_rates[neuron_id] = rate
            
        # Calculate ISI histograms if requested
        isi_histograms = {}
        if 'isi' in metrics:
            for neuron_id, spike_times in extracted_spikes.items():
                if len(spike_times) >= 2:
                    # Calculate inter-spike intervals
                    isis = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
                    
                    # Create histogram
                    max_isi = max(isis) if isis else bin_size
                    bins = np.arange(0, max_isi + bin_size, bin_size)
                    hist, edges = np.histogram(isis, bins=bins)
                    
                    isi_histograms[neuron_id] = {
                        'counts': hist.tolist(),
                        'bin_edges': edges.tolist(),
                        'mean_isi': np.mean(isis) if isis else 0,
                        'std_isi': np.std(isis) if isis else 0
                    }
                else:
                    isi_histograms[neuron_id] = {
                        'counts': [],
                        'bin_edges': [],
                        'mean_isi': 0,
                        'std_isi': 0
                    }
        
        return {
            'firing_rates': firing_rates,
            'isi_histograms': isi_histograms if 'isi' in metrics else {}
        }
    
    def generate_report(self, firing_rates: Dict[int, float], isi_histograms: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate complete analysis report.
        
        Args:
            firing_rates: Dictionary of firing rates
            isi_histograms: Dictionary of ISI histograms
            
        Returns:
            Dictionary with analysis report
        """
        print("Generating analysis report")
        
        # Create a comprehensive report
        report = {
            'parameters': {
                'time_window': self._parameters['time_window'],
                'bin_size': self._parameters['bin_size'],
                'metrics': self._parameters['metrics']
            },
            'summary': {
                'n_neurons': len(firing_rates),
                'mean_firing_rate': sum(firing_rates.values()) / len(firing_rates) if firing_rates else 0,
                'min_firing_rate': min(firing_rates.values()) if firing_rates else 0,
                'max_firing_rate': max(firing_rates.values()) if firing_rates else 0
            },
            'neuron_data': {
                neuron_id: {
                    'firing_rate': firing_rates.get(neuron_id, 0),
                    'isi_stats': isi_histograms.get(neuron_id, {})
                }
                for neuron_id in firing_rates.keys()
            }
        }
        
        return {'analysis_report': report}