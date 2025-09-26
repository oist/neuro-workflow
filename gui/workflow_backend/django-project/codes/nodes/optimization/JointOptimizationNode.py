"""
Joint optimization node for optimizing parameters from multiple nodes.

This module provides a node for evaluating simulation results from multiple nodes
and optimizing their parameters jointly.
"""

from typing import Dict, Any, List, Callable, Optional
import numpy as np
from itertools import product

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType


class JointOptimizationNode(Node):
    """Node for jointly optimizing parameters from multiple nodes."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='joint_optimization',
        description='Jointly optimizes parameters from multiple nodes',
        
        parameters={
            'optimization_method': ParameterDefinition(
                default_value='grid',
                description='Optimization method (grid, random)',
                constraints={'allowed_values': ['grid', 'random']}
            ),
            'grid_points': ParameterDefinition(
                default_value=4,
                description='Number of points per dimension for grid search',
                constraints={'min': 2, 'max': 10}
            ),
            'max_iterations': ParameterDefinition(
                default_value=100,
                description='Maximum number of iterations',
                constraints={'min': 1}
            )
        },
        
        inputs={
            'simulation_results': PortDefinition(
                type=PortType.DICT,
                description='Simulation results from multiple nodes'
            ),
            'parameter_metadata': PortDefinition(
                type=PortType.DICT,
                description='Optimization metadata for parameters from multiple nodes',
                optional=True
            ),
            'objective_target': PortDefinition(
                type=PortType.FLOAT,
                description='Target value for the objective function'
            ),
            'iteration': PortDefinition(
                type=PortType.INT,
                description='Current iteration number',
                optional=True
            )
        },
        
        outputs={
            'error': PortDefinition(
                type=PortType.FLOAT,
                description='Calculated error value'
            ),
            'evaluation_result': PortDefinition(
                type=PortType.DICT,
                description='Complete evaluation result'
            ),
            'parameters': PortDefinition(
                type=PortType.DICT,
                description='Parameters for the next iteration (for all nodes)'
            )
        },
        
        methods={
            'evaluate': MethodDefinition(
                description='Evaluate simulation results',
                inputs=['simulation_results', 'objective_target', 'iteration'],
                outputs=['error', 'evaluation_result']
            ),
            'suggest_parameters': MethodDefinition(
                description='Suggest parameters for next iteration',
                inputs=['evaluation_result', 'parameter_metadata'],
                outputs=['parameters']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the JointOptimizationNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
        self._optimization_history = []
        self._best_error = float('inf')
        self._best_params = {}
        self._best_simulation = None
        self._param_grid = None
        self._param_combinations = None
        self._current_combination = 0
        self._param_sources = {}  # Maps full parameter names to source nodes
        self._current_parameters = {}  # Tracks current parameters for all nodes
        
    def _define_process_steps(self) -> None:
        """Define the process steps for this node."""
        self.add_process_step(
            "evaluate",
            self.evaluate,
            method_key="evaluate"
        )
        
        self.add_process_step(
            "suggest_parameters",
            self.suggest_parameters,
            method_key="suggest_parameters"
        )
    
    def evaluate(self, simulation_results: Dict[str, Any], objective_target: float, 
                iteration: int = 0) -> Dict[str, Any]:
        """Evaluate simulation results using the objective function.
        
        Args:
            simulation_results: Simulation results (without parameters)
            objective_target: Target value for the objective function
            iteration: Current iteration number
            
        Returns:
            Dictionary with error and evaluation result
        """
        # We already know what parameters were used from our internal tracking
        all_parameters = self._current_parameters.copy()
        
        # Extract spike count from simulation results
        spike_count = simulation_results.get('spike_count', 0)
        if not spike_count and 'spike_times' in simulation_results:
            spike_count = len(simulation_results['spike_times'])
        
        # Calculate error using default objective function (spike count)
        print('here: ',objective_target)
        objective_target = 10 ##error here
        error = abs(spike_count - objective_target) 
        
        # Create evaluation result
        evaluation_result = {
            'iteration': iteration,
            'parameters': all_parameters,
            'error': error,
            'spike_count': spike_count,
            'objective_target': objective_target
        }
        
        # Update optimization history
        self._optimization_history.append(evaluation_result)
        
        # Update best result if better
        if error < self._best_error:
            self._best_error = error
            self._best_params = all_parameters.copy()
            self._best_simulation = simulation_results
            
            print(f"New best parameters at iteration {iteration}:")
            for name, value in self._best_params.items():
                print(f"  {name}: {value}")
            print(f"  Error: {self._best_error}")
        
        return {
            'error': error,
            'evaluation_result': evaluation_result
        }
    
    def suggest_parameters(self, evaluation_result: Dict[str, Any], 
                          parameter_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Suggest parameters for the next iteration.
        
        Implements grid search and random search optimization strategies for joint optimization
        of parameters from multiple nodes.
        
        Args:
            evaluation_result: Current evaluation result
            parameter_metadata: Metadata about optimizable parameters from multiple nodes
            
        Returns:
            Dictionary with suggested parameters for next iteration, grouped by node
        """
        # Get current iteration and parameters
        iteration = evaluation_result.get('iteration', 0)
        current_params = evaluation_result.get('parameters', {})
        
        # Get optimization method
        method = self._parameters['optimization_method']
        
        # Initialize next parameters
        next_params_flat = {}
        
        # If we have parameter metadata, use it to determine optimization ranges
        if parameter_metadata and not self._param_grid:
            # Store which node each parameter comes from
            self._param_sources = {}
            optimizable_params = {}
            
            # Collect optimizable parameters and their ranges from all nodes
            for node_name, node_metadata in parameter_metadata.items():
                for param_name, param_info in node_metadata.items():
                    if param_info.get('optimizable', False):
                        full_param_name = f"{node_name}.{param_name}"
                        self._param_sources[full_param_name] = node_name
                        
                        # Get current value from evaluation result if available
                        current_value = None
                        if node_name in evaluation_result.get('simulation_results', {}):
                            node_results = evaluation_result['simulation_results'][node_name]
                            if 'parameters' in node_results and param_name in node_results['parameters']:
                                current_value = node_results['parameters'][param_name]
                        
                        # Use default value if current value not available
                        if current_value is None and param_name in current_params:
                            current_value = current_params.get(param_name)
                        
                        # Store parameter with its optimization range
                        if current_value is not None:
                            optimizable_params[full_param_name] = {
                                'current_value': current_value,
                                'range': param_info.get('range', []),
                                'constraints': param_info.get('constraints', {})
                            }
            
            # Now create the parameter grid based on the method
            if method == 'grid':
                self._create_grid_search(optimizable_params)
            elif method == 'random':
                # For random search, we'll just store the parameter ranges
                self._param_ranges = optimizable_params
        
        # Generate next parameters based on the method
        if method == 'grid':
            next_params_flat = self._get_next_grid_parameters()
        elif method == 'random':
            next_params_flat = self._get_random_parameters()
        
        # Group parameters by their source node
        next_params_by_node = self._group_parameters_by_node(next_params_flat)
        
        # Store the current parameters for tracking
        self._current_parameters = next_params_flat.copy()
        
        return {
            'parameters': next_params_by_node
        }
    
    def _create_grid_search(self, optimizable_params: Dict[str, Dict[str, Any]]) -> None:
        """Create a grid search for the given optimizable parameters.
        
        Args:
            optimizable_params: Dictionary of optimizable parameters with their metadata
        """
        self._param_grid = {}
        grid_points = self._parameters['grid_points']
        
        for param_name, param_info in optimizable_params.items():
            current_value = param_info['current_value']
            
            # Use defined optimization range if available
            if param_info['range'] and len(param_info['range']) == 2:
                min_val, max_val = param_info['range']
            else:
                # Fallback to scaling current value
                min_val = current_value * 0.8
                max_val = current_value * 1.2
            
            # Create linspace for this parameter
            self._param_grid[param_name] = np.linspace(min_val, max_val, grid_points)
        
        # Generate all parameter combinations
        param_names = list(self._param_grid.keys())
        param_values = [self._param_grid[name] for name in param_names]
        self._param_combinations = list(product(*param_values))
        self._current_combination = 0
        
        print(f"Grid search: {len(self._param_combinations)} parameter combinations")
    
    def _get_next_grid_parameters(self) -> Dict[str, Any]:
        """Get the next parameter combination from the grid search.
        
        Returns:
            Dictionary with parameter names and values
        """
        if not self._param_grid or not self._param_combinations:
            return {}
        
        # If we've tried all combinations, return the best parameters
        if self._current_combination >= len(self._param_combinations):
            print("Grid search complete - using best parameters")
            return self._best_params.copy()
        
        # Get the next combination
        combination = self._param_combinations[self._current_combination]
        param_names = list(self._param_grid.keys())
        next_params = {name: value for name, value in zip(param_names, combination)}
        
        self._current_combination += 1
        return next_params
    
    def _get_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within the optimization ranges.
        
        Returns:
            Dictionary with parameter names and values
        """
        if not hasattr(self, '_param_ranges') or not self._param_ranges:
            return {}
        
        next_params = {}
        
        for param_name, param_info in self._param_ranges.items():
            current_value = param_info['current_value']
            
            # Use defined optimization range if available
            if param_info['range'] and len(param_info['range']) == 2:
                min_val, max_val = param_info['range']
            else:
                # Fallback to scaling current value
                min_val = current_value * 0.8
                max_val = current_value * 1.2
            
            # Generate random value within range
            next_params[param_name] = np.random.uniform(min_val, max_val)
        
        return next_params
    
    def _group_parameters_by_node(self, flat_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Group parameters by their source node.
        
        Args:
            flat_params: Dictionary with full parameter names (node.param) and values
            
        Returns:
            Dictionary with parameters grouped by node
        """
        params_by_node = {}
        
        for full_param_name, value in flat_params.items():
            # Split the parameter name to get node and parameter
            if '.' in full_param_name:
                node_name, param_name = full_param_name.split('.', 1)
            elif full_param_name in self._param_sources:
                # Use stored mapping if available
                node_name = self._param_sources[full_param_name]
                param_name = full_param_name.split('.', 1)[1] if '.' in full_param_name else full_param_name
            else:
                # Skip parameters we can't map to a node
                continue
            
            # Initialize node dictionary if needed
            if node_name not in params_by_node:
                params_by_node[node_name] = {}
            
            # Add parameter to the node
            params_by_node[node_name][param_name] = value
        
        return params_by_node
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get the current optimization results.
        
        Returns:
            Dictionary with optimization results
        """
        return {
            'best_parameters': self._best_params,
            'best_error': self._best_error,
            'best_simulation': self._best_simulation,
            'history': self._optimization_history
        }
    
    def reset_optimization(self) -> None:
        """Reset the optimization state."""
        self._optimization_history = []
        self._best_error = float('inf')
        self._best_params = {}
        self._best_simulation = None
        self._param_grid = None
        self._param_combinations = None
        self._current_combination = 0
        self._param_sources = {}
        self._current_parameters = {}