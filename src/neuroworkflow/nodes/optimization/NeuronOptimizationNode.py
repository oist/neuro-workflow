"""
Neuron optimization node for parameter optimization example.

This module provides a node for evaluating neuron simulation results and
calculating optimization errors.
"""

from typing import Dict, Any, List, Callable, Optional
import numpy as np

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType


class NeuronOptimizationNode(Node):
    """Node for evaluating neuron simulation results and calculating optimization errors."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='neuron_optimization',
        description='Evaluates neuron simulation results and calculates optimization errors',
        
        parameters={
            'optimization_method': ParameterDefinition(
                default_value='grid',
                description='Optimization method (grid, random)',
                constraints={'allowed_values': ['grid', 'random']}
            )
        },
        
        inputs={
            'simulation_results': PortDefinition(
                type=PortType.DICT,
                description='Simulation results to evaluate'
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
            'next_parameters': PortDefinition(
                type=PortType.DICT,
                description='Suggested parameters for next iteration'
            ),
            'parameters': PortDefinition(
                type=PortType.DICT,
                description='Parameters for the next iteration (for setup node)'
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
                inputs=['evaluation_result'],
                outputs=['next_parameters']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the NeuronOptimizationNode.
        
        Args:
            name: Name of the node
        """
        super().__init__(name)
        self._define_process_steps()
        self._optimization_history = []
        self._best_error = float('inf')
        self._best_params = {}
        self._best_simulation = None
        
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
            simulation_results: Simulation results
            objective_target: Target value for the objective function (e.g., desired spike count)
            iteration: Current iteration number
            
        Returns:
            Dictionary with error and evaluation result
        """
        # Extract parameters used in this simulation
        parameters = simulation_results.get('parameters', {})
        
        # Calculate error using default objective function (spike count)
        spike_count = len(simulation_results.get('spike_times', []))
        error = abs(spike_count - objective_target)
        
        # Create evaluation result
        evaluation_result = {
            'iteration': iteration,
            'parameters': parameters,
            'error': error,
            'spike_count': spike_count,
            'objective_target': objective_target,
            'simulation_results': simulation_results
        }
        
        # Update optimization history
        self._optimization_history.append(evaluation_result)
        
        # Update best result if better
        if error < self._best_error:
            self._best_error = error
            self._best_params = parameters.copy()
            self._best_simulation = simulation_results
            
            print(f"New best parameters at iteration {iteration}:")
            for name, value in self._best_params.items():
                print(f"  {name}: {value}")
            print(f"  Error: {self._best_error}")
        
        return {
            'error': error,
            'evaluation_result': evaluation_result
        }
    
    def suggest_parameters(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters for the next iteration.
        
        Implements grid search and random search optimization strategies.
        
        Args:
            evaluation_result: Current evaluation result
            
        Returns:
            Dictionary with suggested parameters for next iteration
        """
        # Get current iteration and parameters
        iteration = evaluation_result.get('iteration', 0)
        current_params = evaluation_result.get('parameters', {})
        
        # Get optimization method
        method = self._parameters['optimization_method']
        
        # Initialize next parameters
        next_params = {}
        
        if method == 'grid':
            # Grid search - generate parameter grid on first call
            if not hasattr(self, '_param_grid'):
                # Get all optimizable parameters from the first evaluation
                all_params = current_params.copy()
                
                # Create parameter grid
                self._param_grid = {}
                self._grid_points = 4  # Default grid points
                
                for param_name, param_value in all_params.items():
                    # Assume parameter ranges based on current value
                    # In a real implementation, this would come from the setup node
                    min_val = param_value * 0.8
                    max_val = param_value * 1.2
                    self._param_grid[param_name] = np.linspace(min_val, max_val, self._grid_points)
                
                # Generate all parameter combinations
                from itertools import product
                param_names = list(self._param_grid.keys())
                param_values = [self._param_grid[name] for name in param_names]
                self._param_combinations = list(product(*param_values))
                self._current_combination = 0
                
                print(f"Grid search: {len(self._param_combinations)} parameter combinations")
            
            # Get next parameter combination
            if self._current_combination < len(self._param_combinations):
                combination = self._param_combinations[self._current_combination]
                param_names = list(self._param_grid.keys())
                next_params = {name: value for name, value in zip(param_names, combination)}
                self._current_combination += 1
            else:
                # If we've tried all combinations, return the best parameters
                next_params = self._best_params.copy()
                print("Grid search complete - using best parameters")
                
        elif method == 'random':
            # Random search - generate random parameters
            # Get all optimizable parameters from the first evaluation
            all_params = current_params.copy()
            
            for param_name, param_value in all_params.items():
                # Assume parameter ranges based on current value
                # In a real implementation, this would come from the setup node
                min_val = param_value * 0.8
                max_val = param_value * 1.2
                next_params[param_name] = np.random.uniform(min_val, max_val)
        
        return {
            'next_parameters': next_params,
            'parameters': next_params  # Also output as 'parameters' for direct connection to setup node
        }
    
    
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
    
