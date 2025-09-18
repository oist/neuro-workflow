"""
PopulationBuilderNode - NeuroWorkflow Model Builder for Neural Populations

This node creates populations of neurons with real-scale brain parameters, spatial positioning,
and biological context. It integrates with SingleNeuronBuilderNode to create populations
from individual neuron specifications.

Key Features:
- Real-scale population sizes (single brain hemisphere)
- Spatial positioning in 2D/3D coordinates  
- Multiple model types: point_process, mean_field, rate_model
- Tissue density reference for population size estimation
- Mean firing rate specification
- Script generation: Generates Python code for standalone execution
- Dual modality: NEST execution + Python script generation

Author: NeuroWorkflow Team
Created: 2025-01-11
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

# Core NeuroWorkflow imports - REQUIRED for all custom nodes
from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, 
    PortDefinition, 
    ParameterDefinition, 
    MethodDefinition
)
from neuroworkflow.core.port import PortType

# NEST import (optional - will handle if not available)
try:
    import nest
    NEST_AVAILABLE = True
except ImportError:
    NEST_AVAILABLE = False
    print("Warning: NEST not available. Node will work in script generation mode only.")


class SNNbuilder_Population(Node):
    """
    A NeuroWorkflow node for building neural populations with biological context.
    
    This node creates populations of neurons based on real brain anatomy and physiology,
    including spatial positioning, population sizes, and firing rate characteristics.
    It can work with different model types and provides both NEST execution and 
    Python script generation capabilities.
    
    The node receives neuron objects from SingleNeuronBuilderNode and creates populations
    using NEST's nest.Create() function with spatial positioning support.
    """
    
    # NODE_DEFINITION is REQUIRED - this defines the node's interface
    NODE_DEFINITION = NodeDefinitionSchema(
        # REQUIRED: Unique identifier for the node type
        type='population_builder',
        
        # REQUIRED: Human-readable description
        description='Creates neural populations with real-scale brain parameters and spatial positioning',
        
        # OPTIONAL: Parameters that control the node's behavior
        parameters={
            # Population identification
            'name': ParameterDefinition(
                default_value='Neural Population',
                description='Full name of the population (e.g., "Layer 2/3 Pyramidal Neurons")'
            ),
            'acronym': ParameterDefinition(
                default_value='NeurPop',
                description='Short identifier (e.g., "L23Pyr")'
            ),
            'brain_region': ParameterDefinition(
                default_value='Cortex',
                description='Anatomical region (e.g., "Primary Visual Cortex")'
            ),
            # Model characteristics
            'model_type': ParameterDefinition(
                default_value='point_process',
                description='Type of neural model',
                constraints={'allowed_values': ['point_process', 'mean_field', 'rate_model']}
            ),
            # Population size and reference
            'population_size': ParameterDefinition(
                default_value=1000,
                description='Number of neurons in the population (single hemisphere)',
                constraints={'min': 1}
            ),
            'tissue_reference': ParameterDefinition(
                default_value={},
                description='Reference data for population size estimation (density, volume, study, etc.)'
            ),
            # Spatial organization
            'spatial_dimensions': ParameterDefinition(
                default_value='3D',
                description='Spatial organization: 2D or 3D',
                constraints={'allowed_values': ['2D', '3D']}
            ),
            'spatial_bounds': ParameterDefinition(
                default_value={'x': (0.0, 1000.0), 'y': (0.0, 1000.0), 'z': (0.0, 500.0)},
                description='Spatial boundaries for position generation in μm'
            ),
            # Custom spatial positions
            'custom_positions': ParameterDefinition(
                default_value=None,
                description='Custom positions array - numpy array or list of coordinates. For 2D: [[x1,y1], [x2,y2], ...] or numpy array shape (N,2). For 3D: [[x1,y1,z1], [x2,y2,z2], ...] or numpy array shape (N,3). Must match population_size. Units: micrometers (μm). Overrides spatial generation when provided.'
            ),
            'mean_firing_rate': ParameterDefinition(
                default_value=None,
                description='Average firing rate (Hz) or (min, max) range'
            ),
            # Execution mode
            'execution_mode': ParameterDefinition(
                default_value='both',
                description='How to create the population',
                constraints={'allowed_values': ['execute', 'script', 'both']}
            ),
            'script_format': ParameterDefinition(
                default_value='python',
                description='Format for generated script',
                constraints={'allowed_values': ['python', 'notebook', 'both']}
            )
        },
        
        # Input ports - data the node receives
        inputs={
            # From SingleNeuronBuilderNode - NEST execution outputs
            'nest_neuron': PortDefinition(
                type=PortType.OBJECT,
                description='Created NEST neuron object from SingleNeuronBuilderNode',
                optional=True
            ),
            'nest_model_name': PortDefinition(
                type=PortType.STR,
                description='Name of the created NEST model template',
                optional=True
            ),
            
            # From SingleNeuronBuilderNode - Script generation outputs
            'python_script': PortDefinition(
                type=PortType.STR,
                description='Generated Python script code from SingleNeuronBuilderNode',
                optional=True
            ),
            'notebook_cell': PortDefinition(
                type=PortType.STR,
                description='Generated Jupyter notebook cell code',
                optional=True
            ),
            
            # From SingleNeuronBuilderNode - Metadata outputs
            'neuron_metadata': PortDefinition(
                type=PortType.DICT,
                description='Complete neuron metadata and properties',
                optional=True
            ),
            'biological_properties': PortDefinition(
                type=PortType.DICT,
                description='Biological properties summary',
                optional=True
            ),
            'nest_properties': PortDefinition(
                type=PortType.DICT,
                description='NEST model properties and parameters',
                optional=True
            )
        },
        
        # Output ports - data the node produces
        outputs={
            'nest_population': PortDefinition(
                type=PortType.OBJECT,
                description='NEST population object (if execution mode)'
            ),
            'population_data': PortDefinition(
                type=PortType.DICT,
                description='Population metadata and configuration',
                optional=True
            ),
            'python_script': PortDefinition(
                type=PortType.STR,
                description='Generated Python code for population creation'
            ),
            'notebook_cell': PortDefinition(
                type=PortType.STR,
                description='Generated Jupyter notebook cell code for population creation',
                optional=True
            )
        },
        
        # Methods that can be called on this node
        methods={
            'validate_parameters': MethodDefinition(
                description='Validate population parameters and prepare for creation',
                inputs=[],
                outputs=['validated_params']
            ),
            'create_nest_population': MethodDefinition(
                description='Create NEST population using validated parameters',
                inputs=['validated_params'],
                outputs=['nest_population','population_data']
            ),
            'generate_python_script': MethodDefinition(
                description='Generate Python script for population creation',
                inputs=['validated_params'],
                outputs=['python_script']
            ),
            'generate_notebook_cell': MethodDefinition(
                description='Generate Jupyter notebook cell for population creation',
                inputs=['validated_params'],
                outputs=['notebook_cell']
            ),
            'get_population_summary': MethodDefinition(
                description='Get summary of the created population',
                inputs=[],
                outputs=['summary']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the PopulationBuilderNode."""
        super().__init__(name)
        
        # Node state
        self._validated_parameters = None
        self._created_population = None
        self._generated_script = None
        
        # Define process steps
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define process steps for this node.
        
        Note: All process steps are added here. Individual methods will check
        execution_mode and script_format parameters to determine if they should
        actually execute or skip their processing.
        """
        self.add_process_step(
            'validate_parameters',
            self.validate_parameters,
            method_key='validate_parameters'
        )
        
        # Add NEST population creation step (will check execution_mode internally)
        self.add_process_step(
            'create_nest_population',
            self.create_nest_population,
            method_key='create_nest_population'
        )
        
        # Add Python script generation step (will check execution_mode internally)
        self.add_process_step(
            'generate_python_script',
            self.generate_python_script,
            method_key='generate_python_script'
        )
        
        # Add notebook cell generation step (will check script_format internally)
        self.add_process_step(
            'generate_notebook_cell',
            self.generate_notebook_cell,
            method_key='generate_notebook_cell'
        )
        
        self.add_process_step(
            'get_population_summary',
            self.get_population_summary,
            method_key='get_population_summary'
        )
    
    # ========================================================================
    # VALIDATION METHOD
    # ========================================================================
    
    def validate_parameters(self) -> Dict[str, Any]:
        """
        Validate population parameters and prepare for creation.
        
        Returns:
            Dictionary containing validation results and processed parameters
        """
        print(f"[{self.name}] Validating population parameters...")
        
        errors = []
        warnings = []
        
        # Get parameters
        population_size = self._parameters.get('population_size')
        spatial_dimensions = self._parameters.get('spatial_dimensions')
        model_type = self._parameters.get('model_type')
        mean_firing_rate = self._parameters.get('mean_firing_rate')
        
        # Check population size
        if population_size <= 0:
            errors.append("Population size must be positive")
        

        
        # Report validation results
        if errors:
            error_msg = f"[{self.name}] Validation failed: " + "; ".join(errors)
            print(error_msg)
            raise ValueError(error_msg)
        
        if warnings:
            for warning in warnings:
                print(f"[{self.name}] Warning: {warning}")
        
        # Generate spatial positions if needed
        positions = self._generate_positions()
        
        # Prepare validated parameters
        validated_params = {
            'name': self._parameters.get('name'),
            'acronym': self._parameters.get('acronym'),
            'brain_region': self._parameters.get('brain_region'),
            'population_size': population_size,
            'tissue_reference': self._parameters.get('tissue_reference'),
            'spatial_dimensions': spatial_dimensions,
            'positions': positions,
            'spatial_bounds': self._parameters.get('spatial_bounds'),
            'model_type': model_type,
            'mean_firing_rate': mean_firing_rate,
            'execution_mode': self._parameters.get('execution_mode'),
            'script_format': self._parameters.get('script_format')
        }
        
        self._validated_parameters = validated_params
        
        print(f"[{self.name}] Parameter validation completed successfully")
        return {
            'success': True,
            'validated_params': validated_params,
            'warnings': warnings
        }
    

    # ========================================================================
    # NEST POPULATION CREATION
    # ========================================================================
    
    def create_nest_population(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a NEST population using the validated parameters.
        
        Args:
            validated_params: Validated parameters from validate_parameters()
            
        Returns:
            Dictionary containing NEST population and metadata
        """
        execution_mode = validated_params.get('execution_mode', 'both')
        
        # Check if NEST population creation is requested
        if execution_mode not in ['execute', 'both']:
            print(f"[{self.name}] Skipping NEST population creation (execution_mode: {execution_mode})")
            return {'nest_population': None, 'population_data': None}
        
        print(f"[{self.name}] Creating NEST population...")
        
        if not NEST_AVAILABLE:
            print(f"[{self.name}] Warning: NEST not available, skipping population creation")
            return {'nest_population': None, 'population_data': None}
        
        try:
            # Get neuron model information from input ports (not validated_params)
            nest_model_name = self._input_ports['nest_model_name'].value if 'nest_model_name' in self._input_ports else None
            nest_properties = self._input_ports['nest_properties'].value if 'nest_properties' in self._input_ports else None
            
            if nest_model_name:
                model_name = nest_model_name
            else:
                # Use default model based on model type
                print(f"[{self.name}] Warning: Model was not obtained from SingleNeuronBuilderNode")
                model_type = validated_params['model_type']
                if model_type == 'point_process':
                    model_name = 'iaf_psc_alpha'
                elif model_type == 'rate_model':
                    model_name = 'rate_neuron_ipn'
                else:
                    model_name = 'iaf_psc_alpha'  # fallback
            
            population_size = validated_params['population_size']
            positions = validated_params['positions']
            
            print(f"[{self.name}] Creating {population_size} neurons of type '{model_name}'")
            
            # Create NEST population with positions and parameters
            if positions is not None:
                # Convert positions to NEST spatial format
                nest_positions = nest.spatial.free(positions.tolist())
                custom_positions = self._parameters.get('custom_positions')
                if custom_positions is not None:
                    print(f"[{self.name}] Create NEST population with custom positions from parameters")
                else:
                    print(f"[{self.name}] Create NEST population with generated positions from helper method")
                
                population = nest.Create(model_name, population_size, positions=nest_positions)
                print(f"[{self.name}] Population created with {validated_params['spatial_dimensions']} spatial positions")
            else:
                if nest_properties:
                    population = nest.Create(model_name, population_size, params=nest_properties)
                else:
                    population = nest.Create(model_name, population_size)
                print(f"[{self.name}] Population created without spatial positions")
               
            # Print status of one neuron from the population
            if population:
                neuron_status = nest.GetStatus(population[0])
                print(f"[{self.name}] Status of first neuron: {neuron_status}")
            
            results={
                'nest_population': population,
                'population_data': {'population_size': population_size,
                    'model_name': model_name,
                    'positions': positions,
                    'spatial_dimensions': validated_params['spatial_dimensions']
                    }
            }
            self._created_population = results
            print(f"[{self.name}] NEST population created successfully: {population}")
            return results
            
        except Exception as e:
            print(f"[{self.name}] Error creating NEST population: {e}")
            return {
                'nest_population': None,
                'population_data': {'population_size': validated_params['population_size'],
                    'model_name': None,
                    'positions': validated_params['positions'],
                    'error': str(e)
                    }
            }
    

    # ========================================================================
    # SCRIPT GENERATION
    # ========================================================================
    
    def generate_python_script(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Python script for population creation.
        
        Args:
            validated_params: Validated parameters from validate_parameters()
            
        Returns:
            Dictionary with generated Python script
        """
        execution_mode = validated_params.get('execution_mode', 'both')
        
        # Check if script generation is requested
        if execution_mode not in ['script', 'both']:
            print(f"[{self.name}] Skipping Python script generation (execution_mode: {execution_mode})")
            return {'python_script': ''}
        
        print(f"[{self.name}] Generating Python script...")
        
        script_lines = []
        
        # Add header comment
        script_lines.append(f"# Population: {validated_params['name']} ({validated_params['acronym']})")
        script_lines.append(f"# Generated by NeuroWorkflow PopulationBuilderNode")
        script_lines.append(f"# Brain Region: {validated_params['brain_region']}")
        script_lines.append(f"# Population Size: {validated_params['population_size']} neurons (single hemisphere)")
        script_lines.append(f"# Model Type: {validated_params['model_type']}")
        script_lines.append(f"# Spatial: {validated_params['spatial_dimensions']}")
        script_lines.append("")
        
        # Add imports
        script_lines.append("import numpy as np")
        script_lines.append("")
        
        # Add biological context as comments
        script_lines.append("# === POPULATION BIOLOGICAL CONTEXT ===")
        tissue_ref = validated_params['tissue_reference']
        if tissue_ref:
            if 'density_per_mm3' in tissue_ref:
                script_lines.append(f"# Density: {tissue_ref['density_per_mm3']} neurons/mm³")
            if 'density_per_mm2' in tissue_ref:
                script_lines.append(f"# Density: {tissue_ref['density_per_mm2']} neurons/mm²")
            if 'tissue_volume_mm3' in tissue_ref:
                script_lines.append(f"# Tissue volume: {tissue_ref['tissue_volume_mm3']} mm³")
            if 'tissue_area_mm2' in tissue_ref:
                script_lines.append(f"# Tissue area: {tissue_ref['tissue_area_mm2']} mm²")
            if 'reference_study' in tissue_ref:
                script_lines.append(f"# Reference: {tissue_ref['reference_study']}")
            if 'species' in tissue_ref:
                script_lines.append(f"# Species: {tissue_ref['species']}")
            if 'estimation_method' in tissue_ref:
                script_lines.append(f"# Estimation method: {tissue_ref['estimation_method']}")
        
        script_lines.append(f"# Spatial bounds: {validated_params['spatial_bounds']}")
        script_lines.append("")
        
        # Generate positions section
        positions = validated_params['positions']
        spatial_dims = validated_params['spatial_dimensions']
        custom_positions = self._parameters.get('custom_positions')
        
        if positions is not None:
            script_lines.append("# === SPATIAL POSITIONS ===")
            script_lines.append(f"# {spatial_dims} positions for {len(positions)} neurons")
            script_lines.append(f"# Spatial bounds: {validated_params['spatial_bounds']}")
            script_lines.append("")
            
            if custom_positions is not None:
                # Use custom positions
                script_lines.append("# Using custom positions")
                script_lines.append("positions = np.array([")
                for i, pos in enumerate(positions):
                    if len(pos) == 2:
                        script_lines.append(f"    [{pos[0]:.1f}, {pos[1]:.1f}],  # Neuron {i+1}")
                    else:  # 3D
                        script_lines.append(f"    [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}],  # Neuron {i+1}")
                script_lines.append("])")
            else:
                # Generate random positions
                script_lines.append("# Generate spatial positions")
                bounds = validated_params['spatial_bounds']
                population_size = validated_params['population_size']
                
                if spatial_dims == '2D':
                    script_lines.append(f"positions = np.zeros(({population_size}, 2))")
                    script_lines.append(f"positions[:, 0] = np.random.uniform({bounds['x'][0]}, {bounds['x'][1]}, {population_size})  # x coordinates")
                    script_lines.append(f"positions[:, 1] = np.random.uniform({bounds['y'][0]}, {bounds['y'][1]}, {population_size})  # y coordinates")
                else:  # 3D
                    script_lines.append(f"positions = np.zeros(({population_size}, 3))")
                    script_lines.append(f"positions[:, 0] = np.random.uniform({bounds['x'][0]}, {bounds['x'][1]}, {population_size})  # x coordinates")
                    script_lines.append(f"positions[:, 1] = np.random.uniform({bounds['y'][0]}, {bounds['y'][1]}, {population_size})  # y coordinates")
                    script_lines.append(f"positions[:, 2] = np.random.uniform({bounds['z'][0]}, {bounds['z'][1]}, {population_size})  # z coordinates")
            
            script_lines.append("")
        
        # Population creation section
        script_lines.append("# === POPULATION CREATION ===")
        
        # Get model name from input ports
        nest_model_name = self._input_ports['nest_model_name'].value if 'nest_model_name' in self._input_ports else None
        if nest_model_name:
            model_name = nest_model_name
            script_lines.append(f"# Using neuron model from SingleNeuronBuilderNode: {model_name}")
        else:
            if validated_params['model_type'] == 'point_process':
                model_name = 'iaf_psc_alpha'
            elif validated_params['model_type'] == 'rate_model':
                model_name = 'rate_neuron_ipn'
            else:
                model_name = 'iaf_psc_alpha'
            script_lines.append(f"# Using default model for {validated_params['model_type']}: {model_name}")
        
        population_size = validated_params['population_size']
        acronym = validated_params['acronym'].lower()
        
        # Create population with or without positions
        if positions is not None:
            script_lines.append(f"# Create population with {spatial_dims} spatial positions")
            script_lines.append(f"nest_positions = nest.spatial.free(positions.tolist())")
            script_lines.append(f"{acronym} = nest.Create(\"{model_name}\", {population_size}, positions=nest_positions)")
        else:
            script_lines.append(f"# Create population without spatial positions")
            script_lines.append(f"{acronym} = nest.Create(\"{model_name}\", {population_size})")
        
        script_lines.append("")
        
        # Print status of one neuron from the population
        script_lines.append("# Print status of one neuron from the population")
        script_lines.append(f"neuron_status = nest.GetStatus({acronym}[0])")
        script_lines.append(f"print(f\"Status of first neuron: {{neuron_status}}\")")
        script_lines.append("")
        
        # Add summary
        script_lines.append("# === POPULATION SUMMARY ===")
        script_lines.append(f"print(f\"Created population: {{len({acronym})}} {validated_params['name']}\")")
        script_lines.append(f"print(f\"Population IDs: {{{acronym}}}\")")
        script_lines.append(f"print(f\"Model: {model_name}\")")
        if positions is not None:
            script_lines.append(f"print(f\"Spatial organization: {spatial_dims} positions\")")
        
        # Build the final script
        python_script = "\n".join(script_lines)
        
        # Set output port
        self._output_ports['python_script'].value = python_script
        
        print(f"[{self.name}] Python script generated ({len(script_lines)} lines)")
        
        return {'python_script': python_script}
    
    def generate_notebook_cell(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Jupyter notebook cell code for population creation.
        
        Args:
            validated_params: Validated parameters from validate_parameters()
            
        Returns:
            Dictionary with generated notebook cell
        """
        execution_mode = validated_params.get('execution_mode', 'both')
        script_format = validated_params.get('script_format', 'python')
        
        # Check if notebook cell generation is requested
        if execution_mode not in ['script', 'both']:
            print(f"[{self.name}] Skipping notebook cell generation (execution_mode: {execution_mode})")
            return {'notebook_cell': ''}
        
        if script_format not in ['notebook', 'both']:
            print(f"[{self.name}] Skipping notebook cell generation (script_format: {script_format})")
            return {'notebook_cell': ''}
        
        print(f"[{self.name}] Generating Jupyter notebook cell...")
        
        # Get neuron properties from input ports for notebook generation
        nest_properties = self._input_ports['nest_properties'].value if 'nest_properties' in self._input_ports else None
        nest_model_name = self._input_ports['nest_model_name'].value if 'nest_model_name' in self._input_ports else None
        
        # Get model name
        if nest_model_name:
            model_name = nest_model_name
        else:
            if validated_params['model_type'] == 'point_process':
                model_name = 'iaf_psc_alpha'
            elif validated_params['model_type'] == 'rate_model':
                model_name = 'rate_neuron_ipn'
            else:
                model_name = 'iaf_psc_alpha'
        
        positions = validated_params['positions']
        spatial_dims = validated_params['spatial_dimensions']
        custom_positions = self._parameters.get('custom_positions')
        population_size = validated_params['population_size']
        acronym = validated_params['acronym'].lower()
        
        notebook_lines = []
        notebook_lines.append(f"# Population: {validated_params['name']}")
        notebook_lines.append(f"# {population_size} neurons, {spatial_dims}")
        notebook_lines.append("")
        
        # Add key creation code for notebook
        if positions is not None:
            notebook_lines.append("# Create spatial positions")
            if custom_positions is not None:
                # Use custom positions in notebook
                notebook_lines.append("positions = np.array([")
                for i, pos in enumerate(positions):
                    if len(pos) == 2:
                        notebook_lines.append(f"    [{pos[0]:.1f}, {pos[1]:.1f}],  # Neuron {i+1}")
                    else:  # 3D
                        notebook_lines.append(f"    [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}],  # Neuron {i+1}")
                notebook_lines.append("])")
            else:
                # Generate random positions in notebook
                bounds = validated_params['spatial_bounds']
                if spatial_dims == '2D':
                    notebook_lines.append(f"positions = np.random.uniform([{bounds['x'][0]}, {bounds['y'][0]}], [{bounds['x'][1]}, {bounds['y'][1]}], ({population_size}, 2))")
                else:
                    notebook_lines.append(f"positions = np.random.uniform([{bounds['x'][0]}, {bounds['y'][0]}, {bounds['z'][0]}], [{bounds['x'][1]}, {bounds['y'][1]}, {bounds['z'][1]}], ({population_size}, 3))")
            notebook_lines.append(f"nest_positions = nest.spatial.free(positions.tolist())")
            notebook_lines.append(f"{acronym} = nest.Create('{model_name}', {population_size}, positions=nest_positions)")
        else:
            notebook_lines.append(f"{acronym} = nest.Create('{model_name}', {population_size})")
        
        # Add status check
        notebook_lines.append(f"neuron_status = nest.GetStatus({acronym}[0])")
        notebook_lines.append(f"print(f\"Status of first neuron: {{neuron_status}}\")")
        
        notebook_cell = "\n".join(notebook_lines)
        
        # Set output port
        self._output_ports['notebook_cell'].value = notebook_cell
        
        print(f"[{self.name}] Notebook cell generated ({len(notebook_lines)} lines)")
        
        return {'notebook_cell': notebook_cell}
    
    # ========================================================================
    # GET A SUMMARY
    # ========================================================================
    
    def get_population_summary(self) -> Dict[str, Any]:
        """Get a summary of the created population."""
        if not self._validated_parameters:
            return {'error': 'No population created yet'}
        
        params = self._validated_parameters
        summary = {
            'name': params['name'],
            'acronym': params['acronym'],
            'brain_region': params['brain_region'],
            'population_size': params['population_size'],
            'model_type': params['model_type'],
            'spatial_dimensions': params['spatial_dimensions'],
            'has_positions': params['positions'] is not None,
            'mean_firing_rate': params['mean_firing_rate'],
            'tissue_reference': params['tissue_reference']
        }
        
        if self._created_population:
            summary['nest_created'] = True
            summary['nest_population'] = self._created_population['nest_population']
        
        if self._generated_script:
            summary['script_generated'] = True
            summary['script_lines'] = len(self._generated_script['python_script'].split('\n'))
        
        return {'summary': summary}

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _generate_positions(self) -> Optional[np.ndarray]:
        """Generate spatial positions for neurons if not provided via parameter."""
        # Check if custom positions provided via parameter
        custom_positions = self._parameters.get('custom_positions')
        if custom_positions is not None:
            return np.array(custom_positions)
        
        # Generate random positions within bounds
        population_size = self._parameters.get('population_size')
        spatial_dims = self._parameters.get('spatial_dimensions')
        bounds = self._parameters.get('spatial_bounds')
        
        if spatial_dims == '2D':
            positions = np.zeros((population_size, 2))
            positions[:, 0] = np.random.uniform(bounds['x'][0], bounds['x'][1], population_size)
            positions[:, 1] = np.random.uniform(bounds['y'][0], bounds['y'][1], population_size)
        else:  # 3D
            positions = np.zeros((population_size, 3))
            positions[:, 0] = np.random.uniform(bounds['x'][0], bounds['x'][1], population_size)
            positions[:, 1] = np.random.uniform(bounds['y'][0], bounds['y'][1], population_size)
            positions[:, 2] = np.random.uniform(bounds['z'][0], bounds['z'][1], population_size)
        
        print(f"[{self.name}] Generated random {spatial_dims} positions for {population_size} neurons")
        return positions

