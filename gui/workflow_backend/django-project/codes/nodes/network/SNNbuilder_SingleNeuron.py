"""
Single Neuron Builder Node for NeuroWorkflow Model Builder

This node creates a single neuron with biological and NEST-specific parameters.
It operates in dual modalities:
1. Direct execution: Creates actual NEST objects in the workflow
2. Script generation: Generates Python code for standalone execution

Author: NeuroWorkflow Team
Date: 2025
Version: 1.0
"""

from typing import Dict, Any, List, Optional, Union
import json

# Core NeuroWorkflow imports
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


class SNNbuilder_SingleNeuron(Node):
    """
    Single Neuron Builder Node for Neural Model Construction.
    
    This node creates a single neuron with comprehensive biological and 
    computational parameters. It can operate in two modes:
    
    1. **Execution Mode**: Creates actual NEST neuron objects
    2. **Script Generation Mode**: Generates Python code for standalone execution
    
    Biological Parameters:
    - Name and identification (name, acronym)
    - Cell type classification (model_type, excitatory/inhibitory)
    - Signaling properties (neurotransmitter types, PSP amplitudes, rise times)
    - Morphological properties (dendrite extent and diameter)
    - Activity properties (firing rate ranges for different states)
    
    NEST Parameters:
    - Base NEST model selection
    - Custom parameter overrides
    - Model template naming
    
    Outputs:
    - NEST neuron object (for workflow execution)
    - Python script code (for standalone execution)
    - Neuron metadata and properties
    """
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='single_neuron_builder',
        description='Build a single neuron with biological and NEST parameters',
        
        parameters={
            # === BIOLOGICAL IDENTIFICATION ===
            'name': ParameterDefinition(
                default_value='Neuron_1',
                description='Descriptive name for the neuron'
            ),
            
            'acronym': ParameterDefinition(
                default_value='N1',
                description='Short acronym or identifier for the neuron'
            ),
            
            'model_type': ParameterDefinition(
                default_value='point_process',
                description='Type of neuron model',
                constraints={'allowed_values': ['point_process', 'biophysical', 'other']}
            ),
            
            'cell_class': ParameterDefinition(
                default_value='excitatory',
                description='Functional classification of the neuron',
                constraints={'allowed_values': ['excitatory', 'inhibitory', 'other']}
            ),
            
            # === SIGNALING PROPERTIES ===
            'neurotransmitter_types': ParameterDefinition(
                default_value=['AMPA', 'NMDA'],
                description='List of neurotransmitter receptor types (AMPA, NMDA, GABA, etc.)'
            ),
            
            'psp_amplitudes': ParameterDefinition(
                default_value={'AMPA': 0.5, 'NMDA': 0.3},
                description='PSP amplitude (mV) for each neurotransmitter type'
            ),
            
            'rise_times': ParameterDefinition(
                default_value={'AMPA': 2.0, 'NMDA': 10.0},
                description='Rise time (ms) for each neurotransmitter type'
            ),
            
            # === MORPHOLOGICAL PROPERTIES ===
            'dendrite_extent': ParameterDefinition(
                default_value=200.0,
                description='Average maximal extent of dendritic field (μm)',
                constraints={'min': 10.0, 'max': 1000.0},
                optimizable=True,
                optimization_range=[50.0, 500.0]
            ),
            
            'dendrite_diameter': ParameterDefinition(
                default_value=2.0,
                description='Mean diameter of dendrites (μm)',
                constraints={'min': 0.1, 'max': 10.0},
                optimizable=True,
                optimization_range=[0.5, 5.0]
            ),
            
            # === ACTIVITY PROPERTIES ===
            'firing_rate_resting': ParameterDefinition(
                default_value=[1.0, 5.0],
                description='Firing rate range for resting state [min, max] Hz',
                constraints={'min_length': 2, 'max_length': 2}
            ),
            
            'firing_rate_active': ParameterDefinition(
                default_value=[10.0, 30.0],
                description='Firing rate range for active state [min, max] Hz',
                constraints={'min_length': 2, 'max_length': 2}
            ),
            
            'firing_rate_maximum': ParameterDefinition(
                default_value=[50.0, 100.0],
                description='Firing rate range for maximum activity [min, max] Hz',
                constraints={'min_length': 2, 'max_length': 2}
            ),
            
            'firing_rate_disease': ParameterDefinition(
                default_value=[0.1, 2.0],
                description='Firing rate range for disease condition [min, max] Hz',
                constraints={'min_length': 2, 'max_length': 2}
            ),
            
            # === NEST MODEL PARAMETERS ===
            'nest_model': ParameterDefinition(
                default_value='iaf_psc_alpha',
                description='Base NEST neuron model name'
            ),
            
            'nest_parameters': ParameterDefinition(
                default_value={'V_th': -55.0, 'C_m': 250.0, 'tau_m': 20.0},
                description='Custom NEST model parameters to override'
            ),
            
            'template_suffix': ParameterDefinition(
                default_value='_custom',
                description='Suffix for the custom model template name'
            ),
            
            # === EXECUTION OPTIONS ===
            'execution_mode': ParameterDefinition(
                default_value='both',
                description='Execution mode: execute, script, or both',
                constraints={'allowed_values': ['execute', 'script', 'both']}
            ),
            
            'script_format': ParameterDefinition(
                default_value='python',
                description='Format for generated script',
                constraints={'allowed_values': ['python', 'notebook', 'both']}
            )
        },
        
        inputs={
            # Optional inputs for parameter overrides
            'parameter_overrides': PortDefinition(
                type=PortType.DICT,
                description='Dictionary of parameter overrides',
                optional=True
            ),
            
            'nest_parameter_overrides': PortDefinition(
                type=PortType.DICT,
                description='Dictionary of NEST parameter overrides',
                optional=True
            )
        },
        
        outputs={
            # NEST execution outputs
            'nest_neuron': PortDefinition(
                type=PortType.OBJECT,
                description='Created NEST neuron object',
                optional=True
            ),
            
            'nest_model_name': PortDefinition(
                type=PortType.STR,
                description='Name of the created NEST model template'
            ),
            
            # Script generation outputs
            'python_script': PortDefinition(
                type=PortType.STR,
                description='Generated Python script code'
            ),
            
            'notebook_cell': PortDefinition(
                type=PortType.STR,
                description='Generated Jupyter notebook cell code',
                optional=True
            ),
            
            # Metadata outputs
            'neuron_metadata': PortDefinition(
                type=PortType.DICT,
                description='Complete neuron metadata and properties'
            ),
            
            'biological_properties': PortDefinition(
                type=PortType.DICT,
                description='Biological properties summary'
            ),
            
            'nest_properties': PortDefinition(
                type=PortType.DICT,
                description='NEST model properties and parameters'
            )
        },
        
        methods={
            'validate_parameters': MethodDefinition(
                description='Validate all neuron parameters',
                inputs=['parameter_overrides', 'nest_parameter_overrides'],
                outputs=['validation_result']
            ),
            
            'create_nest_neuron': MethodDefinition(
                description='Create NEST neuron object',
                inputs=['validation_result'],
                outputs=['nest_neuron', 'nest_model_name']
            ),
            
            'generate_python_script': MethodDefinition(
                description='Generate Python script code',
                inputs=['validation_result'],
                outputs=['python_script']
            ),
            
            'generate_notebook_cell': MethodDefinition(
                description='Generate Jupyter notebook cell',
                inputs=['validation_result'],
                outputs=['notebook_cell']
            ),
            
            'compile_metadata': MethodDefinition(
                description='Compile neuron metadata and properties',
                inputs=['validation_result'],
                outputs=['neuron_metadata', 'biological_properties', 'nest_properties']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the Single Neuron Builder Node."""
        super().__init__(name)
        
        # Internal state
        self._validated_parameters = {}
        self._model_template_name = None
        self._created_neuron = None
        
        # Define process steps
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the sequence of neuron building steps.
        
        Note: All process steps are added here. Individual methods will check
        execution_mode and script_format parameters to determine if they should
        actually execute or skip their processing.
        """
        
        # Always validate parameters first
        self.add_process_step(
            "validate_parameters",
            self.validate_parameters,
            method_key="validate_parameters"
        )
        
        # Add NEST neuron creation step (will check execution_mode internally)
        self.add_process_step(
            "create_nest_neuron",
            self.create_nest_neuron,
            method_key="create_nest_neuron"
        )
        
        # Add Python script generation step (will check execution_mode internally)
        self.add_process_step(
            "generate_python_script",
            self.generate_python_script,
            method_key="generate_python_script"
        )
        
        # Add notebook cell generation step (will check script_format internally)
        self.add_process_step(
            "generate_notebook_cell",
            self.generate_notebook_cell,
            method_key="generate_notebook_cell"
        )
        
        # Always compile metadata
        self.add_process_step(
            "compile_metadata",
            self.compile_metadata,
            method_key="compile_metadata"
        )
    
    # ========================================================================
    # MAIN PROCESSING METHODS
    # ========================================================================
    
    def validate_parameters(self, parameter_overrides: Optional[Dict[str, Any]] = None,
                          nest_parameter_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate all neuron parameters and apply overrides.
        
        Args:
            parameter_overrides: Optional parameter overrides
            nest_parameter_overrides: Optional NEST parameter overrides
            
        Returns:
            Dictionary with validated parameters
        """
        print(f"[{self.name}] Validating neuron parameters...")
        
        # Start with current parameters
        validated_params = self._parameters.copy()
        
        # Apply parameter overrides
        if parameter_overrides:
            for key, value in parameter_overrides.items():
                if key in validated_params:
                    validated_params[key] = value
                    print(f"[{self.name}] Override: {key} = {value}")
                else:
                    print(f"[{self.name}] Warning: Unknown parameter '{key}' in overrides")
        
        # Apply NEST parameter overrides
        if nest_parameter_overrides:
            nest_params = validated_params['nest_parameters'].copy()
            nest_params.update(nest_parameter_overrides)
            validated_params['nest_parameters'] = nest_params
            print(f"[{self.name}] NEST parameter overrides applied: {nest_parameter_overrides}")
        
        # Validate parameter consistency
        validation_errors = []
        validation_warnings = []
        
        # Check firing rate ranges
        for rate_type in ['firing_rate_resting', 'firing_rate_active', 'firing_rate_maximum']:
            rate_range = validated_params[rate_type]
            if len(rate_range) != 2 or rate_range[0] >= rate_range[1]:
                validation_errors.append(f"Invalid {rate_type}: must be [min, max] with min < max")
        
        # Check neurotransmitter consistency
        nt_types = validated_params['neurotransmitter_types']
        psp_amps = validated_params['psp_amplitudes']
        rise_times = validated_params['rise_times']
        
        for nt in nt_types:
            if nt not in psp_amps:
                validation_warnings.append(f"No PSP amplitude specified for {nt}")
            if nt not in rise_times:
                validation_warnings.append(f"No rise time specified for {nt}")
        
        # Check NEST model availability (if executing)
        if validated_params['execution_mode'] in ['execute', 'both'] and NEST_AVAILABLE:
            nest_model = validated_params['nest_model']
            try:
                # Check if model exists in NEST
                available_models = nest.node_models
                if nest_model not in available_models:
                    validation_errors.append(f"NEST model '{nest_model}' not available")
            except Exception as e:
                validation_warnings.append(f"Could not verify NEST model availability: {e}")
        
        # Generate model template name
        template_name = validated_params['nest_model'] + '_' + validated_params['acronym'] + validated_params['template_suffix']
        validated_params['_template_name'] = template_name
        
        # Store validated parameters
        self._validated_parameters = validated_params
        
        # Report validation results
        if validation_errors:
            error_msg = "Parameter validation failed: " + "; ".join(validation_errors)
            print(f"[{self.name}] {error_msg}")
            raise ValueError(error_msg)
        
        if validation_warnings:
            for warning in validation_warnings:
                print(f"[{self.name}] Warning: {warning}")
        
        print(f"[{self.name}] Parameter validation completed successfully")
        
        return {
            'validation_result': {
                'validated_parameters': validated_params,
                'template_name': template_name,
                'warnings': validation_warnings,
                'errors': validation_errors
            }
        }
    
    def create_nest_neuron(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create NEST neuron object.
        
        Args:
            validation_result: Results from parameter validation
            
        Returns:
            Dictionary with created NEST neuron and model name
        """
        validated_params = validation_result['validated_parameters']
        execution_mode = validated_params.get('execution_mode', 'both')
        
        # Check if NEST neuron creation is requested
        if execution_mode not in ['execute', 'both']:
            print(f"[{self.name}] Skipping NEST neuron creation (execution_mode: {execution_mode})")
            return {'nest_neuron': None, 'nest_model_name': None}
        
        print(f"[{self.name}] Creating NEST neuron...")
        
        if not NEST_AVAILABLE:
            print(f"[{self.name}] Warning: NEST not available, skipping neuron creation")
            return {'nest_neuron': None, 'nest_model_name': None}
        
        template_name = validation_result['template_name']
        
        try:
            # Step 1: Copy the base model
            base_model = validated_params['nest_model']
            nest_params = validated_params['nest_parameters'].copy()
            
            print(f"[{self.name}] Copying model: {base_model} -> {template_name}")
            nest.CopyModel(base_model, template_name)
            
            # Step 2: Handle multisynapse models - update tau_syn with PSP rise times (layered)
            if 'multisynapse' in base_model.lower():
                # Only apply PSP rise time mapping if user hasn't specified tau_syn
                if 'tau_syn' not in nest_params:
                    rise_times = validated_params['rise_times']
                    neurotransmitter_types = validated_params['neurotransmitter_types']
                    
                    # Create tau_syn tuple from rise times in the order of neurotransmitter types
                    tau_syn_values = []
                    for nt_type in neurotransmitter_types:
                        if nt_type in rise_times:
                            tau_syn_values.append(rise_times[nt_type])
                        else:
                            # Use default value if rise time not specified
                            default_rise_time = 2.0  # ms
                            tau_syn_values.append(default_rise_time)
                            print(f"[{self.name}] Warning: No rise time for {nt_type}, using default {default_rise_time} ms")
                    
                    # Convert to tuple for NEST
                    tau_syn_tuple = tuple(tau_syn_values)
                    nest_params['tau_syn'] = tau_syn_tuple
                    
                    print(f"[{self.name}] Multisynapse model detected: mapping PSP rise times to tau_syn = {tau_syn_tuple}")
                    print(f"[{self.name}] Neurotransmitter mapping: {dict(zip(neurotransmitter_types, tau_syn_values))}")
                else:
                    print(f"[{self.name}] Multisynapse model detected: user-specified tau_syn = {nest_params['tau_syn']} (PSP rise times ignored)")
                    print(f"[{self.name}] Note: User tau_syn takes priority over PSP rise time mapping")
            
            # Step 2b: Apply cell class-based parameter defaults (after user parameters, before final SetDefaults)
            cell_class = validated_params['cell_class']
            cell_class_params = self._get_cell_class_defaults(cell_class)
            
            # Apply cell class defaults only for parameters not already set by user
            applied_cell_params = {}
            for param_name, param_value in cell_class_params.items():
                if param_name not in nest_params:
                    nest_params[param_name] = param_value
                    applied_cell_params[param_name] = param_value
            
            if applied_cell_params:
                print(f"[{self.name}] Applied {cell_class} cell class defaults: {applied_cell_params}")
            
            # Step 3: Set default parameters
            if nest_params:
                print(f"[{self.name}] Setting defaults: {nest_params}")
                nest.SetDefaults(template_name, nest_params)
            
            # Step 4: Create the neuron
            print(f"[{self.name}] Creating neuron from template: {template_name}")
            neuron = nest.Create(template_name, 1)
            
            # Store created objects
            self._model_template_name = template_name
            self._created_neuron = neuron
            
            print(f"[{self.name}] NEST neuron created successfully: {neuron}")
            
            return {
                'nest_neuron': neuron,
                'nest_model_name': template_name
            }
            
        except Exception as e:
            error_msg = f"Failed to create NEST neuron: {e}"
            print(f"[{self.name}] Error: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def generate_python_script(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Python script code for the neuron.
        
        Args:
            validation_result: Results from parameter validation
            
        Returns:
            Dictionary with generated Python script
        """
        validated_params = validation_result['validated_parameters']
        execution_mode = validated_params.get('execution_mode', 'both')
        
        # Check if script generation is requested
        if execution_mode not in ['script', 'both']:
            print(f"[{self.name}] Skipping Python script generation (execution_mode: {execution_mode})")
            return {'python_script': ''}
        
        print(f"[{self.name}] Generating Python script...")
        
        template_name = validation_result['template_name']
        
        # Build the script
        script_lines = []
        
        # Header comment
        script_lines.append(f"# Single Neuron: {validated_params['name']} ({validated_params['acronym']})")
        script_lines.append(f"# Generated by NeuroWorkflow SingleNeuronBuilderNode")
        script_lines.append(f"# Model Type: {validated_params['model_type']}")
        script_lines.append(f"# Cell Class: {validated_params['cell_class']}")
        script_lines.append("")
        
        # Import statement
        script_lines.append("import nest")
        script_lines.append("")
        
        # Biological properties as comments
        script_lines.append("# === BIOLOGICAL PROPERTIES ===")
        script_lines.append(f"# Dendrite extent: {validated_params['dendrite_extent']} μm")
        script_lines.append(f"# Dendrite diameter: {validated_params['dendrite_diameter']} μm")
        script_lines.append(f"# Neurotransmitters: {validated_params['neurotransmitter_types']}")
        script_lines.append(f"# PSP amplitudes: {validated_params['psp_amplitudes']}")
        script_lines.append(f"# Rise times: {validated_params['rise_times']}")
        script_lines.append(f"# Firing rates - Resting: {validated_params['firing_rate_resting']} Hz")
        script_lines.append(f"# Firing rates - Active: {validated_params['firing_rate_active']} Hz")
        script_lines.append(f"# Firing rates - Maximum: {validated_params['firing_rate_maximum']} Hz")
        script_lines.append("")
        
        # NEST model creation
        script_lines.append("# === NEST MODEL CREATION ===")
        
        # CopyModel command
        base_model = validated_params['nest_model']
        script_lines.append(f'nest.CopyModel("{base_model}", "{template_name}")')
        
        # Prepare parameters (including tau_syn for multisynapse models and cell class defaults)
        nest_params = validated_params['nest_parameters'].copy()
        
        # Handle multisynapse models - update tau_syn with PSP rise times (layered)
        if 'multisynapse' in base_model.lower():
            # Only apply PSP rise time mapping if user hasn't specified tau_syn
            if 'tau_syn' not in nest_params:
                rise_times = validated_params['rise_times']
                neurotransmitter_types = validated_params['neurotransmitter_types']
                
                # Create tau_syn tuple from rise times in the order of neurotransmitter types
                tau_syn_values = []
                for nt_type in neurotransmitter_types:
                    if nt_type in rise_times:
                        tau_syn_values.append(rise_times[nt_type])
                    else:
                        # Use default value if rise time not specified
                        default_rise_time = 2.0  # ms
                        tau_syn_values.append(default_rise_time)
                
                # Convert to tuple for NEST
                tau_syn_tuple = tuple(tau_syn_values)
                nest_params['tau_syn'] = tau_syn_tuple
                
                # Add comment explaining the tau_syn mapping
                script_lines.append("")
                script_lines.append("# Multisynapse model: map PSP rise times to tau_syn")
                nt_mapping = dict(zip(neurotransmitter_types, tau_syn_values))
                for nt, tau in nt_mapping.items():
                    script_lines.append(f"# {nt}: {tau} ms")
            else:
                # User specified tau_syn - add comment explaining override
                script_lines.append("")
                script_lines.append("# Multisynapse model: user-specified tau_syn")
                script_lines.append(f"# (PSP rise times ignored: {validated_params['rise_times']})")
        
        # Apply cell class-based parameter defaults (same logic as execution mode)
        cell_class = validated_params['cell_class']
        cell_class_params = self._get_cell_class_defaults(cell_class)
        
        # Apply cell class defaults only for parameters not already set by user
        applied_cell_params = {}
        for param_name, param_value in cell_class_params.items():
            if param_name not in nest_params:
                nest_params[param_name] = param_value
                applied_cell_params[param_name] = param_value
        
        # Add comment about cell class defaults if any were applied
        if applied_cell_params:
            script_lines.append("")
            script_lines.append(f"# {cell_class.capitalize()} cell class defaults applied:")
            for param, value in applied_cell_params.items():
                script_lines.append(f"# {param}: {value}")
        
        # SetDefaults command
        if nest_params:
            # Format parameters nicely
            param_str = json.dumps(nest_params, indent=4).replace('\n', '\n' + ' ' * 20)
            script_lines.append(f'nest.SetDefaults("{template_name}", {param_str})')
        
        # Create command
        neuron_var = validated_params['acronym'].lower()
        script_lines.append(f'{neuron_var} = nest.Create("{template_name}", 1)')
        script_lines.append("")
        
        # Additional information
        script_lines.append(f'print(f"Created neuron: {{{neuron_var}}}")')
        script_lines.append(f'print(f"Model template: {template_name}")')
        
        # Join all lines
        python_script = '\n'.join(script_lines)
        
        print(f"[{self.name}] Python script generated ({len(script_lines)} lines)")
        
        return {'python_script': python_script}
    
    def generate_notebook_cell(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Jupyter notebook cell code.
        
        Args:
            validation_result: Results from parameter validation
            
        Returns:
            Dictionary with generated notebook cell
        """
        validated_params = validation_result['validated_parameters']
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
        
        # Get the Python script first
        python_result = self.generate_python_script(validation_result)
        python_script = python_result['python_script']
        
        # Create notebook cell format
        cell_lines = []
        
        # Markdown cell with neuron description
        cell_lines.append("# Markdown Cell")
        cell_lines.append("```markdown")
        cell_lines.append(f"## Neuron: {validated_params['name']} ({validated_params['acronym']})")
        cell_lines.append("")
        cell_lines.append("**Biological Properties:**")
        cell_lines.append(f"- Model Type: {validated_params['model_type']}")
        cell_lines.append(f"- Cell Class: {validated_params['cell_class']}")
        cell_lines.append(f"- Dendrite Extent: {validated_params['dendrite_extent']} μm")
        cell_lines.append(f"- Dendrite Diameter: {validated_params['dendrite_diameter']} μm")
        cell_lines.append(f"- Neurotransmitters: {', '.join(validated_params['neurotransmitter_types'])}")
        cell_lines.append("")
        cell_lines.append("**NEST Model:**")
        cell_lines.append(f"- Base Model: {validated_params['nest_model']}")
        cell_lines.append(f"- Custom Parameters: {validated_params['nest_parameters']}")
        cell_lines.append("```")
        cell_lines.append("")
        
        # Code cell
        cell_lines.append("# Code Cell")
        cell_lines.append("```python")
        cell_lines.append(python_script)
        cell_lines.append("```")
        
        notebook_cell = '\n'.join(cell_lines)
        
        print(f"[{self.name}] Notebook cell generated")
        
        return {'notebook_cell': notebook_cell}
    
    def compile_metadata(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile comprehensive neuron metadata.
        
        Args:
            validation_result: Results from parameter validation
            
        Returns:
            Dictionary with compiled metadata
        """
        print(f"[{self.name}] Compiling neuron metadata...")
        
        validated_params = validation_result['validated_parameters']
        
        # Biological properties
        biological_properties = {
            'identification': {
                'name': validated_params['name'],
                'acronym': validated_params['acronym'],
                'model_type': validated_params['model_type'],
                'cell_class': validated_params['cell_class']
            },
            'signaling': {
                'neurotransmitter_types': validated_params['neurotransmitter_types'],
                'psp_amplitudes': validated_params['psp_amplitudes'],
                'rise_times': validated_params['rise_times']
            },
            'morphology': {
                'dendrite_extent_um': validated_params['dendrite_extent'],
                'dendrite_diameter_um': validated_params['dendrite_diameter']
            },
            'activity': {
                'firing_rate_resting_hz': validated_params['firing_rate_resting'],
                'firing_rate_active_hz': validated_params['firing_rate_active'],
                'firing_rate_maximum_hz': validated_params['firing_rate_maximum'],
                'firing_rate_disease_hz': validated_params.get('firing_rate_disease')
            }
        }
        
        # NEST properties
        nest_properties = {
            'base_model': validated_params['nest_model'],
            'template_name': validation_result['template_name'],
            'custom_parameters': validated_params['nest_parameters'],
            'created_neuron': self._created_neuron,
            'execution_mode': validated_params['execution_mode']
        }
        
        # Complete metadata
        neuron_metadata = {
            'node_info': {
                'node_name': self.name,
                'node_type': self.NODE_DEFINITION.type,
                'creation_timestamp': self._get_timestamp()
            },
            'biological_properties': biological_properties,
            'nest_properties': nest_properties,
            'validation_info': {
                'warnings': validation_result.get('warnings', []),
                'errors': validation_result.get('errors', [])
            }
        }
        
        print(f"[{self.name}] Metadata compilation completed")
        
        return {
            'neuron_metadata': neuron_metadata,
            'biological_properties': biological_properties,
            'nest_properties': nest_properties
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_cell_class_defaults(self, cell_class: str) -> Dict[str, float]:
        """
        Get default NEST parameters based on cell class (excitatory/inhibitory).
        
        These defaults are applied only if the user hasn't specified the parameter,
        creating a layered parameter system:
        1. User-specified parameters (highest priority)
        2. Cell class defaults (medium priority) 
        3. NEST model defaults (lowest priority)
        
        Args:
            cell_class: 'excitatory', 'inhibitory', or 'other'
            
        Returns:
            Dictionary of default parameters for the cell class
        """
        
        if cell_class == 'excitatory':
            # Excitatory neurons: typically pyramidal cells
            # - Higher threshold (less excitable)
            # - Longer time constants (more integration)
            # - Standard refractory periods
            return {
                'V_th': -50.0,      # Higher threshold (mV)
                'V_reset': -70.0,   # Standard reset potential (mV)
                'tau_m': 20.0,      # Longer membrane time constant (ms)
                't_ref': 2.0        # Standard refractory period (ms)
            }
            
        elif cell_class == 'inhibitory':
            # Inhibitory neurons: typically interneurons (basket, chandelier, etc.)
            # - Lower threshold (more excitable)
            # - Shorter time constants (faster dynamics)
            # - Shorter refractory (higher firing rates)
            return {
                'V_th': -52.0,      # Lower threshold - more excitable (mV)
                'V_reset': -65.0,   # Less hyperpolarized reset (mV)
                'tau_m': 10.0,      # Shorter time constant - faster (ms)
                't_ref': 1.0        # Shorter refractory - higher rates (ms)
            }
            
        else:
            # 'other' or unknown cell class - no defaults applied
            return {}
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_neuron_summary(self) -> Dict[str, Any]:
        """Get a summary of the created neuron."""
        if not self._validated_parameters:
            return {'error': 'Neuron not yet created'}
        
        return {
            'name': self._validated_parameters['name'],
            'acronym': self._validated_parameters['acronym'],
            'nest_model': self._validated_parameters['nest_model'],
            'template_name': self._model_template_name,
            'created_neuron': self._created_neuron,
            'execution_mode': self._validated_parameters['execution_mode']
        }
    
    def export_script_to_file(self, filename: Optional[str] = None) -> str:
        """
        Export generated Python script to file.
        
        Args:
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to the exported file
        """
        if not filename:
            acronym = self._validated_parameters.get('acronym', 'neuron')
            filename = f"{acronym}_neuron_model.py"
        
        # Get the script from output port
        script = self.get_output_port('python_script').value
        
        if script:
            with open(filename, 'w') as f:
                f.write(script)
            print(f"[{self.name}] Script exported to: {filename}")
            return filename
        else:
            raise ValueError("No Python script available to export")

