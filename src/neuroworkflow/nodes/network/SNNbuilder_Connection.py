"""
ConnectionBuilderNode - NeuroWorkflow Model Builder for Neural Connections

This node creates connections between neural populations with comprehensive biological
and computational parameters. It supports multiple NEST connection rules, synapse models,
and biological connection properties.

Key Features:
- NEST synapse model handling with parameter customization
- Multiple connection rules: pairwise_bernoulli, fixed_indegree, fixed_outdegree
- Biological connection parameters: axon organization, bouton number, receptor location
- Spatial connectivity with masks and distance-dependent properties
- PSP-based weight estimation
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

class SNNbuilder_Connection(Node):
    """
    A NeuroWorkflow node for building neural connections with biological context.
    
    This node creates connections between neural populations using NEST's spatial
    connectivity features. It supports multiple connection rules, synapse models,
    and biological parameters that influence connection strength and organization.
    
    The node can work with different connection patterns and provides both NEST 
    execution and Python script generation capabilities.
    """
    
    # NODE_DEFINITION is REQUIRED - this defines the node's interface
    NODE_DEFINITION = NodeDefinitionSchema(
        # REQUIRED: Unique identifier for the node type
        type='connection_builder',
        
        # REQUIRED: Human-readable description
        description='Creates neural connections between populations with biological and spatial parameters',
        
        # OPTIONAL: Parameters that control the node's behavior
        parameters={
            # === CONNECTION IDENTIFICATION ===
            'name': ParameterDefinition(
                default_value='Neural Connection',
                description='Descriptive name for the connection (e.g., "L2/3 to L4 Connection")'
            ),
            
            # === CONNECTION RULE ===
            'connection_rule': ParameterDefinition(
                default_value='pairwise_bernoulli',
                description='NEST connection rule type',
                constraints={'allowed_values': ['pairwise_bernoulli', 'fixed_indegree', 'fixed_outdegree']}
            ),
            
            # === 1. CONNECTION DICTIONARY (cdict) ===
            'connection_dict': ParameterDefinition(
                default_value={
                    #'p': 0.1,
                    'allow_autapses': False,
                    'allow_multapses': False
                },
                description='NEST connection dictionary with probability/degree and constraints (rule comes from connection_rule parameter)'
            ),
            
            # === 2. SYNAPSE MODEL ===
            'synapse_model': ParameterDefinition(
                default_value='static_synapse',
                description='NEST synapse model name',
                constraints={'allowed_values': ['static_synapse', 'stdp_synapse', 'tsodyks_synapse', 'tsodyks2_synapse']}
            ),
            
            # === 3. SYNAPSE DICTIONARY (sdict) ===
            'synapse_dict': ParameterDefinition(
                default_value={
                    'weight': 1.0,
                    'delay': 1.0,
                },
                description='NEST synapse dictionary with weight, delay, receptor type, and custom parameters (synapse_model will be added automatically)'
            ),
            
            # === 4. SYNAPSE MODEL CUSTOMIZATION ===
            'template_suffix': ParameterDefinition(
                default_value='_custom',
                description='Suffix for the custom synapse model template name'
            ),
            
            # === 3. AXON ORGANIZATION ===
            'axon_organization': ParameterDefinition(
                default_value='focused',
                description='Axonal arbor organization: focused (narrow) or diffused (wide)',
                constraints={'allowed_values': ['focused', 'diffused']}
            ),
            'axon_radius': ParameterDefinition(
                default_value=0.5,
                description='Mean radius of axonal arbor domain (mm or relative units)',
                constraints={'min': 0.01},
                optimizable=True,
                optimization_range=[0.1, 2.0]
            ),
            'mask_type': ParameterDefinition(
                default_value='circular',
                description='Spatial mask type for 2D/3D connections',
                constraints={'allowed_values': ['circular', 'spherical', 'rectangular', 'box']}
            ),
            
            # === 4. PERCENTAGE OF PROJECTION NEURONS ===
            'projection_percentage': ParameterDefinition(
                default_value=100.0,
                description='Percentage of source neurons that project to target [0-100]%',
                constraints={'min': 0.0, 'max': 100.0},
                optimizable=True,
                optimization_range=[10.0, 100.0]
            ),
            
            # === 5. BOUTON NUMBER ===
            'bouton_number': ParameterDefinition(
                default_value=50,
                description='Mean number of axonal boutons per neuron',
                constraints={'min': 1},
                optimizable=True,
                optimization_range=[10, 200]
            ),
            
            # === 6. RECEPTOR LOCATION TO SOMA ===
            'receptor_location': ParameterDefinition(
                default_value=None,
                description='Mean distance to soma as proportion of dendrite extent [0-1]. Use None to disable weight attenuation. Receptor location type is determined based on value proximal: 0.0-0.33, medial: 0.33-0.67, distal: 0.67-1.0',
                constraints={'min': 0.0, 'max': 1.0},
                optimizable=True,
                optimization_range=[0.0, 1.0]
            ),
            
            # === 7. REDUNDANCY ===
            'redundancy': ParameterDefinition(
                default_value=1.0,
                description='Mean number of contacts per axon-dendrite pair [1-ν]',
                constraints={'min': 1.0},
                optimizable=True,
                optimization_range=[1.0, 10.0]
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
        
        # OPTIONAL: Input ports - data the node receives
        inputs={
            # Source population inputs
            'source_nest_population': PortDefinition(
                type=PortType.OBJECT,
                description='Source NEST population object',
                optional=True
            ),
            'source_population_metadata': PortDefinition(
                type=PortType.DICT,
                description='Source population metadata and properties',
                optional=True
            ),
            'source_neuron_properties': PortDefinition(
                type=PortType.DICT,
                description='Source neuron biological properties (including PSP data)',
                optional=True
            ),
            
            # Target population inputs
            'target_nest_population': PortDefinition(
                type=PortType.OBJECT,
                description='Target NEST population object',
                optional=True
            ),
            'target_population_metadata': PortDefinition(
                type=PortType.DICT,
                description='Target population metadata and properties',
                optional=True
            ),
            'target_neuron_properties': PortDefinition(
                type=PortType.DICT,
                description='Target neuron biological properties',
                optional=True
            ),
            
            # Optional parameter overrides
            'connection_parameter_overrides': PortDefinition(
                type=PortType.DICT,
                description='Dictionary of connection parameter overrides',
                optional=True
            ),
        },
        
        # OPTIONAL: Output ports - data the node produces
        outputs={
            # NEST execution outputs
            'nest_connections': PortDefinition(
                type=PortType.OBJECT,
                description='Created NEST connection object',
                optional=True
            ),
            'custom_synapse_model': PortDefinition(
                type=PortType.STR,
                description='Name of the created custom synapse model',
                optional=True
            ),
            
            # Connection metadata
            'connection_summary': PortDefinition(
                type=PortType.DICT,
                description='Summary of created connections and parameters'
            ),
            
            # Script generation outputs
            'python_script': PortDefinition(
                type=PortType.STR,
                description='Generated Python script for connection creation',
                optional=True
            ),
            'notebook_cell': PortDefinition(
                type=PortType.STR,
                description='Generated Jupyter notebook cell',
                optional=True
            ),
        },
        
        # OPTIONAL: Method definitions for documentation and process steps
        methods={
            'validate_parameters': MethodDefinition(
                description='Validate connection parameters and inputs',
                inputs=['connection_parameter_overrides'],
                outputs=['validated_params']
            ),
            
            'create_custom_synapse': MethodDefinition(
                description='Create custom synapse model with parameter overrides',
                inputs=['validated_params'],
                outputs=['custom_synapse_model']
            ),
            
            'create_nest_connections': MethodDefinition(
                description='Create NEST connections using validated parameters',
                inputs=['validated_params', 'custom_synapse_model'],
                outputs=['nest_connections', 'connection_summary']
            ),
            

            
            'generate_python_script': MethodDefinition(
                description='Generate Python script for connection creation',
                inputs=['validated_params'],
                outputs=['python_script']
            ),
            
            'generate_notebook_cell': MethodDefinition(
                description='Generate Jupyter notebook cell for connection creation',
                inputs=['validated_params'],
                outputs=['notebook_cell']
            ),
        }
    )
    
    def __init__(self, name: str):
        """Initialize the ConnectionBuilderNode."""
        super().__init__(name)
        
        # Node state
        self._validated_parameters = None

        self._created_connections = None
        self._connection_stats = None
        
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
        
        self.add_process_step(
            'create_custom_synapse',
            self.create_custom_synapse,
            method_key='create_custom_synapse'
        )
        
        self.add_process_step(
            'create_nest_connections',
            self.create_nest_connections,
            method_key='create_nest_connections'
        )
        

        
        self.add_process_step(
            'generate_python_script',
            self.generate_python_script,
            method_key='generate_python_script'
        )
        
        self.add_process_step(
            'generate_notebook_cell',
            self.generate_notebook_cell,
            method_key='generate_notebook_cell'
        )
    
    # ========================================================================
    # VALIDATION METHOD
    # ========================================================================
    
    def validate_parameters(self, connection_parameter_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate connection parameters and prepare for creation.
        
        Args:
            connection_parameter_overrides: Optional parameter overrides
            
        Returns:
            Dictionary containing validation results and processed parameters
        """
        print(f"[{self.name}] Validating connection parameters...")
        
        errors = []
        warnings = []
        
        # Start with current parameters
        validated_params = self._parameters.copy()
        
        # Apply parameter overrides
        if connection_parameter_overrides:
            for key, value in connection_parameter_overrides.items():
                if key in validated_params:
                    validated_params[key] = value
                    print(f"[{self.name}] Override: {key} = {value}")
                else:
                    warnings.append(f"Unknown parameter override: {key}")
        
        # Ensure 'rule' is not in connection_dict (it should come from connection_rule parameter)
        if 'rule' in validated_params['connection_dict']:
            warnings.append("'rule' found in connection_dict parameter - removing it. Use 'connection_rule' parameter instead.")
            validated_params['connection_dict'] = validated_params['connection_dict'].copy()
            del validated_params['connection_dict']['rule']
        
        # Validate synapse model
        synapse_model = validated_params['synapse_model']
        try:
            available_synapses = nest.synapse_models
            if synapse_model not in available_synapses:
                errors.append(f"Synapse model '{synapse_model}' not available in NEST. Available models: {available_synapses}")
        except:
            warnings.append("NEST not available - cannot validate synapse model")
        
        # Validate connection rule parameters
        connection_rule = validated_params['connection_rule']
        if connection_rule == 'pairwise_bernoulli':
            prob = validated_params['connection_dict'].get('p', 0.1)
            if not (0.0 <= prob <= 1.0):
                errors.append(f"Connection probability {prob} must be between 0.0 and 1.0")
        
        elif connection_rule in ['fixed_indegree', 'fixed_outdegree']:
            degree = validated_params['connection_dict'].get('indegree' if connection_rule == 'fixed_indegree' else 'outdegree', 10)
            if degree < 1:
                errors.append(f"Fixed degree {degree} must be at least 1")
        
        # Validate biological parameters
        projection_pct = validated_params['projection_percentage']
        if not (0.0 <= projection_pct <= 100.0):
            errors.append(f"Projection percentage {projection_pct} must be between 0 and 100")
        
        receptor_loc = validated_params['receptor_location']
        if receptor_loc is not None and not (0.0 <= receptor_loc <= 1.0):
            errors.append(f"Receptor location {receptor_loc} must be between 0.0 and 1.0")
        
        # Receptor location type is now determined automatically based on value
        # proximal: 0.0-0.33, medial: 0.33-0.67, distal: 0.67-1.0
        

        
        # Report validation results
        if errors:
            error_msg = f"[{self.name}] Validation failed: " + "; ".join(errors)
            print(error_msg)
            raise ValueError(error_msg)
        
        if warnings:
            for warning in warnings:
                print(f"[{self.name}] Warning: {warning}")
        
        self._validated_parameters = validated_params
        
        print(f"[{self.name}] Parameter validation completed successfully")
        return {
            'success': True,
            'validated_params': validated_params,
            'warnings': warnings
        }
    
    # ========================================================================
    # SYNAPSE MODEL CREATION
    # ========================================================================
    
    def create_custom_synapse(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create custom synapse model with parameter overrides.
        
        Args:
            validated_params: Validated parameters from validation step
            
        Returns:
            Dictionary with custom synapse model name
        """
        execution_mode = validated_params.get('execution_mode', 'both')
        
        # Check if synapse creation is requested
        if execution_mode not in ['execute', 'both']:
            print(f"[{self.name}] Skipping synapse creation (execution_mode: {execution_mode})")
            return {'custom_synapse_model': None}
        
        print(f"[{self.name}] Creating custom synapse model...")
        
        base_synapse = validated_params['synapse_model']
        template_suffix = validated_params['template_suffix']
        custom_name = f"{base_synapse}{template_suffix}_{self.name.replace(' ', '_')}"
        
        # Extract custom synapse parameters from synapse_dict (excluding standard keys)
        #standard_keys = {'weight', 'delay'}
        #synapse_params = {k: v for k, v in validated_params['synapse_dict'].items() if k not in standard_keys}
        
        try:
            # Step 1: Copy the base synapse model
            print(f"[{self.name}] Copying synapse model: {base_synapse} -> {custom_name}")
            nest.CopyModel(base_synapse, custom_name)
            
            # Store the custom synapse name for later use
            self._custom_synapse_name = custom_name
            print(f"[{self.name}] Custom synapse model '{custom_name}' created successfully")
            
            return {'custom_synapse_model': custom_name}
            
        except Exception as e:
            error_msg = f"Failed to create custom synapse model: {e}"
            print(f"[{self.name}] Error: {error_msg}")
            raise RuntimeError(error_msg)
    
    # ========================================================================
    # NEST CONNECTION CREATION
    # ========================================================================
    
    def create_nest_connections(self, validated_params: Dict[str, Any], 
                              custom_synapse_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Create NEST connections using validated parameters.
        
        Args:
            validated_params: Validated parameters from validation step
            custom_synapse_model: Name of custom synapse model
            
        Returns:
            Dictionary with connection results and summary
        """
        execution_mode = validated_params.get('execution_mode', 'both')
        
        # Check if connection creation is requested
        if execution_mode not in ['execute', 'both']:
            print(f"[{self.name}] Skipping connection creation (execution_mode: {execution_mode})")
            return {'nest_connections': None, 'connection_summary': {}}
        
        print(f"[{self.name}] Creating NEST connections...")
                
        # Get source and target populations from input ports
        source_pop = self._input_ports['source_nest_population'].value if 'source_nest_population' in self._input_ports else None
        target_pop = self._input_ports['target_nest_population'].value if 'target_nest_population' in self._input_ports else None
        
        if source_pop is None or target_pop is None:
            print(f"[{self.name}] Warning: Source or target population not available")
            return {'nest_connections': None, 'connection_summary': {}}
        
        try:
            # Build connection dictionary (cdict)
            cdict = self._build_connection_dict(validated_params)
            
            # Build synapse dictionary (sdict) - may return single dict or list of dicts
            sdict_result = self._build_synapse_dict(validated_params, custom_synapse_model)
            
            print(f"[{self.name}] Connection dict: {cdict}")
            
            # Handle single synapse dict or multiple synapse dicts
            if isinstance(sdict_result, list):
                print(f"[{self.name}] Multiple synapse dicts for multisynapse model: {len(sdict_result)} receptors")
                for i, sdict in enumerate(sdict_result):
                    print(f"[{self.name}] Synapse dict {i+1}: {sdict}")
                    print(f"[{self.name}] Connecting {len(source_pop)} source neurons to {len(target_pop)} target neurons (receptor {i+1})")
                    nest.Connect(source_pop, target_pop, cdict, sdict)
            else:
                print(f"[{self.name}] Single synapse dict: {sdict_result}")
                print(f"[{self.name}] Connecting {len(source_pop)} source neurons to {len(target_pop)} target neurons")
                nest.Connect(source_pop, target_pop, cdict, sdict_result)
            
            # Get connection information
            connections = nest.GetConnections(source_pop, target_pop)
            num_connections = len(connections)
            
            # Calculate connection statistics
            connection_summary = {
                'source_population_size': len(source_pop),
                'target_population_size': len(target_pop),
                'total_connections': num_connections,
                'connection_density': num_connections / (len(source_pop) * len(target_pop)),
                'connection_rule': validated_params['connection_rule'],
                'synapse_model': custom_synapse_model or validated_params['synapse_dict']['synapse_model'],
                'creation_timestamp': self._get_timestamp()
            }
            
            self._created_connections = connections
            
            print(f"[{self.name}] Successfully created {num_connections} connections")
            print(f"[{self.name}] Connection density: {connection_summary['connection_density']:.4f}")
            
            return {
                'nest_connections': connections,
                'connection_summary': connection_summary
            }
            
        except Exception as e:
            error_msg = f"Failed to create NEST connections: {e}"
            print(f"[{self.name}] Error: {error_msg}")
            raise RuntimeError(error_msg)
    
    def _build_connection_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Build NEST connection dictionary (cdict) based on connection rule."""
        
        connection_rule = params['connection_rule']
        
        # 1. Start with a copy of the user's connection_dict to preserve custom keys
        cdict = params['connection_dict'].copy()
        cdict['rule'] = connection_rule
        
        # 2. Add rule-specific default parameters if not provided by user
        if connection_rule == 'pairwise_bernoulli':
            if 'p' not in cdict:
                cdict['p'] = 0.1
                
        elif connection_rule == 'fixed_indegree':
            if 'indegree' not in cdict:
                # Estimate indegree from biological parameters
                estimated_indegree = self._estimate_indegree(params)
                cdict['indegree'] = estimated_indegree
                print(f"[{self.name}] Using biological parameter estimation for indegree: {estimated_indegree}")
            else:
                print(f"[{self.name}] Using user-defined indegree: {cdict['indegree']}")
            if 'allow_autapses' not in cdict:
                cdict['allow_autapses'] = False
            if 'allow_multapses' not in cdict:
                cdict['allow_multapses'] = False
                
        elif connection_rule == 'fixed_outdegree':
            if 'outdegree' not in cdict:
                # Estimate outdegree from biological parameters
                estimated_outdegree = self._estimate_outdegree(params)
                cdict['outdegree'] = estimated_outdegree
                print(f"[{self.name}] Using biological parameter estimation for outdegree: {estimated_outdegree}")
            else:
                print(f"[{self.name}] Using user-defined outdegree: {cdict['outdegree']}")
            if 'allow_autapses' not in cdict:
                cdict['allow_autapses'] = False
            if 'allow_multapses' not in cdict:
                cdict['allow_multapses'] = False
        
        # 3. Handle spatial mask - check if user provided mask directly
        user_provided_mask = 'mask' in cdict
        
        if not user_provided_mask:
            # 4. Detect 2D/3D organization from population positions
            spatial_dimensions = self._detect_spatial_dimensions()
            
            # 5. Use axon_radius to create mask if not provided by user
            if params['axon_radius'] > 0:
                mask_type = params['mask_type']
                radius = params['axon_radius']
                
                # Choose appropriate mask based on spatial dimensions
                if spatial_dimensions == '2D':
                    if mask_type in ['circular', 'spherical']:
                        cdict['mask'] = {'circular': {'radius': radius}}
                        print(f"[{self.name}] Created circular mask with radius {radius} based on axon_radius")
                    elif mask_type in ['rectangular', 'box']:
                        cdict['mask'] = {'rectangular': {'lower_left': [-radius, -radius], 'upper_right': [radius, radius]}}
                        print(f"[{self.name}] Created rectangular mask with size {2*radius}x{2*radius} based on axon_radius")
                else:  # 3D
                    if mask_type in ['circular', 'spherical']:
                        cdict['mask'] = {'spherical': {'radius': radius}}
                        print(f"[{self.name}] Created spherical mask with radius {radius} based on axon_radius")
                    elif mask_type in ['rectangular', 'box']:
                        cdict['mask'] = {'box': {'lower_left': [-radius, -radius, -radius], 'upper_right': [radius, radius, radius]}}
                        print(f"[{self.name}] Created box mask with size {2*radius}x{2*radius}x{2*radius} based on axon_radius")
        else:
            print(f"[{self.name}] Using user-provided mask in connection_dict")
        
        return cdict
    
    def _build_synapse_dict(self, params: Dict[str, Any], custom_synapse: Optional[str]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Build NEST synapse dictionary (sdict).
        
        Returns either a single synapse dict or a list of synapse dicts
        for multisynapse models with multiple receptor types.
        """
        
        # 5. Add all elements from synapse_dict parameter (not just weight and delay)
        synapse_dict = params['synapse_dict'].copy()
        
        # Check if weight is user-defined or needs biological estimation
        user_defined_weight = 'weight' in synapse_dict
        
        if user_defined_weight:
            print(f"[{self.name}] Using user-defined weight from synapse_dict")
            return self._build_single_synapse_dict(params, custom_synapse, synapse_dict)
        else:
            print(f"[{self.name}] Weight not defined by user, estimating from biological parameters")
            return self._build_biological_synapse_dict(params, custom_synapse, synapse_dict)
    
    def _build_single_synapse_dict(self, params: Dict[str, Any], custom_synapse: Optional[str], synapse_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build single synapse dict with user-defined weight."""
        
        sdict = {
            'synapse_model': custom_synapse or params['synapse_model']
        }
        
        # Get base weight and delay from synapse_dict
        base_weight = synapse_dict.get('weight', 1.0)
        base_delay = synapse_dict.get('delay', 1.0)
        
        # Calculate effective weight with biological parameters
        effective_weight = self._calculate_effective_weight(params, base_weight)
        
        # 6. Check weight sign based on source population cell_class
        effective_weight = self._apply_weight_sign(effective_weight)
        
        # Set weight and delay
        sdict['weight'] = effective_weight
        sdict['delay'] = base_delay
        
        # Add all other parameters from synapse_dict (receptor_type, custom parameters, etc.)
        for key, value in synapse_dict.items():
            if key not in ['weight', 'delay']:  # Don't overwrite weight and delay
                sdict[key] = value
        
        return sdict
    
    def _build_biological_synapse_dict(self, params: Dict[str, Any], custom_synapse: Optional[str], synapse_dict: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Build synapse dict(s) with biologically estimated weights from PSP amplitudes."""
        
        # Get source population metadata to determine excitatory/inhibitory
        source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
        target_metadata = self._input_ports['target_population_metadata'].value if 'target_population_metadata' in self._input_ports else None
        
        if not source_metadata or not target_metadata:
            print(f"[{self.name}] Missing population metadata, using default weight 1.0")
            return self._build_default_synapse_dict(params, custom_synapse, synapse_dict)
        
        # Determine connection type (excitatory/inhibitory)
        cell_class = self._get_cell_class(source_metadata)
        
        # Check if target neuron is multisynapse model
        is_multisynapse = self._is_multisynapse_model(target_metadata)
        
        # Get PSP amplitudes from target neuron
        psp_amplitudes = self._get_psp_amplitudes(target_metadata)
        
        if not psp_amplitudes:
            print(f"[{self.name}] No PSP amplitudes found in target metadata, using default weight 1.0")
            return self._build_default_synapse_dict(params, custom_synapse, synapse_dict)
        
        # Get redundancy for weight adjustment
        redundancy = params.get('redundancy', 1.0)
        
        # Build synapse dict(s) based on connection type and model type
        if cell_class == 'excitatory':
            return self._build_excitatory_synapses(params, custom_synapse, synapse_dict, psp_amplitudes, redundancy, is_multisynapse)
        elif cell_class == 'inhibitory':
            return self._build_inhibitory_synapses(params, custom_synapse, synapse_dict, psp_amplitudes, redundancy, is_multisynapse)
        else:
            print(f"[{self.name}] Unknown cell_class '{cell_class}', using default weight 1.0")
            return self._build_default_synapse_dict(params, custom_synapse, synapse_dict)
    
    def _build_default_synapse_dict(self, params: Dict[str, Any], custom_synapse: Optional[str], synapse_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build default synapse dict when biological estimation fails."""
        
        sdict = {
            'synapse_model': custom_synapse or params['synapse_model'],
            'weight': 1.0,
            'delay': synapse_dict.get('delay', 1.0)
        }
        
        # Add all other parameters from synapse_dict
        for key, value in synapse_dict.items():
            if key not in ['weight', 'delay']:
                sdict[key] = value
        
        return sdict
    
    def _get_cell_class(self, source_metadata: Dict[str, Any]) -> str:
        """Extract cell_class from source population metadata."""
        try:
            return source_metadata['biological_properties']['identification']['cell_class']
        except KeyError:
            return 'unknown'
    
    def _is_multisynapse_model(self, target_metadata: Dict[str, Any]) -> bool:
        """Check if target neuron model is multisynapse type."""
        model_name = target_metadata.get('model_name', '')
        return 'multisynapse' in model_name.lower()
    
    def _get_psp_amplitudes(self, target_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract PSP amplitudes from target neuron metadata."""
        try:
            return target_metadata['biological_properties']['signaling']['psp_amplitudes']
        except KeyError:
            return {}
    
    def _build_excitatory_synapses(self, params: Dict[str, Any], custom_synapse: Optional[str], 
                                 synapse_dict: Dict[str, Any], psp_amplitudes: Dict[str, float], 
                                 redundancy: float, is_multisynapse: bool) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Build excitatory synapse dict(s) with PSP-based weights."""
        
        # Receptor type mapping
        rec_type = {'AMPA': 1, 'NMDA': 2, 'GABAA': 3, 'GABAB': 4}
        
        # Get excitatory receptor PSPs (AMPA, NMDA)
        excitatory_psps = {k: v for k, v in psp_amplitudes.items() if k in ['AMPA', 'NMDA']}
        
        if not excitatory_psps:
            print(f"[{self.name}] No excitatory PSP amplitudes (AMPA, NMDA) found")
            return self._build_default_synapse_dict(params, custom_synapse, synapse_dict)
        
        print(f"[{self.name}] Excitatory connection - PSP amplitudes: {excitatory_psps}")
        print(f"[{self.name}] Target model multisynapse: {is_multisynapse}")
        print(f"[{self.name}] Redundancy factor: {redundancy}")
        
        if is_multisynapse:
            # Create separate connection for each receptor type
            synapse_dicts = []
            for receptor, psp_amplitude in excitatory_psps.items():
                weight = psp_amplitude * redundancy
                
                sdict = {
                    'synapse_model': custom_synapse or params['synapse_model'],
                    'weight': weight,
                    'delay': synapse_dict.get('delay', 1.0),
                    'receptor_type': rec_type[receptor]
                }
                
                # Add all other parameters from synapse_dict
                for key, value in synapse_dict.items():
                    if key not in ['weight', 'delay', 'receptor_type']:
                        sdict[key] = value
                
                synapse_dicts.append(sdict)
                print(f"[{self.name}] Created {receptor} synapse: weight={weight:.3f}, receptor_type={rec_type[receptor]}")
            
            return synapse_dicts
        else:
            # Single connection with first PSP found
            first_receptor = list(excitatory_psps.keys())[0]
            psp_amplitude = excitatory_psps[first_receptor]
            weight = psp_amplitude * redundancy
            
            sdict = {
                'synapse_model': custom_synapse or params['synapse_model'],
                'weight': weight,
                'delay': synapse_dict.get('delay', 1.0)
            }
            
            # Add all other parameters from synapse_dict
            for key, value in synapse_dict.items():
                if key not in ['weight', 'delay']:
                    sdict[key] = value
            
            print(f"[{self.name}] Created single excitatory synapse using {first_receptor}: weight={weight:.3f}")
            return sdict
    
    def _build_inhibitory_synapses(self, params: Dict[str, Any], custom_synapse: Optional[str], 
                                 synapse_dict: Dict[str, Any], psp_amplitudes: Dict[str, float], 
                                 redundancy: float, is_multisynapse: bool) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Build inhibitory synapse dict(s) with PSP-based weights."""
        
        # Receptor type mapping
        rec_type = {'AMPA': 1, 'NMDA': 2, 'GABAA': 3, 'GABAB': 4}
        
        # Get inhibitory receptor PSPs (GABAA, GABAB)
        inhibitory_psps = {k: v for k, v in psp_amplitudes.items() if k in ['GABAA', 'GABAB']}
        
        if not inhibitory_psps:
            print(f"[{self.name}] No inhibitory PSP amplitudes (GABAA, GABAB) found")
            return self._build_default_synapse_dict(params, custom_synapse, synapse_dict)
        
        print(f"[{self.name}] Inhibitory connection - PSP amplitudes: {inhibitory_psps}")
        print(f"[{self.name}] Target model multisynapse: {is_multisynapse}")
        print(f"[{self.name}] Redundancy factor: {redundancy}")
        
        if is_multisynapse:
            # Create separate connection for each receptor type
            synapse_dicts = []
            for receptor, psp_amplitude in inhibitory_psps.items():
                # For inhibitory connections, PSP amplitudes should be negative
                weight = -abs(psp_amplitude * redundancy)
                
                sdict = {
                    'synapse_model': custom_synapse or params['synapse_model'],
                    'weight': weight,
                    'delay': synapse_dict.get('delay', 1.0),
                    'receptor_type': rec_type[receptor]
                }
                
                # Add all other parameters from synapse_dict
                for key, value in synapse_dict.items():
                    if key not in ['weight', 'delay', 'receptor_type']:
                        sdict[key] = value
                
                synapse_dicts.append(sdict)
                print(f"[{self.name}] Created {receptor} synapse: weight={weight:.3f}, receptor_type={rec_type[receptor]}")
            
            return synapse_dicts
        else:
            # Single connection with first PSP found
            first_receptor = list(inhibitory_psps.keys())[0]
            psp_amplitude = inhibitory_psps[first_receptor]
            # For inhibitory connections, PSP amplitudes should be negative
            weight = -abs(psp_amplitude * redundancy)
            
            sdict = {
                'synapse_model': custom_synapse or params['synapse_model'],
                'weight': weight,
                'delay': synapse_dict.get('delay', 1.0)
            }
            
            # Add all other parameters from synapse_dict
            for key, value in synapse_dict.items():
                if key not in ['weight', 'delay']:
                    sdict[key] = value
            
            print(f"[{self.name}] Created single inhibitory synapse using {first_receptor}: weight={weight:.3f}")
            return sdict
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _detect_spatial_dimensions(self) -> str:
        """
        Detect 2D/3D organization from population positions.
        
        Returns:
            str: '2D' or '3D' based on population positions
        """
        # Check source population positions
        source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
        target_metadata = self._input_ports['target_population_metadata'].value if 'target_population_metadata' in self._input_ports else None
        
        # Check source population spatial dimensions
        if source_metadata and 'spatial_dimensions' in source_metadata:
            source_dims = source_metadata['spatial_dimensions']
            print(f"[{self.name}] Source population spatial dimensions: {source_dims}")
            if source_dims == '3D':
                return '3D'
        
        # Check target population spatial dimensions
        if target_metadata and 'spatial_dimensions' in target_metadata:
            target_dims = target_metadata['spatial_dimensions']
            print(f"[{self.name}] Target population spatial dimensions: {target_dims}")
            if target_dims == '3D':
                return '3D'
        
        # Check positions directly if metadata doesn't have spatial_dimensions
        for metadata, pop_name in [(source_metadata, 'source'), (target_metadata, 'target')]:
            if metadata and 'positions' in metadata:
                positions = metadata['positions']
                if hasattr(positions, 'shape') and len(positions.shape) >= 2:
                    if positions.shape[1] >= 3:  # Has 3 or more coordinates
                        print(f"[{self.name}] Detected 3D from {pop_name} population positions (shape: {positions.shape})")
                        return '3D'
                elif isinstance(positions, (list, tuple)) and len(positions) > 0:
                    if isinstance(positions[0], (list, tuple)) and len(positions[0]) >= 3:
                        print(f"[{self.name}] Detected 3D from {pop_name} population positions (list format)")
                        return '3D'
        
        # Default to 2D if no 3D indicators found
        print(f"[{self.name}] Defaulting to 2D spatial organization")
        return '2D'
    
    def _apply_weight_sign(self, weight) -> Union[float, Any]:
        """
        Apply correct sign to weight based on source population cell_class.
        
        Handles both numeric weights and NEST expressions like:
        - nest.math.exp(-nest.spatial.distance / 0.15)
        - nest.random.uniform(0.5, 2.0)
        
        Args:
            weight: Weight value (can be float or NEST expression)
            
        Returns:
            Union[float, Any]: Weight with correct sign applied
        """
        # Get source population metadata
        source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
        
        # Determine cell class
        cell_class = 'excitatory'  # Default
        if source_metadata and 'biological_properties' in source_metadata:
            bio_props = source_metadata['biological_properties']
            if 'identification' in bio_props and 'cell_class' in bio_props['identification']:
                cell_class = bio_props['identification']['cell_class']
        
        # Handle different weight types
        if isinstance(weight, (int, float)):
            # Numeric weight - apply sign directly
            if cell_class == 'excitatory':
                result_weight = abs(weight)
                print(f"[{self.name}] Excitatory connection: weight = +{result_weight:.3f}")
                return result_weight
            elif cell_class == 'inhibitory':
                result_weight = -abs(weight)
                print(f"[{self.name}] Inhibitory connection: weight = {result_weight:.3f}")
                return result_weight
            else:
                print(f"[{self.name}] Unknown cell_class '{cell_class}', using positive weight")
                return abs(weight)
        else:
            # Non-numeric weight (likely NEST expression)
            # For expressions, we'll need to wrap them to apply sign
            if cell_class == 'excitatory':
                print(f"[{self.name}] Excitatory connection: using expression weight as-is")
                return weight  # Keep expression unchanged for excitatory
            elif cell_class == 'inhibitory':
                print(f"[{self.name}] Inhibitory connection: will negate expression weight")
                # For inhibitory, we need to negate the expression
                # This will be handled by wrapping the expression
                # For now, store the sign information for later processing
                if hasattr(weight, '__class__') and 'nest' in str(weight.__class__):
                    # This is likely a NEST expression - mark it for negation
                    # We'll create a wrapper or use nest.math operations
                    try:
                        import nest
                        # Wrap the expression with negation
                        return -weight
                    except ImportError:
                        print(f"[{self.name}] NEST not available, returning expression unchanged")
                        return weight
                else:
                    # Unknown expression type, return as-is with warning
                    print(f"[{self.name}] Unknown expression type, returning unchanged")
                    return weight
            else:
                print(f"[{self.name}] Unknown cell_class '{cell_class}', using expression as-is")
                return weight
    
    def _calculate_effective_weight(self, params: Dict[str, Any], base_weight) -> Union[float, Any]:
        """
        Calculate effective weight with biological parameters.
        
        Currently returns the same base weight as placeholder.
        Future enhancements will include biological scaling factors.
        
        Args:
            params: Validated parameters dictionary
            base_weight: Base weight value from synapse_dict (can be numeric or NEST expression)
            
        Returns:
            Union[float, Any]: Effective weight (currently same as base_weight)
        """
        # Placeholder implementation - returns base weight unchanged
        # Future enhancements will include:
        # - Bouton number scaling
        # - Redundancy factors  
        # - Receptor location attenuation
        # - Other biological parameters
        
        # For now, just return the base weight as-is
        # This preserves both numeric values and NEST expressions
        return base_weight
    
    def _estimate_indegree(self, params: Dict[str, Any]) -> int:
        """
        Estimate indegree from biological parameters using the formula:
        Vx,y = (Px,y * Ny/Nx * alphax,y) / rhox,y
        
        Where:
        - Vx,y: in-degree (number of input synapses)
        - Px,y: proportion of neurons in source projecting to target (projection_percentage)
        - Ny: number of neurons in source population
        - Nx: number of neurons in target population  
        - alphax,y: average number of synapses per neuron (bouton_number)
        - rhox,y: redundancy factor (average contacts per neuron pair)
        
        Redundancy adjusts the in-degree based on available neurons within the spatial mask.
        It represents a trade-off between population size, spatial mask size, and redundancy.
        
        Args:
            params: Validated parameters dictionary
            
        Returns:
            int: Estimated indegree value adjusted for redundancy
        """
        try:
            # Get biological parameters
            projection_percentage = params.get('projection_percentage', 100.0) / 100.0  # Convert to proportion
            bouton_number = params.get('bouton_number', 50)  # alphax,y
            redundancy = params.get('redundancy', 1.0)  # rhox,y
            
            # Get population sizes from metadata
            source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
            target_metadata = self._input_ports['target_population_metadata'].value if 'target_population_metadata' in self._input_ports else None
            
            # Extract population sizes
            Ny = source_metadata.get('population_size', 500) if source_metadata else 500  # Source population size
            Nx = target_metadata.get('population_size', 500) if target_metadata else 500   # Target population size
            
            # Calculate base indegree: Vx,y = Px,y * Ny/Nx * alphax,y
            base_indegree = projection_percentage * (Ny / Nx) * bouton_number
            
            # Apply redundancy adjustment: Vx,y / rhox,y
            adjusted_indegree = base_indegree / redundancy
            adjusted_indegree = max(1, int(round(adjusted_indegree)))  # Ensure at least 1 and integer
            
            print(f"[{self.name}] Indegree calculation: P={projection_percentage:.2f}, Ny={Ny}, Nx={Nx}, alpha={bouton_number}, rho={redundancy:.2f}")
            print(f"[{self.name}] Base formula: {projection_percentage:.2f} * ({Ny}/{Nx}) * {bouton_number} = {base_indegree:.2f}")
            print(f"[{self.name}] Redundancy adjustment: {base_indegree:.2f} / {redundancy:.2f} = {adjusted_indegree}")
            print(f"[{self.name}] Final indegree (adjusted for redundancy): {adjusted_indegree}")
            
            return adjusted_indegree
            
        except Exception as e:
            print(f"[{self.name}] Error estimating indegree: {e}, using default value 10")
            return 10
    
    def _estimate_outdegree(self, params: Dict[str, Any]) -> int:
        """
        Estimate outdegree from biological parameters.
        
        For outdegree, we use alphax,y (bouton_number) directly as it represents
        the average number of synapses each neuron makes.
        
        Args:
            params: Validated parameters dictionary
            
        Returns:
            int: Estimated outdegree value
        """
        try:
            # Get bouton number (alphax,y)
            bouton_number = params.get('bouton_number', 50)  # alphax,y
            
            # For outdegree, we use bouton_number directly
            outdegree = max(1, int(round(bouton_number)))  # Ensure at least 1 and integer
            
            print(f"[{self.name}] Outdegree calculation: alpha (bouton_number) = {bouton_number}")
            print(f"[{self.name}] Using bouton_number directly as outdegree: {outdegree}")
            
            return outdegree
            
        except Exception as e:
            print(f"[{self.name}] Error estimating outdegree: {e}, using default value 10")
            return 10
    
    def _get_receptor_location_type(self, receptor_location: float) -> str:
        """Determine receptor location type based on numeric value."""
        if receptor_location is None:
            return 'none'
        elif receptor_location <= 0.33:
            return 'proximal'
        elif receptor_location <= 0.67:
            return 'medial'
        else:
            return 'distal'
    

    # ========================================================================
    # SCRIPT GENERATION METHODS
    # ========================================================================
    
    def generate_python_script(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Python script for connection creation.
        
        Args:
            validated_params: Validated parameters from validation step
            
        Returns:
            Dictionary containing generated Python script
        """
        execution_mode = validated_params.get('execution_mode', 'both')
        
        # Check if script generation is requested
        if execution_mode not in ['script', 'both']:
            print(f"[{self.name}] Skipping Python script generation (execution_mode: {execution_mode})")
            return {'python_script': None}
        
        print(f"[{self.name}] Generating Python script for connection creation...")
        
        # Determine connection type from source population metadata (mirror execution logic)
        source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
        connection_type = self._get_cell_class(source_metadata) if source_metadata else 'unknown'
        
        # Build the script
        script_lines = []
        
        # Header
        script_lines.extend([
            f"# Neural Connection: {validated_params['name']}",
            f"# Generated by NeuroWorkflow ConnectionBuilderNode",
            f"# Timestamp: {self._get_timestamp()}",
            "",
            "import nest",
            "import numpy as np",
            "",
            f"# Connection: {validated_params['name']}",
            f"# Type: {connection_type}",
            f"# Rule: {validated_params['connection_rule']}",
            ""
        ])
        
        # Synapse model creation
        synapse_model = validated_params['synapse_model']
        template_suffix = validated_params['template_suffix']
        custom_name = f"{synapse_model}{template_suffix}_{self.name.replace(' ', '_')}"
        
        # Extract custom synapse parameters from synapse_dict (excluding standard keys)
        standard_keys = {'weight', 'delay', 'receptor_type'}
        synapse_params = {k: v for k, v in validated_params['synapse_dict'].items() if k not in standard_keys}
        
        script_lines.extend([
            "# === SYNAPSE MODEL SETUP ===",
            f"base_synapse = '{synapse_model}'",
            f"custom_synapse = '{custom_name}'",
            "",
            "# Copy base synapse model",
            "nest.CopyModel(base_synapse, custom_synapse)",
            ""
        ])
        
        if synapse_params:
            script_lines.extend([
                "# Set custom synapse parameters",
                f"synapse_parameters = {synapse_params}",
                "nest.SetDefaults(custom_synapse, synapse_parameters)",
                ""
            ])
        
        # Connection dictionary
        cdict = self._build_connection_dict(validated_params)
        script_lines.extend([
            "# === CONNECTION DICTIONARY ===",
            f"cdict = {cdict}",
            ""
        ])
        
        # Synapse dictionary - mirror the execution logic
        synapse_dict = validated_params['synapse_dict'].copy()
        user_defined_weight = 'weight' in synapse_dict
        
        if user_defined_weight:
            script_lines.extend([
                "# === SYNAPSE DICTIONARY (User-defined weight) ===",
                f"# User provided weight in synapse_dict",
                ""
            ])
            
            base_weight = synapse_dict.get('weight', 1.0)
            base_delay = synapse_dict.get('delay', 1.0)
            
            sdict_base = {
                'synapse_model': custom_name,
                'weight': self._calculate_effective_weight(validated_params, base_weight),
                'delay': base_delay
            }
            
            # Add other parameters from synapse_dict
            for key, value in synapse_dict.items():
                if key not in ['weight', 'delay']:
                    sdict_base[key] = value
            
            script_lines.extend([
                f"sdict = {sdict_base}",
                ""
            ])
        else:
            # Get current metadata to show what would happen in execution
            target_metadata = self._input_ports['target_population_metadata'].value if 'target_population_metadata' in self._input_ports else None
            redundancy = validated_params.get('redundancy', 1.0)
            
            script_lines.extend([
                "# === SYNAPSE DICTIONARY (Biological weight estimation) ===",
                f"# Weight not defined by user, estimating from biological parameters",
                f"# Source connection type: {connection_type}",
                f"# Redundancy factor: {redundancy}",
                "",
                "# Biological weight estimation logic (mirrors execution):",
                "def estimate_biological_weights(source_metadata, target_metadata, redundancy):",
                "    \"\"\"Estimate weights from PSP amplitudes - mirrors ConnectionBuilderNode execution\"\"\"",
                "    ",
                "    # Extract cell_class from source population",
                "    try:",
                "        cell_class = source_metadata['biological_properties']['identification']['cell_class']",
                "    except (KeyError, TypeError):",
                "        cell_class = 'unknown'",
                "    ",
                "    # Extract PSP amplitudes from target neuron",
                "    try:",
                "        psp_amplitudes = target_metadata['biological_properties']['signaling']['psp_amplitudes']",
                "    except (KeyError, TypeError):",
                "        psp_amplitudes = {}",
                "    ",
                "    # Check if multisynapse model",
                "    model_name = target_metadata.get('model_name', '')",
                "    is_multisynapse = 'multisynapse' in model_name.lower()",
                "    ",
                "    # Receptor type mapping",
                "    rec_type = {'AMPA': 1, 'NMDA': 2, 'GABAA': 3, 'GABAB': 4}",
                "    ",
                "    if cell_class == 'excitatory':",
                "        # Use AMPA/NMDA receptors",
                "        exc_psps = {k: v for k, v in psp_amplitudes.items() if k in ['AMPA', 'NMDA']}",
                "        if is_multisynapse and exc_psps:",
                "            # Multiple connections for multisynapse",
                "            synapse_dicts = []",
                "            for receptor, psp_amp in exc_psps.items():",
                "                weight = psp_amp * redundancy",
                f"                sdict = {{'synapse_model': '{custom_name}', 'weight': weight, 'delay': {synapse_dict.get('delay', 1.0)}, 'receptor_type': rec_type[receptor]}}",
                "                synapse_dicts.append(sdict)",
                "            return synapse_dicts",
                "        elif exc_psps:",
                "            # Single connection",
                "            first_receptor = list(exc_psps.keys())[0]",
                "            weight = exc_psps[first_receptor] * redundancy",
                f"            return {{'synapse_model': '{custom_name}', 'weight': weight, 'delay': {synapse_dict.get('delay', 1.0)}}}",
                "    ",
                "    elif cell_class == 'inhibitory':",
                "        # Use GABAA/GABAB receptors",
                "        inh_psps = {k: v for k, v in psp_amplitudes.items() if k in ['GABAA', 'GABAB']}",
                "        if is_multisynapse and inh_psps:",
                "            # Multiple connections for multisynapse",
                "            synapse_dicts = []",
                "            for receptor, psp_amp in inh_psps.items():",
                "                weight = -abs(psp_amp * redundancy)  # Negative for inhibitory",
                f"                sdict = {{'synapse_model': '{custom_name}', 'weight': weight, 'delay': {synapse_dict.get('delay', 1.0)}, 'receptor_type': rec_type[receptor]}}",
                "                synapse_dicts.append(sdict)",
                "            return synapse_dicts",
                "        elif inh_psps:",
                "            # Single connection",
                "            first_receptor = list(inh_psps.keys())[0]",
                "            weight = -abs(inh_psps[first_receptor] * redundancy)  # Negative for inhibitory",
                f"            return {{'synapse_model': '{custom_name}', 'weight': weight, 'delay': {synapse_dict.get('delay', 1.0)}}}",
                "    ",
                "    # Fallback to default",
                f"    return {{'synapse_model': '{custom_name}', 'weight': 1.0, 'delay': {synapse_dict.get('delay', 1.0)}}}",
                "",
                "# Example usage (replace with your actual metadata):",
                "# source_metadata = your_source_population_metadata",
                "# target_metadata = your_target_population_metadata",
                f"# sdict_result = estimate_biological_weights(source_metadata, target_metadata, {redundancy})",
                "",
                "# For script generation, using simplified version:",
                f"sdict = {{",
                f"    'synapse_model': '{custom_name}',",
                f"    'weight': 1.0,  # Would be estimated from PSP amplitudes * {redundancy}",
                f"    'delay': {synapse_dict.get('delay', 1.0)}",
            ])
            
            # Add other parameters from synapse_dict
            for key, value in synapse_dict.items():
                if key not in ['weight', 'delay']:
                    script_lines.append(f"    '{key}': {repr(value)},")
            
            script_lines.extend([
                "}",
                ""
            ])
        
        # Get population variable names from metadata acronyms
        source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
        target_metadata = self._input_ports['target_population_metadata'].value if 'target_population_metadata' in self._input_ports else None
        
        source_var_name = source_metadata.get('acronym', 'source_population').lower() if source_metadata else 'source_population'
        target_var_name = target_metadata.get('acronym', 'target_population').lower() if target_metadata else 'target_population'
        
        # Connection creation
        script_lines.extend([
            "# === CREATE CONNECTIONS ===",
            f"# Note: Make sure you have defined your population variables:",
            f"# {source_var_name} = your_source_population",
            f"# {target_var_name} = your_target_population",
            "",
        ])
        
        if user_defined_weight:
            script_lines.extend([
                "# Create connections (single synapse dict - user-defined weight)",
                f"nest.Connect({source_var_name}, {target_var_name}, cdict, sdict)",
            ])
        else:
            script_lines.extend([
                "# Create connections (biological weight estimation - mirrors execution)",
                "# Get the estimated synapse dict(s)",
                "sdict_result = estimate_biological_weights(source_metadata, target_metadata, redundancy)",
                "",
                "# Handle single synapse dict or multiple synapse dicts (mirrors execution logic)",
                "if isinstance(sdict_result, list):",
                "    print(f'Multiple synapse dicts for multisynapse model: {len(sdict_result)} receptors')",
                "    for i, sdict in enumerate(sdict_result):",
                "        print(f'Synapse dict {i+1}: {sdict}')",
                "        print(f'Connecting with receptor_type {sdict.get(\"receptor_type\", \"N/A\")}')",
                f"        nest.Connect({source_var_name}, {target_var_name}, cdict, sdict)",
                "else:",
                "    print(f'Single synapse dict: {sdict_result}')",
                f"    nest.Connect({source_var_name}, {target_var_name}, cdict, sdict_result)",
            ])
        
        script_lines.extend([
            "",
            "# Get connection information",
            f"# connections = nest.GetConnections({source_var_name}, {target_var_name})",
            "# print(f'Created {{len(connections)}} connections')",
            ""
        ])
        
        # Biological properties as comments
        base_weight_for_comment = validated_params['synapse_dict'].get('weight', 1.0)
        script_lines.extend([
            "# === BIOLOGICAL PROPERTIES ===",
            f"# Axon organization: {validated_params['axon_organization']} (radius: {validated_params['axon_radius']} mm)",
            f"# Projection percentage: {validated_params['projection_percentage']}%",
            f"# Bouton number: {validated_params['bouton_number']}",
            f"# Receptor location: {self._get_receptor_location_type(validated_params['receptor_location'])} ({validated_params['receptor_location']})",
            f"# Redundancy: {validated_params['redundancy']}",
            f"# Base weight: {base_weight_for_comment} (effective weight calculated at runtime)",
            ""
        ])
        
        script = "\n".join(script_lines)
        
        print(f"[{self.name}] Python script generated ({len(script_lines)} lines)")
        return {'python_script': script}
    
    def generate_notebook_cell(self, validated_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Jupyter notebook cell for connection creation.
        
        Args:
            validated_params: Validated parameters from validation step
            
        Returns:
            Dictionary containing generated notebook cell
        """
        execution_mode = validated_params.get('execution_mode', 'both')
        script_format = validated_params.get('script_format', 'python')
        
        # Check if notebook cell generation is requested
        if execution_mode not in ['script', 'both']:
            print(f"[{self.name}] Skipping notebook cell generation (execution_mode: {execution_mode})")
            return {'notebook_cell': None}
        
        if script_format not in ['notebook', 'both']:
            print(f"[{self.name}] Skipping notebook cell generation (script_format: {script_format})")
            return {'notebook_cell': None}
        
        print(f"[{self.name}] Generating Jupyter notebook cell...")
        
        # Get the Python script and format it for Jupyter
        script_result = self.generate_python_script(validated_params)
        python_script = script_result.get('python_script', '')
        
        if not python_script:
            return {'notebook_cell': None}
        
        # Format for Jupyter notebook (add proper line endings)
        notebook_lines = python_script.split('\n')
        
        # Determine connection type from source population metadata (mirror execution logic)
        source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
        connection_type = self._get_cell_class(source_metadata) if source_metadata else 'unknown'
        
        # Add markdown header
        header_lines = [
            f"### Neural Connection: {validated_params['name']}\n",
            f"**Connection Type:** {connection_type}  \n",
            f"**Connection Rule:** {validated_params['connection_rule']}  \n",
            f"**Synapse Model:** {validated_params['synapse_model']}  \n",
            f"**Axon Organization:** {validated_params['axon_organization']} (radius: {validated_params['axon_radius']} mm)  \n",
            f"**Effective Weight:** {self._calculate_effective_weight(validated_params, validated_params['synapse_dict'].get('weight', 1.0)):.4f}  \n",
            "\n"
        ]
        
        # Combine header and code
        notebook_cell = ''.join(header_lines) + '\n```python\n' + '\n'.join(notebook_lines) + '\n```'
        
        print(f"[{self.name}] Notebook cell generated")
        return {'notebook_cell': notebook_cell}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """
        Get summary of the created connections.
        
        Returns:
            Dictionary with connection summary information
        """
        if self._validated_parameters is None:
            return {'error': 'Node not yet executed'}
        
        # Determine connection type from source population metadata (mirror execution logic)
        source_metadata = self._input_ports['source_population_metadata'].value if 'source_population_metadata' in self._input_ports else None
        connection_type = self._get_cell_class(source_metadata) if source_metadata else 'unknown'
        
        summary = {
            'connection_name': self._validated_parameters['name'],
            'connection_type': connection_type,
            'synapse_model': self._validated_parameters['synapse_model'],
            'custom_synapse_name': getattr(self, '_custom_synapse_name', f"{self._validated_parameters['synapse_model']}{self._validated_parameters['template_suffix']}_{self.name.replace(' ', '_')}"),
            'connection_rule': self._validated_parameters['connection_rule'],
            'execution_mode': self._validated_parameters['execution_mode'],
            'script_format': self._validated_parameters['script_format'],
            'biological_parameters': {
                'axon_organization': self._validated_parameters['axon_organization'],
                'axon_radius': self._validated_parameters['axon_radius'],
                'projection_percentage': self._validated_parameters['projection_percentage'],
                'bouton_number': self._validated_parameters['bouton_number'],
                'receptor_location': self._validated_parameters['receptor_location'],
                'redundancy': self._validated_parameters['redundancy']
            },
            'effective_weight': self._calculate_effective_weight(self._validated_parameters, self._validated_parameters['synapse_dict'].get('weight', 1.0)),
            'created_connections': len(self._created_connections) if self._created_connections else 0
        }
        
        return summary