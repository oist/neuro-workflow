"""
Stimulation Node for NeuroWorkflow Model Builder

This node creates various types of stimulation devices (Poisson generators, DC generators, etc.)
and connects them to specific neurons or regions within a population.
It operates in dual modalities:
1. Direct execution: Creates actual NEST stimulation objects and connections
2. Script generation: Generates Python code for standalone execution

Author: NeuroWorkflow Team
Date: 2025
Version: 1.0
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
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


class SNNbuilder_Stimulation(Node):
    """
    Stimulation Builder Node for Neural Model Stimulation.
    
    This node creates stimulation devices and connects them to target neurons
    within a population. It supports various stimulation types and targeting methods.
    
    Stimulation Types:
    - **poisson_generator**: Generates Poisson-distributed spike trains
    - **dc_generator**: Generates constant current injection
    
    Targeting Options:
    - **full_population**: Stimulate all neurons in the population
    - **define_volume_area**: Stimulate neurons within a volume area (sphere/circle) defined by center coordinates and radius
    
    Connection Parameters:
    - Synapse model and parameters (weight, delay)
    - Multiple stimulation devices can be created
    
    Outputs:
    - NEST stimulation objects and connections
    - Python script code for standalone execution
    - Stimulation metadata and connection summary
    """
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='stimulation_builder',
        description='Build stimulation devices and connect them to neural populations',
        
        parameters={
            # === STIMULATION CONFIGURATION ===
            'stimulation_type': ParameterDefinition(
                default_value='poisson_generator',
                description='Type of stimulation device',
                constraints={'allowed_values': ['poisson_generator', 'dc_generator']}
            ),
            
            'number_of_devices': ParameterDefinition(
                default_value=1,
                description='Number of stimulation devices to create',
                constraints={'min': 1, 'max': 100}
            ),
            
            # === POISSON GENERATOR PARAMETERS ===
            'poisson_parameters': ParameterDefinition(
                default_value={
                    'label': 'poisson_stim',
                    'start': 0.0,
                    'stop': 1000.0,
                    'rate': 100.0
                },
                description='Parameters for poisson_generator stimulation'
            ),
            
            # === DC GENERATOR PARAMETERS ===
            'dc_parameters': ParameterDefinition(
                default_value={
                    'label': 'dc_stim',
                    'start': 0.0,
                    'stop': 1000.0,
                    'amplitude': 100.0
                },
                description='Parameters for dc_generator stimulation'
            ),
            
            # === CONNECTION PARAMETERS ===
            'synapse_specification': ParameterDefinition(
                default_value={
                    'synapse_model': 'static_synapse',
                    'weight': 1.0,
                    'delay': 1.0
                },
                description='Synapse parameters for connecting stimulation to neurons'
            ),
            
            # === TARGET SPECIFICATION ===
            'target_identification': ParameterDefinition(
                default_value='full_population',
                description='Target identification method for stimulation',
                constraints={'allowed_values': ['full_population', 'define_volume_area']}
            ),
            
            'volume_area_specification': ParameterDefinition(
                default_value={
                    'center_coordinates': [0.0, 0.0, 0.0],
                    'radius': 0.5
                },
                description='Volume area specification with center coordinates and radius for spatial targeting'
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
            ),
            
            'template_suffix': ParameterDefinition(
                default_value='_custom',
                description='Suffix for custom model template names'
            )
        },
        
        inputs={
            # Required inputs from SNNbuilder_Population
            'nest_population': PortDefinition(
                type=PortType.OBJECT,
                description='NEST population object to stimulate'
            ),
            
            'population_data': PortDefinition(
                type=PortType.DICT,
                description='Population metadata including spatial information'
            ),

        },
        
        outputs={
            # NEST execution outputs
            'nest_stimulation_devices': PortDefinition(
                type=PortType.OBJECT,
                description='Created NEST stimulation device objects'
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
            'stimulation_summary': PortDefinition(
                type=PortType.DICT,
                description='Summary of created stimulation devices and connections'
            ),
            
            'target_neurons': PortDefinition(
                type=PortType.OBJECT,
                description='NEST NodeCollection of targeted neurons',
                optional=True
            )
        },
        
        methods={
            'validate_parameters': MethodDefinition(
                description='Validate stimulation and connection parameters',
                inputs=['nest_population', 'population_data'],
                outputs=['validated_parameters']
            ),
            
            'create_stimulation_devices': MethodDefinition(
                description='Create NEST stimulation devices',
                inputs=['validated_parameters'],
                outputs=['nest_stimulation_devices']
            ),
            
            'identify_target_neurons': MethodDefinition(
                description='Identify target neurons based on targeting specification',
                inputs=['validated_parameters'],
                outputs=['target_neurons']
            ),
            
            'create_connections': MethodDefinition(
                description='Connect stimulation devices to target neurons',
                inputs=['nest_stimulation_devices', 'target_neurons', 'validated_parameters'],
                outputs=[]
            ),
            
            'generate_python_script': MethodDefinition(
                description='Generate Python script code',
                inputs=['validated_parameters'],
                outputs=['python_script']
            ),
            
            'generate_notebook_cell': MethodDefinition(
                description='Generate Jupyter notebook cell',
                inputs=['validated_parameters'],
                outputs=['notebook_cell']
            ),
            
            'compile_summary': MethodDefinition(
                description='Compile stimulation summary and metadata',
                inputs=['validated_parameters', 'nest_stimulation_devices'],
                outputs=['stimulation_summary']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the Stimulation Builder Node."""
        super().__init__(name)
        
        # Internal state
        self._validated_parameters = {}
        self._created_devices = None
        self._target_neurons = None
        self._connections = None
        self._custom_synapse_name = None
        
        # Define process steps
        self._define_process_steps()
    
    def _define_process_steps(self) -> None:
        """Define the sequence of stimulation building steps."""
        
        # Always validate parameters first
        self.add_process_step(
            "validate_parameters",
            self.validate_parameters,
            method_key="validate_parameters"
        )
        
        # Create stimulation devices (will check execution_mode internally)
        self.add_process_step(
            "create_stimulation_devices",
            self.create_stimulation_devices,
            method_key="create_stimulation_devices"
        )
        
        # Identify target neurons
        self.add_process_step(
            "identify_target_neurons",
            self.identify_target_neurons,
            method_key="identify_target_neurons"
        )
        
        # Create connections
        self.add_process_step(
            "create_connections",
            self.create_connections,
            method_key="create_connections"
        )
        
        # Generate Python script (will check script_format internally)
        self.add_process_step(
            "generate_python_script",
            self.generate_python_script,
            method_key="generate_python_script"
        )
        
        # Generate notebook cell (will check script_format internally)
        self.add_process_step(
            "generate_notebook_cell",
            self.generate_notebook_cell,
            method_key="generate_notebook_cell"
        )
        
        # Compile summary
        self.add_process_step(
            "compile_summary",
            self.compile_summary,
            method_key="compile_summary"
        )
    
    # ========================================================================
    # VALIDATION METHODS
    # ========================================================================
    
    def validate_parameters(self, nest_population: Any, population_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Validate and merge all stimulation parameters.
        
        Args:
            nest_population: NEST population object
            population_data: Population metadata
            
        Returns:
            Dictionary containing validated parameters
        """
        print(f"[{self.name}] Validating stimulation parameters...")
        
        # Start with base parameters
        validated_params = self._parameters.copy()
        
        # Validate stimulation type parameters
        stim_type = validated_params['stimulation_type']
        if stim_type == 'poisson_generator':
            self._validate_poisson_parameters(validated_params['poisson_parameters'])
        elif stim_type == 'dc_generator':
            self._validate_dc_parameters(validated_params['dc_parameters'])
        
        # Validate target parameters
        self._validate_target_identification(validated_params['target_identification'])
        if validated_params['target_identification'] == 'define_volume_area':
            self._validate_volume_area_specification(validated_params['volume_area_specification'], population_data)
        
        # Validate synapse specification
        self._validate_synapse_specification(validated_params['synapse_specification'])
        
        # Store validated parameters
        self._validated_parameters = validated_params
        
        print(f"[{self.name}] Parameter validation completed successfully")
        return {'validated_parameters': validated_params}
    
    def _validate_poisson_parameters(self, params: Dict[str, Any]) -> None:
        """Validate Poisson generator parameters."""
        required_keys = ['label', 'start', 'stop', 'rate']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required Poisson parameter: {key}")
        
        if params['stop'] <= params['start']:
            raise ValueError("Poisson generator 'stop' must be greater than 'start'")
        
        if params['rate'] < 0:
            raise ValueError("Poisson generator 'rate' must be non-negative")
    
    def _validate_dc_parameters(self, params: Dict[str, Any]) -> None:
        """Validate DC generator parameters."""
        required_keys = ['label', 'start', 'stop', 'amplitude']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"Missing required DC parameter: {key}")
        
        if params['stop'] <= params['start']:
            raise ValueError("DC generator 'stop' must be greater than 'start'")
    
    def _validate_target_identification(self, target_identification: str) -> None:
        """Validate target identification parameter."""
        if target_identification not in ['full_population', 'define_volume_area']:
            raise ValueError(f"Invalid target_identification: {target_identification}. Must be 'full_population' or 'define_volume_area'")
    
    def _validate_volume_area_specification(self, volume_spec: Dict[str, Any], population_data: Dict[str, Any] = None) -> None:
        """Validate volume area specification parameters."""
        if 'center_coordinates' not in volume_spec:
            raise ValueError("volume_area_specification requires 'center_coordinates'")
        if 'radius' not in volume_spec:
            raise ValueError("volume_area_specification requires 'radius'")
        
        coords = volume_spec['center_coordinates']
        radius = volume_spec['radius']
        
        # Validate coordinates format
        if not isinstance(coords, (list, tuple)) or len(coords) not in [2, 3]:
            raise ValueError("center_coordinates must be a list/tuple of 2 or 3 numbers [x, y] or [x, y, z]")
        
        # Validate radius
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("radius must be a positive number")
        
        # Check spatial dimensions compatibility (only if population_data is available)
        if population_data is not None:
            spatial_dims = population_data.get('spatial_dimensions', '3D')
            
            if spatial_dims == '2D' and len(coords) != 2:
                raise ValueError("2D population requires 2D center_coordinates [x, y]")
            elif spatial_dims == '3D' and len(coords) != 3:
                raise ValueError("3D population requires 3D center_coordinates [x, y, z]")
    
    def _validate_synapse_specification(self, synapse_spec: Dict[str, Any]) -> None:
        """Validate synapse specification parameters."""
        required_keys = ['synapse_model', 'weight', 'delay']
        for key in required_keys:
            if key not in synapse_spec:
                raise ValueError(f"Missing required synapse parameter: {key}")
        
        if synapse_spec['delay'] <= 0:
            raise ValueError("Synapse 'delay' must be positive")
    
    # ========================================================================
    # NEST EXECUTION METHODS
    # ========================================================================
    
    def create_stimulation_devices(self, validated_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create NEST stimulation devices.
        
        Args:
            validated_parameters: Validated parameters
            
        Returns:
            Dictionary containing created devices and custom synapse model name
        """
        execution_mode = validated_parameters.get('execution_mode', 'both')
        
        if execution_mode in ['execute', 'both']:
            if not NEST_AVAILABLE:
                print(f"[{self.name}] NEST not available, skipping device creation")
                return {'nest_stimulation_devices': None}
            
            print(f"[{self.name}] Creating stimulation devices...")
            
            stim_type = validated_parameters['stimulation_type']
            num_devices = validated_parameters['number_of_devices']
            
            # Create custom synapse model
            synapse_spec = validated_parameters['synapse_specification']
            custom_synapse_name = self._create_custom_synapse_model(synapse_spec, validated_parameters['template_suffix'])
            
            # Create stimulation devices (all at once)
            if stim_type == 'poisson_generator':
                nest_devices = self._create_poisson_devices(validated_parameters['poisson_parameters'], num_devices)
            elif stim_type == 'dc_generator':
                nest_devices = self._create_dc_devices(validated_parameters['dc_parameters'], num_devices)
            
            print(f"[{self.name}] Created {num_devices} {stim_type} device(s)")
            
            self._created_devices = nest_devices
            self._custom_synapse_name = custom_synapse_name
            
            print(f"[{self.name}] Successfully created {num_devices} stimulation device(s)")
            return {'nest_stimulation_devices': nest_devices}
        
        else:
            print(f"[{self.name}] Skipping device creation (execution_mode: {execution_mode})")
            return {'nest_stimulation_devices': None}
    
    def _create_custom_synapse_model(self, synapse_spec: Dict[str, Any], suffix: str) -> str:
        """Create a custom synapse model for stimulation connections."""
        base_model = synapse_spec['synapse_model']
        custom_name = f"{base_model}{suffix}_{self.name.replace(' ', '_')}"
        
        try:
            # Copy the base synapse model
            nest.CopyModel(base_model, custom_name)
            print(f"[{self.name}] Custom synapse model '{custom_name}' created successfully")
            return custom_name
        except Exception as e:
            print(f"[{self.name}] Warning: Could not create custom synapse model: {e}")
            return base_model
    
    def _create_poisson_devices(self, params: Dict[str, Any], num_devices: int) -> Any:
        """Create multiple Poisson generator devices at once."""
        device_params = {
            'start': params['start'],
            'stop': params['stop'],
            'rate': params['rate']
        }
        
        devices = nest.Create('poisson_generator', num_devices, params=device_params)
        return devices
    
    def _create_dc_devices(self, params: Dict[str, Any], num_devices: int) -> Any:
        """Create multiple DC generator devices at once."""
        device_params = {
            'start': params['start'],
            'stop': params['stop'],
            'amplitude': params['amplitude']
        }
        
        devices = nest.Create('dc_generator', num_devices, params=device_params)
        return devices
    
    def identify_target_neurons(self, validated_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify target neurons based on targeting specification.
        
        Args:
            validated_parameters: Validated parameters
            
        Returns:
            Dictionary containing target neurons
        """
        execution_mode = validated_parameters.get('execution_mode', 'both')
        
        if execution_mode in ['execute', 'both']:
            if not NEST_AVAILABLE:
                print(f"[{self.name}] NEST not available, skipping target identification")
                return {'target_neurons': None}
            
            print(f"[{self.name}] Identifying target neurons...")
            
            nest_population = self._input_ports['nest_population'].value
            population_data = self._input_ports['population_data'].value
            target_identification = validated_parameters['target_identification']
            
            if target_identification == 'full_population':
                target_neurons = nest_population
                print(f"[{self.name}] Targeting full population ({len(nest_population)} neurons)")
            
            elif target_identification == 'define_volume_area':
                volume_spec = validated_parameters['volume_area_specification']
                target_neurons = self._get_volume_area_targets(nest_population, population_data, volume_spec)
                print(f"[{self.name}] Targeting volume area ({len(target_neurons)} neurons)")
            
            self._target_neurons = target_neurons
            return {'target_neurons': target_neurons}
        
        else:
            print(f"[{self.name}] Skipping target identification (execution_mode: {execution_mode})")
            return {'target_neurons': None}
    
    def _get_volume_area_targets(self, population: Any, population_data: Dict[str, Any], volume_spec: Dict[str, Any]) -> Any:
        """Get neurons within a volume area using efficient vectorized operations.
        
        This method works for both 2D and 3D coordinates automatically based on the
        dimensions of the position data and center coordinates.
        """
        center_coordinates = np.array(volume_spec['center_coordinates'])
        radius = volume_spec['radius']
        
        # Get population positions
        positions = population_data.get('positions')
        if positions is None:
            raise ValueError("Population has no spatial positions for spatial targeting")
        
        # Efficient vectorized distance calculation (avoids sqrt)
        # Calculate squared distances: d² = (positions - center)²
        # This works for both 2D and 3D coordinates automatically
        d_squared = ((positions - center_coordinates)**2).sum(axis=1)
        
        # Find neurons within radius using squared distance comparison
        target_indices = np.where(d_squared <= radius * radius)[0]
        
        # Convert to NEST NodeCollection
        if len(target_indices) > 0:
            first_id = population[0].global_id
            target_ids = [first_id + i for i in target_indices]
            return nest.NodeCollection(target_ids)
        else:
            raise ValueError(f"No neurons found within radius {radius} of center coordinates {center_coordinates}")
    
    def create_connections(self, nest_stimulation_devices: Any, target_neurons: Any, 
                         validated_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create connections between stimulation devices and target neurons.
        
        Args:
            nest_stimulation_devices: Created stimulation devices
            target_neurons: Target neurons
            validated_parameters: Validated parameters
            
        Returns:
            Dictionary containing created connections
        """
        execution_mode = validated_parameters.get('execution_mode', 'both')
        
        if execution_mode in ['execute', 'both'] and nest_stimulation_devices is not None and target_neurons is not None:
            print(f"[{self.name}] Creating connections...")
            
            synapse_spec = validated_parameters['synapse_specification']
            custom_synapse_name = self._custom_synapse_name or synapse_spec['synapse_model']
            
            # Prepare synapse dictionary
            sdict = {
                'synapse_model': custom_synapse_name,
                'weight': synapse_spec['weight'],
                'delay': synapse_spec['delay']
            }
            
            num_devices = validated_parameters['number_of_devices']
            
            # Efficient connection: no need to iterate over devices
            if num_devices == 1:
                # Single device to many neurons (all-to-all)
                nest.Connect(nest_stimulation_devices, target_neurons, syn_spec=sdict)
                print(f"[{self.name}] Connected 1 device to {len(target_neurons)} target neurons")
            else:
                # Multiple devices to neurons (one-to-one if same count, or all-to-all)
                if len(nest_stimulation_devices) == len(target_neurons):
                    # One-to-one connection
                    nest.Connect(nest_stimulation_devices, target_neurons, "one_to_one", syn_spec=sdict)
                    print(f"[{self.name}] Connected {num_devices} devices to {len(target_neurons)} neurons (one-to-one)")
                else:
                    # All-to-all connection
                    nest.Connect(nest_stimulation_devices, target_neurons, syn_spec=sdict)
                    print(f"[{self.name}] Connected {num_devices} devices to {len(target_neurons)} neurons (all-to-all)")
            
            # Get connection information for tracking
            connections = nest.GetConnections(nest_stimulation_devices, target_neurons)
            self._connections = connections
            total_connections = len(connections)
            
            print(f"[{self.name}] Successfully created {total_connections} connections")
            return {}
        
        else:
            print(f"[{self.name}] Skipping connection creation")
            return {}
    
    # ========================================================================
    # CODE GENERATION METHODS
    # ========================================================================
    
    def generate_python_script(self, validated_parameters: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate Python script for stimulation creation.
        
        Args:
            validated_parameters: Validated parameters (optional, uses stored if None)
            
        Returns:
            Dictionary containing generated Python script
        """
        
        # Use stored parameters if none provided
        if validated_parameters is None:
            validated_parameters = self.parameters
        script_format = validated_parameters.get('script_format', 'python')
        
        if script_format not in ['python', 'both']:
            print(f"[{self.name}] Skipping Python script generation (script_format: {script_format})")
            return {'python_script': ''}
        
        print(f"[{self.name}] Generating Python script...")
        
        # Get population variable name from metadata
        population_data = self._input_ports['population_data'].value
        population_var_name = population_data.get('acronym', 'population').lower() if population_data else 'population'
        
        script_lines = []
        
        # Header
        script_lines.extend([
            "# === STIMULATION SETUP ===",
            f"# Generated by {self.__class__.__name__}: {self.name}",
            f"# Stimulation type: {validated_parameters['stimulation_type']}",
            "",
            "import nest",
            "import numpy as np",
            ""
        ])
        
        # Parameters
        stim_type = validated_parameters['stimulation_type']
        num_devices = validated_parameters['number_of_devices']
        
        if stim_type == 'poisson_generator':
            params = validated_parameters['poisson_parameters']
            script_lines.extend([
                "# Poisson generator parameters",
                f"poisson_params = {json.dumps(params, indent=4)}",
                ""
            ])
        elif stim_type == 'dc_generator':
            params = validated_parameters['dc_parameters']
            script_lines.extend([
                "# DC generator parameters",
                f"dc_params = {json.dumps(params, indent=4)}",
                ""
            ])
        
        # Synapse parameters
        synapse_spec = validated_parameters['synapse_specification']
        custom_synapse_name = f"{synapse_spec['synapse_model']}{validated_parameters['template_suffix']}_{self.name.replace(' ', '_')}"
        
        script_lines.extend([
            "# Synapse specification",
            f"synapse_spec = {json.dumps(synapse_spec, indent=4)}",
            "",
            "# Create custom synapse model",
            f"custom_synapse_name = '{custom_synapse_name}'",
            f"nest.CopyModel('{synapse_spec['synapse_model']}', custom_synapse_name)",
            ""
        ])
        
        # Device creation
        script_lines.extend([
            f"# Create {num_devices} {stim_type} device(s)"
        ])
        
        if stim_type == 'poisson_generator':
            script_lines.append(f"devices = nest.Create('poisson_generator', {num_devices}, params=poisson_params)")
        elif stim_type == 'dc_generator':
            script_lines.append(f"devices = nest.Create('dc_generator', {num_devices}, params=dc_params)")
        
        script_lines.append("")
        
        # Target identification
        target_identification = validated_parameters['target_identification']
        
        script_lines.extend([
            "# Identify target neurons",
            f"# Note: Make sure you have defined your population variable:",
            f"# {population_var_name} = your_population",
            ""
        ])
        
        if target_identification == 'full_population':
            script_lines.extend([
                f"# Target: Full population",
                f"target_neurons = {population_var_name}"
            ])
        elif target_identification == 'define_volume_area':
            volume_spec = validated_parameters['volume_area_specification']
            coords = volume_spec['center_coordinates']
            radius = volume_spec['radius']
            script_lines.extend([
                f"# Target: Volume area (efficient vectorized approach - works for 2D/3D)",
                f"center_coordinates = np.array({coords})",
                f"target_radius = {radius}",
                f"",
                f"# Get population positions (assuming they're stored in population_data)",
                f"positions = population_data.get('positions')",
                f"if positions is None:",
                f"    raise ValueError('Population has no spatial positions for spatial targeting')",
                f"",
                f"# Efficient vectorized distance calculation (avoids sqrt, works for 2D/3D)",
                f"d_squared = ((positions - center_coordinates)**2).sum(axis=1)",
                f"target_indices = np.where(d_squared <= target_radius * target_radius)[0]",
                f"",
                f"# Convert to NEST NodeCollection",
                f"if len(target_indices) > 0:",
                f"    first_id = {population_var_name}[0].global_id",
                f"    target_ids = [first_id + i for i in target_indices]",
                f"    target_neurons = nest.NodeCollection(target_ids)",
                f"else:",
                f"    raise ValueError(f'No neurons found within radius {{target_radius}} of center {{center_coordinates}}')"
            ])
        
        script_lines.append("")
        
        # Connections
        num_devices = validated_parameters['number_of_devices']
        script_lines.extend([
            "# Create connections (efficient approach)",
            "sdict = {",
            f"    'synapse_model': custom_synapse_name,",
            f"    'weight': {synapse_spec['weight']},",
            f"    'delay': {synapse_spec['delay']}",
            "}",
            "",
            "# Efficient connection: no need to iterate over devices"
        ])
        
        if num_devices == 1:
            script_lines.extend([
                "# Single device to many neurons (all-to-all)",
                "nest.Connect(devices, target_neurons, syn_spec=sdict)"
            ])
        else:
            script_lines.extend([
                "# Multiple devices to neurons",
                "if len(devices) == len(target_neurons):",
                "    # One-to-one connection",
                "    nest.Connect(devices, target_neurons, 'one_to_one', syn_spec=sdict)",
                "else:",
                "    # All-to-all connection", 
                "    nest.Connect(devices, target_neurons, syn_spec=sdict)"
            ])
        
        script_lines.extend([
            "",
            f"print(f'Created {num_devices} stimulation devices')",
            "print(f'Connected to {{len(target_neurons)}} target neurons')"
        ])
        
        script = "\n".join(script_lines)
        
        print(f"[{self.name}] Python script generated ({len(script_lines)} lines)")
        return {'python_script': script}
    
    def generate_notebook_cell(self, validated_parameters: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate Jupyter notebook cell for stimulation creation.
        
        Args:
            validated_parameters: Validated parameters
            
        Returns:
            Dictionary containing generated notebook cell
        """
        script_format = validated_parameters.get('script_format', 'python')
        
        if script_format not in ['notebook', 'both']:
            print(f"[{self.name}] Skipping notebook cell generation (script_format: {script_format})")
            return {'notebook_cell': ''}
        
        print(f"[{self.name}] Generating Jupyter notebook cell...")
        
        # Get the Python script and format it for Jupyter
        script_result = self.generate_python_script(validated_parameters)
        python_script = script_result.get('python_script', '')
        
        if not python_script:
            return {'notebook_cell': ''}
        
        # Format for Jupyter notebook
        notebook_lines = python_script.split('\n')
        
        # Add markdown header
        stim_type = validated_parameters['stimulation_type']
        target_identification = validated_parameters['target_identification']
        num_devices = validated_parameters['number_of_devices']
        
        header_lines = [
            f"### Neural Stimulation: {validated_parameters.get('name', self.name)}\n",
            f"**Stimulation Type:** {stim_type}  \n",
            f"**Number of Devices:** {num_devices}  \n",
            f"**Target Identification:** {target_identification}  \n",
            f"**Synapse Model:** {validated_parameters['synapse_specification']['synapse_model']}  \n",
            "\n"
        ]
        
        # Combine header and code
        notebook_cell = ''.join(header_lines) + '\n```python\n' + '\n'.join(notebook_lines) + '\n```'
        
        print(f"[{self.name}] Notebook cell generated")
        return {'notebook_cell': notebook_cell}
    
    def compile_summary(self, validated_parameters: Dict[str, Any], 
                       nest_stimulation_devices: Any = None) -> Dict[str, Dict[str, Any]]:
        """
        Compile stimulation summary and metadata.
        
        Args:
            validated_parameters: Validated parameters
            nest_stimulation_devices: Created stimulation devices
            
        Returns:
            Dictionary containing stimulation summary
        """
        print(f"[{self.name}] Compiling stimulation summary...")
        
        # Get timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Basic summary
        summary = {
            'stimulation_name': self.name,
            'stimulation_type': validated_parameters['stimulation_type'],
            'number_of_devices': validated_parameters['number_of_devices'],
            'target_identification': validated_parameters['target_identification'],
            'synapse_model': self._custom_synapse_name or validated_parameters['synapse_specification']['synapse_model'],
            'creation_timestamp': timestamp
        }
        
        # Add execution results if available
        if nest_stimulation_devices is not None:
            summary['devices_created'] = True
            if isinstance(nest_stimulation_devices, list):
                summary['device_count'] = len(nest_stimulation_devices)
            else:
                summary['device_count'] = 1
        else:
            summary['devices_created'] = False
            summary['device_count'] = 0
        
        # Connection information from internal state
        if hasattr(self, '_connections') and self._connections is not None:
            summary['connections_created'] = True
            summary['total_connections'] = len(self._connections)
        else:
            summary['connections_created'] = False
            summary['total_connections'] = 0
        
        # Add parameter details
        if validated_parameters['stimulation_type'] == 'poisson_generator':
            summary['stimulation_parameters'] = validated_parameters['poisson_parameters']
        elif validated_parameters['stimulation_type'] == 'dc_generator':
            summary['stimulation_parameters'] = validated_parameters['dc_parameters']
        
        summary['synapse_parameters'] = validated_parameters['synapse_specification']
        summary['target_identification'] = validated_parameters['target_identification']
        if validated_parameters['target_identification'] == 'define_volume_area':
            summary['volume_area_specification'] = validated_parameters['volume_area_specification']
        
        print(f"[{self.name}] Stimulation summary compiled")
        return {'stimulation_summary': summary}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_stimulation_summary(self) -> Dict[str, Any]:
        """
        Get summary of the created stimulation.
        
        Returns:
            Dictionary with stimulation summary
        """
        return self.get_output('stimulation_summary')