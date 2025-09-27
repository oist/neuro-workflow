"""
Recording Node for NeuroWorkflow Model Builder

This node creates various types of recording devices (spike recorders, multimeters, etc.)
and connects them to specific neurons or regions within a population.
It operates in dual modalities:
1. Direct execution: Creates actual NEST recording objects and connections
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


class SNNbuilder_Recordable(Node):
    """
    Recording Builder Node for Neural Model Recording.
    
    This node creates recording devices and connects them to target neurons
    within a population. It supports various recording types and targeting methods.
    
    Recording Types:
    - **spike_recorder**: Records spike times from neurons
    
    Targeting Options:
    - **full_population**: Record from all neurons in the population
    - **define_volume_area**: Record from neurons within a volume area (sphere/circle) defined by center coordinates and radius
    
    Connection Parameters:
    - Direct connections (no synapse specifications needed for recording devices)
    - Multiple recording devices can be created
    
    Outputs:
    - NEST recording objects and connections
    - Python script code for standalone execution
    - Recording metadata and connection summary
    """
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='recording_builder',
        description='Build recording devices and connect them to neural populations',
        
        parameters={
            # === RECORDING CONFIGURATION ===
            'recording_type': ParameterDefinition(
                default_value='spike_recorder',
                description='Type of recording device',
                constraints={'allowed_values': ['spike_recorder']}
            ),
            
            'number_of_devices': ParameterDefinition(
                default_value=1,
                description='Number of recording devices to create',
                constraints={'min': 1, 'max': 100}
            ),
            
            # === SPIKE RECORDER PARAMETERS ===
            'spike_recorder_parameters': ParameterDefinition(
                default_value={
                    'record_to': 'ascii',
                    'start': 0.0,
                    'stop': 1000.0
                },
                description='Parameters for spike_recorder device'
            ),
            
            # === TARGET IDENTIFICATION ===
            'target_identification': ParameterDefinition(
                default_value='full_population',
                description='Target identification method for recording',
                constraints={'allowed_values': ['full_population', 'define_volume_area']}
            ),
            
            'volume_area_specification': ParameterDefinition(
                default_value={
                    'center_coordinates': [0.0, 0.0, 0.0],
                    'radius': 0.5
                },
                description='Volume area specification with center coordinates and radius for spatial targeting'
            ),
            
            # === SCRIPT GENERATION ===
            'script_format': ParameterDefinition(
                default_value='python',
                description='Format for generated script',
                constraints={'allowed_values': ['python', 'notebook','both']}
            ),
            
            'execution_mode': ParameterDefinition(
                default_value='execution',
                description='Mode of operation',
                constraints={'allowed_values': ['execution', 'script','both']}
            )
        },
        
        inputs={
            'nest_population': PortDefinition(
                type=PortType.OBJECT,
                description='NEST population object to record from'
            ),
            
            'population_data': PortDefinition(
                type=PortType.DICT,
                description='Population metadata including spatial information'
            ),
        },
        
        outputs={
            # NEST execution outputs
            'nest_recorders': PortDefinition(
                type=PortType.OBJECT,
                description='Created NEST recording device objects'
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
            'recording_summary': PortDefinition(
                type=PortType.DICT,
                description='Summary of created recording devices and connections'
            ),
            
            'target_neurons': PortDefinition(
                type=PortType.OBJECT,
                description='NEST NodeCollection of targeted neurons',
                optional=True
            )
        },
        
        methods={
            'validate_parameters': MethodDefinition(
                description='Validate recording parameters and configuration',
                inputs=['nest_population', 'population_data'],
                outputs=['validated_parameters']
            ),
            
            'create_recording_devices': MethodDefinition(
                description='Create NEST recording devices with specified parameters',
                inputs=['validated_parameters'],
                outputs=['nest_recorders']
            ),
            
            'identify_target_neurons': MethodDefinition(
                description='Identify target neurons based on targeting specification',
                inputs=['validated_parameters'],
                outputs=['target_neurons']
            ),
            
            'create_connections': MethodDefinition(
                description='Connect recording devices to target neurons',
                inputs=['nest_recorders', 'target_neurons', 'validated_parameters'],
                outputs=[]
            ),
            
            'generate_python_script': MethodDefinition(
                description='Generate Python script code for recording setup',
                inputs=['validated_parameters'],
                outputs=['python_script']
            ),
            
            'generate_notebook_cell': MethodDefinition(
                description='Generate Jupyter notebook cell for recording setup',
                inputs=['validated_parameters'],
                outputs=['notebook_cell']
            ),
            
            'compile_summary': MethodDefinition(
                description='Compile recording summary and metadata',
                inputs=['validated_parameters', 'nest_recorders'],
                outputs=['recording_summary']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the Recording Builder Node."""
        super().__init__(name)
        self._define_process_steps()
        
        # Internal state
        self._validated_parameters = None
        self._recording_devices = None
        self._connections = None
        self._population_data = None
        
    def _define_process_steps(self):
        """Define the processing steps for the recording node."""
        # Always validate parameters first
        self.add_process_step(
            "validate_parameters",
            self.validate_parameters,
            method_key="validate_parameters"
        )
        
        self.add_process_step(
            "create_recording_devices",
            self.create_recording_devices,
            method_key="create_recording_devices"
        )
        
        self.add_process_step(
            "identify_target_neurons",
            self.identify_target_neurons,
            method_key="identify_target_neurons"
        )
        
        self.add_process_step(
            "create_connections",
            self.create_connections,
            method_key="create_connections"
        )
        
        self.add_process_step(
            "generate_python_script",
            self.generate_python_script,
            method_key="generate_python_script"
        )
        
        self.add_process_step(
            "generate_notebook_cell",
            self.generate_notebook_cell,
            method_key="generate_notebook_cell"
        )
        
        self.add_process_step(
            "compile_summary",
            self.compile_summary,
            method_key="compile_summary"
        )
    
    # ========================================================================
    # PARAMETER VALIDATION
    # ========================================================================
    
    def validate_parameters(self, nest_population=None, population_data=None) -> Dict[str, Any]:
        """
        Validate recording parameters and prepare for creation.
        
        Args:
            nest_population: NEST population object to record from
            population_data: Population metadata including spatial information
        
        Returns:
            Dictionary containing validation results and processed parameters
        """
        print(f"[{self.name}] Validating recording parameters...")
        
        errors = []
        warnings = []
        
        # Start with current parameters
        validated_params = self._parameters.copy()
        
        # Validate recording type and parameters
        recording_type = validated_params['recording_type']
        
        if recording_type == 'spike_recorder':
            spike_params = validated_params['spike_recorder_parameters']
            self._validate_spike_recorder_parameters(spike_params)
        
        # Validate target identification
        target_identification = validated_params['target_identification']
        self._validate_target_identification(target_identification)
        
        if target_identification == 'define_volume_area':
            volume_spec = validated_params['volume_area_specification']
            # Use provided population_data or fall back to input port
            pop_data = population_data if population_data is not None else self._input_ports['population_data'].value
            self._validate_volume_area_specification(volume_spec, pop_data)
        
        # Store validated parameters
        self._validated_parameters = validated_params
        
        print(f"[{self.name}] Parameter validation completed successfully")
        return {'validated_parameters': validated_params}
    
    def _validate_spike_recorder_parameters(self, params: Dict[str, Any]) -> None:
        """Validate spike recorder parameters using NEST-based validation."""
        
        # Validate logical constraints (business logic)
        if 'start' in params and 'stop' in params:
            if params['stop'] <= params['start']:
                raise ValueError("Spike recorder 'stop' must be greater than 'start'")
        
        # Check against NEST model defaults (if NEST available)
        if NEST_AVAILABLE:
            try:
                nest_defaults = nest.GetDefaults('spike_recorder')
                # Validate that provided parameters exist in NEST model
                for key in params.keys():
                    if key not in nest_defaults:
                        print(f"[{self.name}] Warning: Parameter '{key}' not found in NEST spike_recorder defaults")
            except Exception as e:
                print(f"[{self.name}] Warning: Could not validate against NEST defaults: {e}")
        else:
            print(f"[{self.name}] Warning: NEST not available - cannot validate against model defaults")
    
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
        if not isinstance(coords, (list, tuple)) or len(coords) not in [2, 3]:
            raise ValueError("center_coordinates must be a list/tuple of 2 or 3 coordinates")
        
        radius = volume_spec['radius']
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("radius must be a positive number")
        
        # Validate against population spatial dimensions if available
        if population_data and 'spatial_dimensions' in population_data:
            pop_dims = population_data['spatial_dimensions']
            coord_dims = len(coords)
            if pop_dims == '2D' and coord_dims != 2:
                raise ValueError(f"Population is 2D but center_coordinates has {coord_dims} dimensions")
            elif pop_dims == '3D' and coord_dims != 3:
                raise ValueError(f"Population is 3D but center_coordinates has {coord_dims} dimensions")
    
    # ========================================================================
    # RECORDING DEVICE CREATION
    # ========================================================================
    
    def create_recording_devices(self, validated_parameters=None) -> Dict[str, Any]:
        """
        Create NEST recording devices with specified parameters.
        
        Args:
            validated_parameters: Validated parameters for device creation
        
        Returns:
            Dictionary containing created recording devices and metadata
        """
        # Use provided parameters or fall back to stored ones
        validated_params = validated_parameters if validated_parameters is not None else self._validated_parameters
        
        if not validated_params:
            raise ValueError("Parameters must be validated before creating recording devices")
        
        print(f"[{self.name}] Creating recording devices...")
        
        # Store for later use if not already stored
        if validated_parameters is not None:
            self._validated_parameters = validated_parameters
        recording_type = validated_params['recording_type']
        number_of_devices = validated_params['number_of_devices']
        
        if not NEST_AVAILABLE:
            print(f"[{self.name}] NEST not available - skipping device creation")
            return {'nest_recorders': None, 'device_count': number_of_devices}
        
        try:
            if recording_type == 'spike_recorder':
                devices = self._create_spike_recorders(validated_params, number_of_devices)
            else:
                raise ValueError(f"Unsupported recording type: {recording_type}")
            
            self._recording_devices = devices
            
            print(f"[{self.name}] Created {len(devices)} {recording_type} device(s)")
            return {'nest_recorders': devices, 'device_count': len(devices)}
            
        except Exception as e:
            print(f"[{self.name}] Error creating recording devices: {e}")
            raise
    
    def _create_spike_recorders(self, validated_params: Dict[str, Any], count: int):
        """Create spike recorder devices."""
        spike_params = validated_params['spike_recorder_parameters']
        
        # Create devices efficiently (single NEST call)
        devices = nest.Create('spike_recorder', count, params=spike_params)
        
        return devices
    
    # ========================================================================
    # CONNECTION CREATION
    # ========================================================================
    
    def create_connections(self, nest_recorders=None, target_neurons=None, validated_parameters=None) -> Dict[str, Any]:
        """
        Create connections between target neurons and recording devices.
        
        Args:
            nest_recorders: Created NEST recording device objects
            target_neurons: NEST NodeCollection of targeted neurons
            validated_parameters: Validated parameters for connection creation
        
        Returns:
            Dictionary containing connection information and statistics
        """
        # Use provided parameters or fall back to stored ones
        validated_params = validated_parameters if validated_parameters is not None else self._validated_parameters
        
        if not validated_params:
            raise ValueError("Parameters must be validated before creating connections")
        
        print(f"[{self.name}] Creating connections...")
        
        # Store for later use if not already stored
        if validated_parameters is not None:
            self._validated_parameters = validated_parameters
        
        population = self._input_ports['nest_population'].value
        population_data = self._input_ports['population_data'].value
        
        if not NEST_AVAILABLE:
            print(f"[{self.name}] NEST not available - skipping connection creation")
            return {'connections_created': 0, 'target_neurons': 0}
        
        if population is None:
            raise ValueError("No population provided for connection")
        
        # Get target neurons
        target_neurons = self._get_target_neurons(population, population_data, validated_params)
        
        # Create connections (no synapse specifications needed for recording devices)
        devices = self._recording_devices
        if devices is None:
            raise ValueError("No recording devices available for connection")
        
        try:
            # Connect neurons to recording devices (neurons -> recorders)
            nest.Connect(target_neurons, devices)
            
            connection_count = len(target_neurons) * len(devices)
            
            print(f"[{self.name}] Connected {len(target_neurons)} neurons to {len(devices)} recording device(s)")
            print(f"[{self.name}] Total connections: {connection_count}")
            
            self._connections = {
                'source_neurons': target_neurons,
                'target_devices': devices,
                'connection_count': connection_count
            }
            
            return {
                'connections_created': connection_count,
                'target_neurons': len(target_neurons),
                'recording_devices': len(devices)
            }
            
        except Exception as e:
            print(f"[{self.name}] Error creating connections: {e}")
            raise
    
    def identify_target_neurons(self, validated_parameters=None) -> Dict[str, Any]:
        """
        Identify target neurons based on targeting specification.
        
        Args:
            validated_parameters: Validated parameters for target identification
        
        Returns:
            Dictionary containing target neurons and metadata
        """
        # Use provided parameters or fall back to stored ones
        validated_params = validated_parameters if validated_parameters is not None else self._validated_parameters
        
        if not validated_params:
            raise ValueError("Parameters must be validated before identifying target neurons")
        
        # Store for later use if not already stored
        if validated_parameters is not None:
            self._validated_parameters = validated_parameters
        execution_mode = validated_params.get('execution_mode', 'execution')
        
        if execution_mode == 'execution':
            if not NEST_AVAILABLE:
                print(f"[{self.name}] NEST not available, skipping target identification")
                return {'target_neurons': None}

            print(f"[{self.name}] Identifying target neurons...")

            nest_population = self._input_ports['nest_population'].value
            population_data = self._input_ports['population_data'].value
            target_identification = validated_params['target_identification']

            if target_identification == 'full_population':
                target_neurons = nest_population
                print(f"[{self.name}] Targeting full population ({len(nest_population)} neurons)")

            elif target_identification == 'define_volume_area':
                volume_spec = validated_params['volume_area_specification']
                target_neurons = self._get_volume_area_targets(nest_population, population_data, volume_spec)
                print(f"[{self.name}] Targeting volume area ({len(target_neurons)} neurons)")

            self._target_neurons = target_neurons
            return {'target_neurons': target_neurons}

        else:
            print(f"[{self.name}] Skipping target identification (execution_mode: {execution_mode})")
            return {'target_neurons': None}

    def _get_target_neurons(self, population, population_data: Dict[str, Any], validated_params: Dict[str, Any]):
        """Get target neurons based on targeting method."""
        target_identification = validated_params['target_identification']
        
        if target_identification == 'full_population':
            return population
        elif target_identification == 'define_volume_area':
            volume_spec = validated_params['volume_area_specification']
            return self._get_volume_area_targets(population, population_data, volume_spec)
        else:
            raise ValueError(f"Unknown target identification method: {target_identification}")
    
    def _get_volume_area_targets(self, population, population_data: Dict[str, Any], volume_spec: Dict[str, Any]):
        """Get neurons within specified volume area using efficient vectorized approach."""
        center_coordinates = np.array(volume_spec['center_coordinates'])
        radius = volume_spec['radius']
        
        positions = population_data.get('positions')
        if positions is None:
            raise ValueError("Population has no spatial positions for spatial targeting")
        
        # Efficient vectorized distance calculation (avoids sqrt, works for 2D/3D)
        d_squared = np.sum((positions - center_coordinates)**2, axis=1)
        
        # Create boolean mask for neurons within radius
        mask = d_squared <= radius * radius
        
        # Direct NodeCollection boolean indexing (NEST-native approach)
        target_neurons = population[mask]
        
        if len(target_neurons) > 0:
            return target_neurons
        else:
            raise ValueError(f"No neurons found within radius {radius} of center {center_coordinates}")
    
    # ========================================================================
    # SCRIPT GENERATION
    # ========================================================================
    
    def generate_python_script(self, validated_parameters=None) -> Dict[str, str]:
        """
        Generate standalone Python script for recording setup.
        
        Args:
            validated_parameters: Validated parameters for script generation
        
        Returns:
            Dictionary containing the generated Python script
        """
        # Use provided parameters or fall back to stored ones
        validated_params = validated_parameters if validated_parameters is not None else self._validated_parameters
        
        if not validated_params:
            raise ValueError("Parameters must be validated before generating script")
        
        print(f"[{self.name}] Generating Python script...")
        
        # Store for later use if not already stored
        if validated_parameters is not None:
            self._validated_parameters = validated_parameters
        
        # Get population variable name from metadata
        population_data = self._input_ports['population_data'].value
        population_var_name = population_data.get('acronym', 'population').lower() if population_data else 'population'
        
        script_lines = []
        
        # Header
        script_lines.extend([
            "# === RECORDING SETUP ===",
            f"# Generated by SNNbuilder_Recordable: {self.name}",
            f"# Recording type: {validated_params['recording_type']}",
            "",
            "import nest",
            "import numpy as np",
            ""
        ])
        
        # Device creation
        recording_type = validated_params['recording_type']
        number_of_devices = validated_params['number_of_devices']
        
        if recording_type == 'spike_recorder':
            spike_params = validated_params['spike_recorder_parameters']
            script_lines.extend([
                "# Spike recorder parameters",
                f"spike_params = {json.dumps(spike_params, indent=4).replace('    ', '    ')}",
                "",
                f"# Create {number_of_devices} spike_recorder device(s)",
                f"recorders = nest.Create('spike_recorder', {number_of_devices}, params=spike_params)",
                ""
            ])
        
        # Target identification
        target_identification = validated_params['target_identification']
        
        script_lines.extend([
            "# Identify target neurons",
            f"# Note: Make sure you have defined your population variable:",
            f"# {population_var_name} = nest.Create('model_name', size, positions=nest_positions)",
            f"# Population must have spatial properties for volume area targeting",
            ""
        ])
        
        if target_identification == 'full_population':
            script_lines.extend([
                f"# Target: Full population",
                f"target_neurons = {population_var_name}"
            ])
        elif target_identification == 'define_volume_area':
            volume_spec = validated_params['volume_area_specification']
            coords = volume_spec['center_coordinates']
            radius = volume_spec['radius']
            script_lines.extend([
                f"# Target: Volume area (efficient vectorized approach - works for 2D/3D)",
                f"center_coordinates = np.array({coords})",
                f"target_radius = {radius}",
                f"",
                f"# Get population positions directly from NEST population",
                f"positions = nest.GetPosition({population_var_name})",
                f"",
                f"# Efficient vectorized distance calculation (avoids sqrt, works for 2D/3D)",
                f"d_squared = np.sum((positions - center_coordinates)**2, axis=1)",
                f"",
                f"# Create boolean mask for neurons within radius",
                f"mask = d_squared <= target_radius * target_radius",
                f"",
                f"# Direct NodeCollection boolean indexing (NEST-native approach)",
                f"target_neurons = {population_var_name}[mask]",
                f"",
                f"if len(target_neurons) == 0:",
                f"    raise ValueError(f'No neurons found within radius {{target_radius}} of center {{center_coordinates}}')"
            ])
        
        script_lines.append("")
        
        # Connections
        script_lines.extend([
            "# Create connections (efficient approach)",
            "# Recording devices: Direct connections (no synapse properties needed)",
            "nest.Connect(target_neurons, recorders)",
            "# Note: Recording devices receive spikes directly, no synaptic transmission",
            "",
            f"print(f'Created {number_of_devices} recording devices')",
            f"print(f'Connected {{len(target_neurons)}} target neurons')"
        ])
        
        # Build the final script
        python_script = "\n".join(script_lines)
        
        print(f"[{self.name}] Python script generated ({len(script_lines)} lines)")
        return {'python_script': python_script}
    
    def generate_notebook_cell(self, validated_parameters=None) -> Dict[str, str]:
        """
        Generate Jupyter notebook cell for recording setup.
        
        Args:
            validated_parameters: Validated parameters for notebook cell generation
        
        Returns:
            Dictionary containing the generated notebook cell
        """
        # For now, use the same as Python script
        script_result = self.generate_python_script(validated_parameters)
        notebook_cell = script_result['python_script']
        
        return {'notebook_cell': notebook_cell}
    
    # ========================================================================
    # SUMMARY COMPILATION
    # ========================================================================
    
    def compile_summary(self, validated_parameters=None, nest_recorders=None) -> Dict[str, Any]:
        """
        Compile summary of recording configuration and connections.
        
        Args:
            validated_parameters: Validated parameters for summary compilation
            nest_recorders: Created NEST recording device objects
        
        Returns:
            Dictionary containing comprehensive recording summary
        """
        # Use provided parameters or fall back to stored ones
        params = validated_parameters if validated_parameters is not None else self._validated_parameters
        
        if not params:
            return {'error': 'No validated parameters available'}
        
        # Store for later use if not already stored
        if validated_parameters is not None:
            self._validated_parameters = validated_parameters
        population_data = self._input_ports['population_data'].value
        
        summary = {
            'recording_type': params['recording_type'],
            'number_of_devices': params['number_of_devices'],
            'target_identification': params['target_identification'],
            'execution_mode': params['execution_mode'],
            'script_format': params['script_format']
        }
        
        # Add device-specific parameters
        if params['recording_type'] == 'spike_recorder':
            summary['spike_recorder_parameters'] = params['spike_recorder_parameters']
        
        # Add targeting information
        if params['target_identification'] == 'define_volume_area':
            summary['volume_area_specification'] = params['volume_area_specification']
        
        # Add population information if available
        if population_data:
            summary['population_info'] = {
                'name': population_data.get('name', 'Unknown'),
                'acronym': population_data.get('acronym', 'Unknown'),
                'size': population_data.get('cell_count', 'Unknown'),
                'spatial_dimensions': population_data.get('spatial_dimensions', 'Unknown')
            }
        
        # Add connection information if available
        if self._connections:
            summary['connections'] = {
                'target_neurons': len(self._connections['source_neurons']),
                'recording_devices': len(self._connections['target_devices']),
                'total_connections': self._connections['connection_count']
            }
        
        return summary