"""
Schema definitions for the NeuroWorkflow system.

This module contains the dataclass definitions that form the schema for
node definitions, ports, parameters, and methods in the workflow system.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Union, Any, Type, Optional, Tuple


class PortType(Enum):
    """Enumeration of port data types."""
    # Memory-based types (existing)
    ANY = auto()
    INT = auto()
    FLOAT = auto()
    STR = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()
    OBJECT = auto()
    
    # I/O-based types (new - Snakemake boundaries)
    FILE_PATH = auto()      # Generic file path
    CSV_FILE = auto()       # CSV data file
    JSON_FILE = auto()      # JSON configuration/data
    PICKLE_FILE = auto()    # Python pickle file
    NUMPY_FILE = auto()     # NumPy array file
    HDF5_FILE = auto()      # HDF5 dataset file
    
    def to_python_type(self) -> Type:
        """Convert PortType to Python type."""
        type_map = {
            # Memory types
            PortType.INT: int,
            PortType.FLOAT: float,
            PortType.STR: str,
            PortType.BOOL: bool,
            PortType.LIST: list,
            PortType.DICT: dict,
            PortType.OBJECT: object,
            PortType.ANY: object,
            
            # I/O types (all represented as file paths)
            PortType.FILE_PATH: str,
            PortType.CSV_FILE: str,
            PortType.JSON_FILE: str,
            PortType.PICKLE_FILE: str,
            PortType.NUMPY_FILE: str,
            PortType.HDF5_FILE: str,
        }
        return type_map[self]
    
    def is_io_type(self) -> bool:
        """Check if this port type represents I/O (file-based) data."""
        io_types = {
            PortType.FILE_PATH, PortType.CSV_FILE, PortType.JSON_FILE,
            PortType.PICKLE_FILE, PortType.NUMPY_FILE, PortType.HDF5_FILE
        }
        return self in io_types
    
    def is_memory_type(self) -> bool:
        """Check if this port type represents in-memory data."""
        return not self.is_io_type()


@dataclass
class PortDefinition:
    """Definition of a port in a node."""
    type: Union[PortType, Type] = PortType.ANY
    description: str = ""
    optional: bool = False
    fan_in: bool = False
    
    def is_io_port(self) -> bool:
        """Check if this is an I/O port (Snakemake boundary)."""
        return isinstance(self.type, PortType) and self.type.is_io_type()
    
    def is_memory_port(self) -> bool:
        """Check if this is a memory port (internal computation)."""
        return not self.is_io_port()


@dataclass
class ParameterDefinition:
    """Definition of a parameter in a node.

    Attributes:
        default_value: Default value for the parameter
        description: Human-readable description
        constraints: Validation constraints (min, max, allowed_values, etc.)
        optimizable: Whether this parameter can be tuned during optimization
        optimization_range: [min, max] range for parameter tuning
        is_objective: Whether this parameter serves as an optimization objective/target
        objective_range: [min, max] acceptable range for the objective value
        suggested_values: List of suggested values for the parameter
        secret: If True, the parameter holds a vault SecretRef, never a plaintext credential
    """
    default_value: Any = None
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    optimizable: bool = False
    optimization_range: Optional[List[Any]] = None
    is_objective: bool = False
    objective_range: Optional[List[Any]] = None
    metadata_sources: List[str] = field(default_factory=list)
    species_specific: bool = False
    suggested_values: List[Dict[str, Any]] = field(default_factory=list)
    secret: bool = False


@dataclass
class MethodDefinition:
    """Definition of a method in a node."""
    description: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class ResourceRequirements:
    """Resource requirements for HPC job execution."""
    cpus: int = 1
    memory_gb: float = 4.0
    gpus: int = 0
    walltime_hours: float = 1.0
    queue: Optional[str] = None
    account: Optional[str] = None
    nodes: int = 1  # Number of compute nodes
    tasks_per_node: int = 1  # Tasks per node (for MPI)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cpus': self.cpus,
            'memory_gb': self.memory_gb,
            'gpus': self.gpus,
            'walltime_hours': self.walltime_hours,
            'queue': self.queue,
            'account': self.account,
            'nodes': self.nodes,
            'tasks_per_node': self.tasks_per_node
        }


@dataclass
class NodeDefinitionSchema:
    """Schema for node definition."""
    type: str
    description: str
    parameters: Dict[str, Union[ParameterDefinition, Dict[str, Any], Any]] = field(default_factory=dict)
    inputs: Dict[str, Union[PortDefinition, Dict[str, Any], str]] = field(default_factory=dict)
    outputs: Dict[str, Union[PortDefinition, Dict[str, Any], str]] = field(default_factory=dict)
    methods: Dict[str, Union[MethodDefinition, Dict[str, Any], str]] = field(default_factory=dict)
    stage: Optional[str] = None        # brain modeling stage (see NODE_CREATION_GUIDE.md)
    tool: Optional[str] = None         # simulator or library (e.g. "NEST", "TVB", "Brian2")
    model_source: Optional[str] = None # URL to the origin model, paper, or repository
