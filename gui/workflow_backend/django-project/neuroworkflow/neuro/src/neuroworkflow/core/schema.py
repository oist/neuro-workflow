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
    
    def is_io_port(self) -> bool:
        """Check if this is an I/O port (Snakemake boundary)."""
        return isinstance(self.type, PortType) and self.type.is_io_type()
    
    def is_memory_port(self) -> bool:
        """Check if this is a memory port (internal computation)."""
        return not self.is_io_port()


@dataclass
class ParameterDefinition:
    """Definition of a parameter in a node."""
    default_value: Any = None
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    optimizable: bool = False
    optimization_range: Optional[List[Any]] = None


@dataclass
class MethodDefinition:
    """Definition of a method in a node."""
    description: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


@dataclass
class NodeDefinitionSchema:
    """Schema for node definition."""
    type: str
    description: str
    parameters: Dict[str, Union[ParameterDefinition, Dict[str, Any], Any]] = field(default_factory=dict)
    inputs: Dict[str, Union[PortDefinition, Dict[str, Any], str]] = field(default_factory=dict)
    outputs: Dict[str, Union[PortDefinition, Dict[str, Any], str]] = field(default_factory=dict)
    methods: Dict[str, Union[MethodDefinition, Dict[str, Any], str]] = field(default_factory=dict)