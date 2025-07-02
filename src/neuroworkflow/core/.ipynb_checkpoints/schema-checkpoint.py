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
    ANY = auto()
    INT = auto()
    FLOAT = auto()
    STR = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()
    OBJECT = auto()
    
    def to_python_type(self) -> Type:
        """Convert PortType to Python type."""
        type_map = {
            PortType.INT: int,
            PortType.FLOAT: float,
            PortType.STR: str,
            PortType.BOOL: bool,
            PortType.LIST: list,
            PortType.DICT: dict,
            PortType.OBJECT: object,
            PortType.ANY: object
        }
        return type_map[self]


@dataclass
class PortDefinition:
    """Definition of a port in a node."""
    type: Union[PortType, Type] = PortType.ANY
    description: str = ""
    optional: bool = False


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