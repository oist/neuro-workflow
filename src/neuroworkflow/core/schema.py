"""
Schema definitions for the NeuroWorkflow system.

This module contains the dataclass definitions that form the schema for
node definitions, ports, parameters, and methods in the workflow system.
"""

import warnings
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


def _range_problem(value: Any) -> Optional[str]:
    """Describe what is wrong with a [min, max] pair, or None if it is usable.

    Non-numeric bounds are left alone: a range may legitimately be expressed in
    terms this module does not interpret.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return f"must be [min, max], got {value}"
    low, high = value
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in (low, high)):
        return None
    if low > high:
        return f"min {low} is above max {high}"
    return None


@dataclass
class ParameterDefinition:
    """Definition of a parameter in a node.

    Attributes:
        default_value: Default value for the parameter
        description: Human-readable description
        constraints: Validation constraints (min, max, allowed_values, etc.)
        optimizable: The node author's hint that this parameter is one a
            researcher would tune. An optimization study (NW_Optimization) is
            prefilled from it; the engine searches it only when no study says
            otherwise.
        optimization_range: The node author's default search range, [min, max].
            For a dict-valued parameter, a range per key instead:
            {"V_th": [-60.0, -45.0], "C_m": [200.0, 300.0]}
        is_objective: The node author's hint that this parameter states a
            target. Metadata only: targets are declared on the study
            (NW_Optimization.objectives), which an editor may prefill from this.
        objective_range: The node author's default target range, [min, max]
        suggested_values: List of suggested values for the parameter
        unit: Physical unit of the value (e.g. "Hz", "pF", "ms")
    """
    default_value: Any = None
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    optimizable: bool = False
    optimization_range: Optional[Union[List[Any], Dict[str, Any]]] = None
    is_objective: bool = False
    objective_range: Optional[List[Any]] = None
    metadata_sources: List[str] = field(default_factory=list)
    species_specific: bool = False
    suggested_values: List[Dict[str, Any]] = field(default_factory=list)
    unit: str = ""

    def __post_init__(self) -> None:
        """Warn when the search window does not lie inside the constraints.

        ``constraints`` are hard bounds: ``configure()`` rejects values outside
        them. ``optimization_range`` only says where a search should look, so a
        range reaching past the constraints describes points that could never be
        evaluated.

        A dict-valued parameter declares one range per key
        (``{"V_th": [-60.0, -45.0]}``). Only the shape of each pair is checked
        there: ``constraints`` belongs to the parameter as a whole and cannot
        bound an individual key.

        This only warns. A partly-specified optimization declaration must never
        stop a node from being imported or uploaded.
        """
        if not self.optimization_range:  # None or {} or [] all mean "unspecified"
            return

        def warn(problem: str) -> None:
            hint = self.description[:60] or f"default_value={self.default_value!r}"
            warnings.warn(
                f"optimization_range {problem} ({hint})",
                UserWarning,
                stacklevel=3,
            )

        if isinstance(self.optimization_range, dict):
            for key, pair in self.optimization_range.items():
                problem = _range_problem(pair)
                if problem:
                    warn(f"for key {key!r} {problem}")
            return

        problem = _range_problem(self.optimization_range)
        if problem:
            warn(problem)
            return

        low, high = self.optimization_range
        if not all(isinstance(v, (int, float)) for v in (low, high)):
            return  # non-numeric ranges are not checked against constraints

        c_min = self.constraints.get('min')
        c_max = self.constraints.get('max')
        if isinstance(c_min, (int, float)) and low < c_min:
            warn(f"min {low} is below the constraint min {c_min}")
        if isinstance(c_max, (int, float)) and high > c_max:
            warn(f"max {high} is above the constraint max {c_max}")


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
