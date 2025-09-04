"""
Port system for connecting nodes in the NeuroWorkflow system.

This module defines the Port classes that handle data flow between nodes,
including type checking and value propagation.
"""

from typing import Any, List, Type, Optional, Union
import inspect

from neuroworkflow.core.schema import PortType


class Port:
    """Base class for input and output ports."""
    
    def __init__(self, name: str, data_type: Type, description: str = "", port_type: Optional[PortType] = None):
        """Initialize a port.
        
        Args:
            name: Name of the port
            data_type: Python type for the port data
            description: Description of the port
            port_type: PortType enum value (optional)
        """
        self.name = name
        self.data_type = data_type
        self.description = description
        self.port_type = port_type
        self.value = None
    
    def is_compatible_with(self, other_port: 'Port') -> bool:
        """Check if this port is compatible with another port.
        
        Args:
            other_port: The port to check compatibility with
            
        Returns:
            True if the ports are compatible, False otherwise
        """
        # Any type is compatible with any other type
        if self.data_type == object or other_port.data_type == object:
            return True
            
        # Check if types are directly compatible
        if issubclass(other_port.data_type, self.data_type):
            return True
            
        # Special case for numeric types
        if self.data_type in (int, float) and other_port.data_type in (int, float):
            # int can be converted to float
            if self.data_type == float and other_port.data_type == int:
                return True
                
        return False


class InputPort(Port):
    """Input port for a node."""
    
    def __init__(self, name: str, data_type: Type, description: str = "", 
                optional: bool = False, port_type: Optional[PortType] = None):
        """Initialize an input port.
        
        Args:
            name: Name of the port
            data_type: Python type for the port data
            description: Description of the port
            optional: Whether this port is optional
            port_type: PortType enum value (optional)
        """
        super().__init__(name, data_type, description, port_type)
        self.optional = optional
        self.connected_to = None
    
    def get_value(self) -> Any:
        """Get the value of this port.
        
        If the port is connected to an output port, get the value from there.
        Otherwise, return the local value.
        
        Returns:
            The port value
        """
        if self.connected_to is not None:
            return self.connected_to.value
        return self.value
    
    def set_value(self, value: Any) -> None:
        """Set the value of this port.
        
        Args:
            value: The value to set
        """
        self.value = value


class OutputPort(Port):
    """Output port for a node."""
    
    def __init__(self, name: str, data_type: Type, description: str = "", 
                port_type: Optional[PortType] = None):
        """Initialize an output port.
        
        Args:
            name: Name of the port
            data_type: Python type for the port data
            description: Description of the port
            port_type: PortType enum value (optional)
        """
        super().__init__(name, data_type, description, port_type)
        self.connected_to: List[InputPort] = []
    
    def set_value(self, value: Any) -> None:
        """Set the value of this port.
        
        Args:
            value: The value to set
        """
        self.value = value
    
    def propagate(self) -> None:
        """Propagate the value to connected input ports."""
        for input_port in self.connected_to:
            input_port.set_value(self.value)