"""
Base node implementation for the NeuroWorkflow system.

This module defines the Node class, which is the foundation for all workflow nodes,
and the ProcessStep class, which represents a processing step within a node.
"""

from typing import Dict, List, Any, Callable, Optional, Type, Union
import inspect

from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import InputPort, OutputPort, PortType


class ProcessStep:
    """Represents a processing step in a node."""
    
    def __init__(self, name: str, method: Callable, description: str = "", 
                inputs: List[str] = None, outputs: List[str] = None, method_key: str = None):
        """Initialize a process step.
        
        Args:
            name: Name of the process step
            method: Method to call for this step
            description: Description of the step
            inputs: List of input names
            outputs: List of output names
            method_key: Key in NODE_DEFINITION.methods (optional)
        """
        self.name = name
        self.method = method
        self.description = description
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.method_key = method_key


class Node:
    """Base class for all workflow nodes."""
    
    # Node definition schema to be overridden by subclasses
    NODE_DEFINITION = NodeDefinitionSchema(
        type='base_node',
        description='Base node class'
    )
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a node.
        
        Args:
            name: Name of the node
            description: Description of the node (optional)
        """
        self.name = name
        self.description = description or self.__class__.NODE_DEFINITION.description
        self._input_ports: Dict[str, InputPort] = {}
        self._output_ports: Dict[str, OutputPort] = {}
        self._process_steps: List[ProcessStep] = []
        self._context: Dict[str, Any] = {}
        
        # Initialize parameters from NODE_DEFINITION
        self._parameters: Dict[str, Any] = {}
        self._optimizable_parameters: Dict[str, Dict[str, Any]] = {}
        self._initialize_parameters()
        
        # Auto-create ports from NODE_DEFINITION
        self._define_ports_from_definition()
    
    def _initialize_parameters(self) -> None:
        """Initialize parameters from NODE_DEFINITION schema."""
        for name, param_def in self.__class__.NODE_DEFINITION.parameters.items():
            if isinstance(param_def, ParameterDefinition):
                self._parameters[name] = param_def.default_value
                
                # Store optimization metadata if parameter is optimizable
                if param_def.optimizable:
                    self._optimizable_parameters[name] = {
                        'optimizable': True,
                        'range': param_def.optimization_range or [],
                        'constraints': param_def.constraints
                    }
            elif isinstance(param_def, dict):
                # Handle dictionary format
                if 'default_value' in param_def:
                    self._parameters[name] = param_def['default_value']
                
                # Store optimization metadata if parameter is optimizable
                if param_def.get('optimizable', False):
                    self._optimizable_parameters[name] = {
                        'optimizable': True,
                        'range': param_def.get('optimization_range', []),
                        'constraints': param_def.get('constraints', {})
                    }
            else:
                # If it's just a value, use it as the default
                self._parameters[name] = param_def
    
    def _define_ports_from_definition(self) -> None:
        """Define input and output ports from NODE_DEFINITION."""
        # Create input ports from NODE_DEFINITION
        for name, port_def in self.__class__.NODE_DEFINITION.inputs.items():
            if isinstance(port_def, PortDefinition):
                port_type = port_def.type if isinstance(port_def.type, PortType) else None
                data_type = port_def.type.to_python_type() if isinstance(port_def.type, PortType) else port_def.type
                self.register_input(name, data_type, port_def.description, port_def.optional, port_type)
            elif isinstance(port_def, dict):
                data_type = port_def.get('type', object)
                if isinstance(data_type, str):
                    # Convert string type to actual type
                    type_map = {'int': int, 'float': float, 'str': str, 'bool': bool, 
                               'list': list, 'dict': dict, 'object': object}
                    data_type = type_map.get(data_type, object)
                desc = port_def.get('description', '')
                optional = port_def.get('optional', False)
                self.register_input(name, data_type, desc, optional)
            else:
                # If it's just a string description
                self.register_input(name, object, str(port_def))
                
        # Create output ports from NODE_DEFINITION
        for name, port_def in self.__class__.NODE_DEFINITION.outputs.items():
            if isinstance(port_def, PortDefinition):
                port_type = port_def.type if isinstance(port_def.type, PortType) else None
                data_type = port_def.type.to_python_type() if isinstance(port_def.type, PortType) else port_def.type
                self.register_output(name, data_type, port_def.description, port_type)
            elif isinstance(port_def, dict):
                data_type = port_def.get('type', object)
                if isinstance(data_type, str):
                    # Convert string type to actual type
                    type_map = {'int': int, 'float': float, 'str': str, 'bool': bool, 
                               'list': list, 'dict': dict, 'object': object}
                    data_type = type_map.get(data_type, object)
                desc = port_def.get('description', '')
                self.register_output(name, data_type, desc)
            else:
                # If it's just a string description
                self.register_output(name, object, str(port_def))
    
    def _define_process_steps(self) -> None:
        """Define process steps for this node.
        
        This method should be overridden by subclasses to define their process steps.
        """
        pass
    
    def register_input(self, name: str, data_type: Type, description: str = "", 
                      optional: bool = False, port_type: Optional[PortType] = None) -> InputPort:
        """Register an input port with type information.
        
        Args:
            name: Name of the port
            data_type: Python type for the port data
            description: Description of the port
            optional: Whether this port is optional
            port_type: PortType enum value (optional)
            
        Returns:
            The created input port
        """
        port = InputPort(name, data_type, description, optional, port_type)
        self._input_ports[name] = port
        return port
    
    def register_output(self, name: str, data_type: Type, description: str = "", 
                       port_type: Optional[PortType] = None) -> OutputPort:
        """Register an output port with type information.
        
        Args:
            name: Name of the port
            data_type: Python type for the port data
            description: Description of the port
            port_type: PortType enum value (optional)
            
        Returns:
            The created output port
        """
        port = OutputPort(name, data_type, description, port_type)
        self._output_ports[name] = port
        return port
    
    def add_process_step(self, name: str, method: Callable, description: str = "", 
                        inputs: List[str] = None, outputs: List[str] = None, method_key: str = None) -> ProcessStep:
        """Add a process step to this node with explicit link to NODE_DEFINITION method.
        
        Args:
            name: Name of the process step
            method: Method to call for this step
            description: Description of the step
            inputs: List of input names
            outputs: List of output names
            method_key: Key in NODE_DEFINITION.methods (optional)
            
        Returns:
            The created process step
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
            
        # If method_key is provided, use it to get additional information from NODE_DEFINITION
        if method_key is None:
            # Try to use the method name as the key
            method_key = method.__name__
            
        # If the method is in NODE_DEFINITION, use its definition
        if method_key in self.__class__.NODE_DEFINITION.methods:
            method_def = self.__class__.NODE_DEFINITION.methods[method_key]
            
            if isinstance(method_def, MethodDefinition):
                # Use description from NODE_DEFINITION if not explicitly provided
                if not description:
                    description = method_def.description
                # Use inputs/outputs from NODE_DEFINITION if not explicitly provided
                if not inputs and method_def.inputs:
                    inputs = method_def.inputs
                if not outputs and method_def.outputs:
                    outputs = method_def.outputs
            elif isinstance(method_def, dict):
                # If it's a dictionary, extract information
                if not description and 'description' in method_def:
                    description = method_def['description']
                if not inputs and 'inputs' in method_def:
                    inputs = method_def['inputs']
                if not outputs and 'outputs' in method_def:
                    outputs = method_def['outputs']
            else:
                # If it's just a string description
                if not description:
                    description = str(method_def)
                    
                # Try to infer inputs and outputs from method signature and docstring
                if not inputs or not outputs:
                    sig = inspect.signature(method)
                    
                    # Skip 'self' and 'context' parameters
                    params = list(sig.parameters.keys())
                    if len(params) > 0 and params[0] == 'self':
                        params = params[1:]
                    if len(params) > 0 and params[0] == 'context':
                        # Context-based method, can't infer inputs directly
                        pass
                    else:
                        # Add parameters as inputs
                        if not inputs:
                            inputs = params
        
        step = ProcessStep(name, method, description, inputs, outputs, method_key)
        self._process_steps.append(step)
        return step
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this node.
        
        Returns:
            Dictionary with node information
        """
        return {
            'name': self.name,
            'type': self.__class__.NODE_DEFINITION.type,
            'description': self.description,
            'parameters': self._parameters,
            'optimizable_parameters': self._optimizable_parameters,
            'input_ports': {name: {'type': port.data_type.__name__, 
                                  'description': port.description,
                                  'optional': port.optional} 
                           for name, port in self._input_ports.items()},
            'output_ports': {name: {'type': port.data_type.__name__, 
                                   'description': port.description} 
                            for name, port in self._output_ports.items()},
            'process_steps': [{'name': step.name, 
                              'description': step.description,
                              'inputs': step.inputs,
                              'outputs': step.outputs,
                              'method_key': step.method_key or step.method.__name__} 
                             for step in self._process_steps],
            'methods': {name: (method_def.description if isinstance(method_def, MethodDefinition) else 
                              (method_def.get('description', '') if isinstance(method_def, dict) else str(method_def)))
                       for name, method_def in self.__class__.NODE_DEFINITION.methods.items()}
        }
    
    def get_input_port(self, name: str) -> InputPort:
        """Get an input port by name.
        
        Args:
            name: Name of the port
            
        Returns:
            The input port
            
        Raises:
            ValueError: If the port is not found
        """
        if name not in self._input_ports:
            raise ValueError(f"Input port '{name}' not found in {self.name}")
        return self._input_ports[name]
    
    def get_output_port(self, name: str) -> OutputPort:
        """Get an output port by name.
        
        Args:
            name: Name of the port
            
        Returns:
            The output port
            
        Raises:
            ValueError: If the port is not found
        """
        if name not in self._output_ports:
            raise ValueError(f"Output port '{name}' not found in {self.name}")
        return self._output_ports[name]
    
    def connect_to(self, output_port: str, target_node: 'Node', input_port: str) -> None:
        """Connect an output port of this node to an input port of another node.
        
        Args:
            output_port: Name of the output port
            target_node: Target node
            input_port: Name of the input port
            
        Raises:
            ValueError: If the ports are not found
            TypeError: If the ports are not compatible
        """
        # Get node types for informational purposes only
        source_type = self.__class__.NODE_DEFINITION.type
        target_type = target_node.__class__.NODE_DEFINITION.type
        
        source_port = self.get_output_port(output_port)
        target_port = target_node.get_input_port(input_port)
        
        # Check type compatibility
        if not target_port.is_compatible_with(source_port):
            raise TypeError(f"Type mismatch: Cannot connect {self.name}.{output_port} "
                           f"({source_port.data_type.__name__}) to {target_node.name}.{input_port} "
                           f"({target_port.data_type.__name__})")
                           
        source_port.connected_to.append(target_port)
        target_port.connected_to = source_port
    
    def get_optimizable_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get all optimizable parameters with their metadata.
        
        Returns:
            Dictionary of parameter names to optimization metadata
        """
        return self._optimizable_parameters
        
    def get_output(self, name: str) -> Any:
        """Get the current value of an output port.
        
        Args:
            name: Name of the output port
            
        Returns:
            The current value of the output port
            
        Raises:
            ValueError: If the port is not found
        """
        port = self.get_output_port(name)
        return port.value
        
    def set_input(self, name: str, value: Any) -> None:
        """Set the value of an input port.
        
        Args:
            name: Name of the input port
            value: Value to set
            
        Raises:
            ValueError: If the port is not found
        """
        port = self.get_input_port(name)
        port.value = value
    
    def configure(self, **parameters) -> 'Node':
        """Configure node parameters.
        
        Only parameters defined in NODE_DEFINITION can be configured.
        Parameter values are validated against any constraints defined in NODE_DEFINITION.
        
        Args:
            **parameters: Parameter values to set
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If a parameter is invalid
        """
        for param_name, value in parameters.items():
            if param_name in self._parameters:
                # Get parameter definition
                param_def = self.__class__.NODE_DEFINITION.parameters.get(param_name)
                
                # Check parameter constraints
                if isinstance(param_def, ParameterDefinition) and param_def.constraints:
                    constraints = param_def.constraints
                    
                    # Check min/max constraints for numeric values
                    if 'min' in constraints and value < constraints['min']:
                        raise ValueError(f"Parameter '{param_name}' value {value} is below minimum {constraints['min']}")
                    if 'max' in constraints and value > constraints['max']:
                        raise ValueError(f"Parameter '{param_name}' value {value} is above maximum {constraints['max']}")
                    
                    # Check allowed values constraint
                    if 'allowed_values' in constraints and value not in constraints['allowed_values']:
                        raise ValueError(f"Parameter '{param_name}' value {value} not in allowed values: {constraints['allowed_values']}")
                
                # Check constraints in dictionary format
                elif isinstance(param_def, dict) and 'constraints' in param_def:
                    constraints = param_def['constraints']
                    
                    # Check min/max constraints for numeric values
                    if 'min' in constraints and value < constraints['min']:
                        raise ValueError(f"Parameter '{param_name}' value {value} is below minimum {constraints['min']}")
                    if 'max' in constraints and value > constraints['max']:
                        raise ValueError(f"Parameter '{param_name}' value {value} is above maximum {constraints['max']}")
                    
                    # Check allowed values constraint
                    if 'allowed_values' in constraints and value not in constraints['allowed_values']:
                        raise ValueError(f"Parameter '{param_name}' value {value} not in allowed values: {constraints['allowed_values']}")
                
                # Set the parameter value
                self._parameters[param_name] = value
            else:
                raise ValueError(f"Unknown parameter '{param_name}' for node '{self.name}'")
                
        return self
    
    def process(self) -> bool:
        """Process this node.
        
        Returns:
            True if processing was successful, False otherwise
        """
        # Update context with input port values
        for name, port in self._input_ports.items():
            self._context[name] = port.get_value()
            
        # Execute each process step
        for step in self._process_steps:
            try:
                # Get the method to call
                method = step.method
                method_key = step.method_key or method.__name__
                
                # Extract inputs for the method from context
                inputs = {input_name: self._context.get(input_name) for input_name in step.inputs}
                
                # Call the method with extracted inputs
                outputs = method(**inputs)
                
                # If the method returns None, use an empty dict
                if outputs is None:
                    outputs = {}
                # If the method returns a non-dict, wrap it in a dict
                elif not isinstance(outputs, dict):
                    # Use the first output name as the key
                    if step.outputs:
                        outputs = {step.outputs[0]: outputs}
                    else:
                        outputs = {}
                
                # Update context with outputs
                self._context.update(outputs)
                
                # Update output ports
                for output_name in step.outputs:
                    if output_name in self._output_ports and output_name in outputs:
                        self._output_ports[output_name].set_value(outputs[output_name])
                    elif output_name in step.outputs and output_name not in outputs:
                        # Warn about missing output
                        print(f"Warning: Expected output '{output_name}' not produced by process step '{step.name}' (method: {method_key})")
                        
            except Exception as e:
                print(f"Error executing process step '{step.name}' in node '{self.name}': {e}")
                return False
                
        # Propagate output values
        for port in self._output_ports.values():
            port.propagate()
            
        return True
        
    def validate(self) -> bool:
        """Validate that the node is properly configured.
        
        Returns:
            True if the node is valid, False otherwise
        """
        # Check that all required input ports have values or connections
        for name, port in self._input_ports.items():
            if not port.optional and port.value is None and port.connected_to is None:
                print(f"Required input port '{name}' in node '{self.name}' has no value or connection")
                return False
                
        return True
        
    def __str__(self) -> str:
        """Get a string representation of this node.
        
        Returns:
            String representation
        """
        result = [f"Node: {self.name} ({self.__class__.NODE_DEFINITION.type})"]
        result.append(f"Description: {self.description}")
        
        if self._parameters:
            result.append("Parameters:")
            for name, value in self._parameters.items():
                result.append(f"  {name}: {value}")
                
        if self._input_ports:
            result.append("Input Ports:")
            for name, port in self._input_ports.items():
                connected = "connected" if port.connected_to is not None else "not connected"
                optional = "optional" if port.optional else "required"
                result.append(f"  {name} ({port.data_type.__name__}, {optional}, {connected}): {port.description}")
                
        if self._output_ports:
            result.append("Output Ports:")
            for name, port in self._output_ports.items():
                connected = f"connected to {len(port.connected_to)} ports" if port.connected_to else "not connected"
                result.append(f"  {name} ({port.data_type.__name__}, {connected}): {port.description}")
                
        if self._process_steps:
            result.append("Process Steps:")
            for step in self._process_steps:
                result.append(f"  {step.name}: {step.description}")
                if step.inputs:
                    result.append(f"    Inputs: {', '.join(step.inputs)}")
                if step.outputs:
                    result.append(f"    Outputs: {', '.join(step.outputs)}")
                    
        return "\n".join(result)