"""
Workflow engine for executing connected nodes.

This module defines the Workflow class, which represents a complete workflow
with nodes and connections, and the WorkflowBuilder class, which provides
a fluent interface for creating workflows.
"""

from typing import Dict, List, Set, Optional, Any
import time
from neuroworkflow.core.node import Node


class Connection:
    """Represents a connection between two nodes in a workflow."""
    
    def __init__(self, from_node: str, from_port: str, to_node: str, to_port: str):
        """Initialize a connection.
        
        Args:
            from_node: Source node name
            from_port: Source port name
            to_node: Target node name
            to_port: Target port name
        """
        self.from_node = from_node
        self.from_port = from_port
        self.to_node = to_node
        self.to_port = to_port
        
    def __str__(self) -> str:
        """Get a string representation of this connection.
        
        Returns:
            String representation
        """
        return f"{self.from_node}.{self.from_port} -> {self.to_node}.{self.to_port}"


class Workflow:
    """Represents a complete workflow with nodes and connections."""
    
    def __init__(self, name: str, nodes: Dict[str, Node], connections: List[Connection]):
        """Initialize a workflow.
        
        Args:
            name: Name of the workflow
            nodes: Dictionary of nodes (name -> node)
            connections: List of connections
        """
        self.name = name
        self.nodes = nodes
        self.connections = connections
        self._execution_order: List[str] = []
        self._execution_sequence: List[Dict[str, Any]] = []  # Track execution order and metadata
        
    def _compute_execution_order(self) -> None:
        """Compute the execution order of nodes based on dependencies.
        
        This method uses a topological sort to determine the order in which
        nodes should be executed.
        
        Raises:
            ValueError: If the workflow contains a cycle
        """
        # Simple topological sort
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []
        
        def visit(node_name: str) -> None:
            """Visit a node in the topological sort.
            
            Args:
                node_name: Name of the node to visit
                
            Raises:
                ValueError: If a cycle is detected
            """
            if node_name in temp_visited:
                raise ValueError(f"Cycle detected in workflow: {node_name} is part of a cycle")
                
            if node_name in visited:
                return
                
            temp_visited.add(node_name)
            
            # Visit all nodes that depend on this node
            for conn in self.connections:
                if conn.from_node == node_name:
                    visit(conn.to_node)
                    
            temp_visited.remove(node_name)
            visited.add(node_name)
            order.append(node_name)
            
        # Visit all nodes
        for node_name in self.nodes:
            if node_name not in visited:
                visit(node_name)
                
        # Reverse to get correct execution order
        self._execution_order = list(reversed(order))
        
    def validate(self) -> bool:
        """Validate the workflow.
        
        Returns:
            True if the workflow is valid, False otherwise
        """
        # Check that all nodes are properly configured
        for node in self.nodes.values():
            if not node.validate():
                return False
                
        # Check that all connections are valid
        for conn in self.connections:
            if conn.from_node not in self.nodes:
                print(f"Source node '{conn.from_node}' not found in workflow")
                return False
                
            if conn.to_node not in self.nodes:
                print(f"Target node '{conn.to_node}' not found in workflow")
                return False
                
            source_node = self.nodes[conn.from_node]
            target_node = self.nodes[conn.to_node]
            
            try:
                source_port = source_node.get_output_port(conn.from_port)
            except ValueError:
                print(f"Output port '{conn.from_port}' not found in node '{conn.from_node}'")
                return False
                
            try:
                target_port = target_node.get_input_port(conn.to_port)
            except ValueError:
                print(f"Input port '{conn.to_port}' not found in node '{conn.to_node}'")
                return False
                
            # Get node types for informational purposes only
            source_type = source_node.__class__.NODE_DEFINITION.type
            target_type = target_node.__class__.NODE_DEFINITION.type
                
            # Check type compatibility
            if not target_port.is_compatible_with(source_port):
                print(f"Type mismatch: Cannot connect {conn.from_node}.{conn.from_port} "
                     f"({source_port.data_type.__name__}) to {conn.to_node}.{conn.to_port} "
                     f"({target_port.data_type.__name__})")
                return False
                
        # Check for cycles
        try:
            self._compute_execution_order()
        except ValueError as e:
            print(str(e))
            return False
            
        return True
        
    def execute(self) -> bool:
        """Execute the workflow with execution tracking.
        
        Returns:
            True if execution was successful, False otherwise
        """
        # Clear previous execution sequence
        self._execution_sequence.clear()
        
        # Compute execution order if not already done
        if not self._execution_order:
            self._compute_execution_order()
            
        # Execute nodes in order with tracking
        for node_name in self._execution_order:
            node = self.nodes[node_name]
            
            # Track execution start
            start_time = time.time()
            print(f"Executing node: {node_name}")
            
            # Execute the node
            success = node.process()
            
            # Track execution metadata
            execution_entry = {
                'node_name': node_name,
                'node_instance': node,  # Direct reference to node object
                'node_type': getattr(getattr(node, 'NODE_DEFINITION', None), 'type', 'unknown'),
                'execution_order': len(self._execution_sequence),
                'timestamp': start_time,
                'duration': time.time() - start_time,
                'success': success,
                'has_python_script': self._node_has_output_port(node, 'python_script'),
                'has_notebook_cell': self._node_has_output_port(node, 'notebook_cell'),
                'output_ports': list(node._output_ports.keys()) if hasattr(node, '_output_ports') else []
            }
            
            self._execution_sequence.append(execution_entry)
            
            if not success:
                print(f"Error executing node: {node_name}")
                return False
                
        return True
    
    def _node_has_output_port(self, node, port_name: str) -> bool:
        """Check if node has a specific output port with content.
        
        Args:
            node: The node to check
            port_name: Name of the output port
            
        Returns:
            True if node has the port with content, False otherwise
        """
        if not hasattr(node, '_output_ports'):
            return False
        if port_name not in node._output_ports:
            return False
        value = node._output_ports[port_name].value
        return value is not None and str(value).strip() != ''
    
    def get_execution_sequence(self) -> Dict[str, Any]:
        """Get execution sequence as dictionary.
        
        Returns:
            Dictionary containing execution sequence and metadata
        """
        return {
            'workflow_name': self.name,
            'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_nodes': len(self._execution_sequence),
            'execution_sequence': self._execution_sequence,
            'workflow_nodes': self.nodes,  # Direct reference to all nodes
            'connections': [
                {
                    'from_node': conn.from_node,
                    'from_port': conn.from_port,
                    'to_node': conn.to_node,
                    'to_port': conn.to_port
                } for conn in self.connections
            ]
        }
        
    def get_info(self) -> Dict[str, Any]:
        """Get information about this workflow.
        
        Returns:
            Dictionary with workflow information
        """
        return {
            'name': self.name,
            'nodes': {name: node.get_info() for name, node in self.nodes.items()},
            'connections': [{'from_node': conn.from_node, 
                            'from_port': conn.from_port, 
                            'to_node': conn.to_node, 
                            'to_port': conn.to_port} 
                           for conn in self.connections],
            'execution_order': self._execution_order
        }
        
    def __str__(self) -> str:
        """Get a string representation of this workflow.
        
        Returns:
            String representation
        """
        result = [f"Workflow: {self.name}"]
        result.append("Nodes:")
        for name in self.nodes:
            result.append(f"  {name}")
            
        result.append("Connections:")
        for conn in self.connections:
            result.append(f"  {conn}")
            
        if self._execution_order:
            result.append("Execution Order:")
            result.append(f"  {' -> '.join(self._execution_order)}")
            
        return "\n".join(result)


class WorkflowBuilder:
    """Builder pattern for creating workflows."""
    
    def __init__(self, name: str):
        """Initialize a workflow builder.
        
        Args:
            name: Name of the workflow
        """
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []
        self._execution_sequence: List[Dict[str, Any]] = []  # Track execution order and metadata
        
    def add_node(self, node: Node) -> 'WorkflowBuilder':
        """Add a node to the workflow.
        
        Args:
            node: Node to add
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If a node with the same name already exists
        """
        if node.name in self.nodes:
            raise ValueError(f"Node with name '{node.name}' already exists in workflow")
            
        self.nodes[node.name] = node
        return self
        
    def connect(self, from_node: str, from_port: str, to_node: str, to_port: str, 
                allow_duplicates: bool = False, strict: bool = False) -> 'WorkflowBuilder':
        """Connect two nodes.
        
        Args:
            from_node: Source node name
            from_port: Source port name
            to_node: Target node name
            to_port: Target port name
            allow_duplicates: If True, create duplicate connections (default: False)
            strict: If True, raise error on duplicates; if False, silently skip (default: False)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If the nodes are not found, or if strict=True and connection already exists
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found in workflow")
            
        if to_node not in self.nodes:
            raise ValueError(f"Target node '{to_node}' not found in workflow")
        
        # Check for duplicate connections
        if not allow_duplicates:
            for existing_conn in self.connections:
                if (existing_conn.from_node == from_node and 
                    existing_conn.from_port == from_port and
                    existing_conn.to_node == to_node and 
                    existing_conn.to_port == to_port):
                    
                    if strict:
                        # Strict mode: raise error
                        raise ValueError(f"Connection already exists: {from_node}.{from_port} -> {to_node}.{to_port}. "
                                       f"Use allow_duplicates=True to override.")
                    else:
                        # Default mode: silently skip (Jupyter-friendly)
                        return self
        
        # Get nodes for port-level duplicate checking
        source_node = self.nodes[from_node]
        target_node = self.nodes[to_node]
        
        # Check for port-level duplicates
        if not allow_duplicates:
            try:
                source_port_obj = source_node.get_output_port(from_port)
                target_port_obj = target_node.get_input_port(to_port)
                
                # Check if this specific connection already exists at port level
                if target_port_obj in source_port_obj.connected_to:
                    if strict:
                        # Strict mode: raise error
                        raise ValueError(f"Port connection already exists: {from_node}.{from_port} -> {to_node}.{to_port}. "
                                       f"Use allow_duplicates=True to override.")
                    else:
                        # Default mode: silently skip (Jupyter-friendly)
                        return self
                    
            except ValueError as e:
                # If it's a port not found error, let it propagate
                if "not found" in str(e):
                    raise e
                # Otherwise, it might be our duplicate check, so re-raise
                raise e
            
        # Create the connection
        connection = Connection(from_node, from_port, to_node, to_port)
        self.connections.append(connection)
        
        # Connect the nodes
        source_node.connect_to(from_port, target_node, to_port)
        
        return self
    
    def connection_exists(self, from_node: str, from_port: str, to_node: str, to_port: str) -> bool:
        """Check if a connection already exists.
        
        Args:
            from_node: Source node name
            from_port: Source port name
            to_node: Target node name
            to_port: Target port name
            
        Returns:
            True if connection exists, False otherwise
        """
        for existing_conn in self.connections:
            if (existing_conn.from_node == from_node and 
                existing_conn.from_port == from_port and
                existing_conn.to_node == to_node and 
                existing_conn.to_port == to_port):
                return True
        return False
    
    def connect_safe(self, from_node: str, from_port: str, to_node: str, to_port: str) -> 'WorkflowBuilder':
        """Connect two nodes, silently skipping if connection already exists.
        
        This method is identical to the default connect() behavior but makes the intent explicit.
        
        Args:
            from_node: Source node name
            from_port: Source port name
            to_node: Target node name
            to_port: Target port name
            
        Returns:
            Self for method chaining
        """
        return self.connect(from_node, from_port, to_node, to_port, allow_duplicates=False, strict=False)
    
    def connect_strict(self, from_node: str, from_port: str, to_node: str, to_port: str) -> 'WorkflowBuilder':
        """Connect two nodes, raising error if connection already exists.
        
        This method will raise a ValueError if the connection already exists,
        useful for catching duplicate connection bugs during development.
        
        Args:
            from_node: Source node name
            from_port: Source port name
            to_node: Target node name
            to_port: Target port name
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If connection already exists
        """
        return self.connect(from_node, from_port, to_node, to_port, allow_duplicates=False, strict=True)
    
    def connect_force(self, from_node: str, from_port: str, to_node: str, to_port: str) -> 'WorkflowBuilder':
        """Connect two nodes, allowing duplicate connections.
        
        This method will create duplicate connections if called multiple times
        with the same parameters.
        
        Args:
            from_node: Source node name
            from_port: Source port name
            to_node: Target node name
            to_port: Target port name
            
        Returns:
            Self for method chaining
        """
        return self.connect(from_node, from_port, to_node, to_port, allow_duplicates=True, strict=False)
    
    def clear_connections(self) -> None:
        """Clear all connections from the workflow.
        
        This removes connections both from the workflow's connection list
        and from the individual port objects.
        """
        # Clear workflow connections
        self.connections.clear()
        
        # Clear port connections
        for node in self.nodes.values():
            # Clear output port connections
            for port in node._output_ports.values():
                port.connected_to.clear()
            
            # Clear input port connections
            for port in node._input_ports.values():
                port.connected_to = None
    
    def get_connection_count(self) -> int:
        """Get the total number of connections in the workflow.
        
        Returns:
            Number of connections
        """
        return len(self.connections)
    
    def list_connections(self) -> List[str]:
        """Get a list of all connections as strings.
        
        Returns:
            List of connection strings
        """
        return [str(conn) for conn in self.connections]
        
    def build(self) -> Workflow:
        """Build the workflow.
        
        Returns:
            The built workflow
        """
        return Workflow(self.name, self.nodes, self.connections)
    
    def execute_workflow(self) -> bool:
        """Execute the workflow and track execution sequence.
        
        Returns:
            True if execution was successful, False otherwise
        """
        # Clear previous execution sequence
        self._execution_sequence.clear()
        
        # Build and execute workflow
        workflow = self.build()
        
        # Execute with tracking
        return self._execute_with_tracking(workflow)
    
    def _execute_with_tracking(self, workflow: Workflow) -> bool:
        """Execute workflow while tracking execution sequence.
        
        Args:
            workflow: The workflow to execute
            
        Returns:
            True if execution was successful, False otherwise
        """
        # Compute execution order if not already done
        if not workflow._execution_order:
            workflow._compute_execution_order()

        # Execute nodes in order with tracking
        for node_name in workflow._execution_order:
            node = workflow.nodes[node_name]
            
            # Track execution start
            start_time = time.time()
            print(f"Executing node: {node_name}")
            
            # Execute the node
            success = node.process()
            
            # Track execution metadata
            execution_entry = {
                'node_name': node_name,
                'node_instance': node,  # Direct reference to node object
                'node_type': getattr(getattr(node, 'NODE_DEFINITION', None), 'type', 'unknown'),
                'execution_order': len(self._execution_sequence),
                'timestamp': start_time,
                'duration': time.time() - start_time,
                'success': success,
                'has_python_script': self._node_has_output_port(node, 'python_script'),
                'has_notebook_cell': self._node_has_output_port(node, 'notebook_cell'),
                'output_ports': list(node._output_ports.keys()) if hasattr(node, '_output_ports') else []
            }
            
            self._execution_sequence.append(execution_entry)
            
            if not success:
                print(f"Error executing node: {node_name}")
                return False

        return True
    
    def _node_has_output_port(self, node, port_name: str) -> bool:
        """Check if node has a specific output port with content.
        
        Args:
            node: The node to check
            port_name: Name of the output port
            
        Returns:
            True if node has the port with content, False otherwise
        """
        if not hasattr(node, '_output_ports'):
            return False
        if port_name not in node._output_ports:
            return False
        value = node._output_ports[port_name].value
        return value is not None and str(value).strip() != ''
    
    def get_execution_sequence(self) -> Dict[str, Any]:
        """Get execution sequence as dictionary.
        
        Returns:
            Dictionary containing execution sequence and metadata
        """
        return {
            'workflow_name': self.name,
            'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_nodes': len(self._execution_sequence),
            'execution_sequence': self._execution_sequence,
            'workflow_nodes': self.nodes,  # Direct reference to all nodes
            'connections': [
                {
                    'from_node': conn.from_node,
                    'from_port': conn.from_port,
                    'to_node': conn.to_node,
                    'to_port': conn.to_port
                } for conn in self.connections
            ]
        }
    
    def export_execution_sequence(self) -> Dict[str, Any]:
        """Export execution sequence (alias for get_execution_sequence).
        
        Returns:
            Dictionary containing execution sequence and metadata
        """
        return self.get_execution_sequence()