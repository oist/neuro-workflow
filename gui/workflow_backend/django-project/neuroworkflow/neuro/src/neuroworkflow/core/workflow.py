"""
Workflow engine for executing connected nodes.

This module defines the Workflow class, which represents a complete workflow
with nodes and connections, and the WorkflowBuilder class, which provides
a fluent interface for creating workflows.
"""

from typing import Dict, List, Set, Optional, Any
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
        """Execute the workflow.
        
        Returns:
            True if execution was successful, False otherwise
        """
        # Compute execution order if not already done
        if not self._execution_order:
            self._compute_execution_order()
            
        # Execute nodes in order
        for node_name in self._execution_order:
            node = self.nodes[node_name]
            print(f"Executing node: {node_name}")
            if not node.process():
                print(f"Error executing node: {node_name}")
                return False
                
        return True
        
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
        
    def connect(self, from_node: str, from_port: str, to_node: str, to_port: str) -> 'WorkflowBuilder':
        """Connect two nodes.
        
        Args:
            from_node: Source node name
            from_port: Source port name
            to_node: Target node name
            to_port: Target port name
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If the nodes are not found
        """
        if from_node not in self.nodes:
            raise ValueError(f"Source node '{from_node}' not found in workflow")
            
        if to_node not in self.nodes:
            raise ValueError(f"Target node '{to_node}' not found in workflow")
            
        # Create the connection
        connection = Connection(from_node, from_port, to_node, to_port)
        self.connections.append(connection)
        
        # Connect the nodes
        source_node = self.nodes[from_node]
        target_node = self.nodes[to_node]
        source_node.connect_to(from_port, target_node, to_port)
        
        return self
        
    def build(self) -> Workflow:
        """Build the workflow.
        
        Returns:
            The built workflow
        """
        return Workflow(self.name, self.nodes, self.connections)