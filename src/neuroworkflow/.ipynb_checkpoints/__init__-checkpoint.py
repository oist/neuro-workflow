"""
NeuroWorkflow: A Python library for building and executing neural simulation workflows.

This library provides a flexible, type-safe framework for creating and executing
workflows for neural simulations and analysis.
"""

__version__ = "0.1.0"

# Core components
from neuroworkflow.core.node import Node
from neuroworkflow.core.workflow import Workflow, WorkflowBuilder
from neuroworkflow.core.schema import (
    NodeDefinitionSchema, 
    PortDefinition, 
    ParameterDefinition,
    MethodDefinition,
    PortType
)