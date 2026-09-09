# NeuroWorkflow Node Schema Documentation

This document provides a comprehensive guide to the node schema used in the NeuroWorkflow package. The schema defines how nodes are structured, how they communicate with each other, and how they process data in neural simulation workflows.

## Table of Contents

1. [Overview](#overview)
2. [Node Definition Schema](#node-definition-schema)
3. [Port System](#port-system)
4. [Parameters](#parameters)
5. [Suggested Values (Optional)](#suggested-values-optional)
6. [Optimization Metadata (Optional)](#optimization-metadata-optional)
6. [Methods](#methods)
7. [Process Steps](#process-steps)
8. [Creating Custom Nodes](#creating-custom-nodes)
9. [Example Node Implementation](#example-node-implementation)

## Overview

NeuroWorkflow uses a node-based architecture where each node represents a specific processing step in a neural simulation workflow. Nodes are connected through ports, which define the inputs and outputs of each node. The schema provides a standardized way to define nodes, ensuring type safety and proper data flow between components.

Key components of the node schema:

- **Node Definition**: Describes the node's type, inputs, outputs, parameters, and methods
- **Ports**: Define the data entry and exit points for a node
- **Parameters**: Configurable values that control the node's behavior
- **Methods**: Functions that implement the node's processing logic
- **Process Steps**: Ordered sequence of method executions within a node

## Schema Quick Reference

Minimal fields you need for a valid node:

```python
NODE_DEFINITION = NodeDefinitionSchema(
    type="my_node_type",
    description="Short description",
    parameters={
        "param": ParameterDefinition(default_value=1.0, description="Example parameter")
    },
    inputs={
        "in": PortDefinition(type=PortType.FLOAT, description="Example input")
    },
    outputs={
        "out": PortDefinition(type=PortType.FLOAT, description="Example output")
    },
    methods={
        "compute": MethodDefinition(
            description="Example method",
            inputs=["in"],
            outputs=["out"]
        )
    }
)
```

Workflow-level context (optional):

```python
from neuroworkflow.core.workflow import WorkflowBuilder

builder = WorkflowBuilder(
    "example_workflow",
    context={
        "species": "human",
        "metadata_sources": ["literature"],
        "resource_requirements": {"cpus": 4, "memory_gb": 16}
    }
)
```

## Node Definition Schema

The `NodeDefinitionSchema` is the core of the node system. It defines all aspects of a node's structure and behavior.

```python
@dataclass
class NodeDefinitionSchema:
    type: str                  # Unique identifier for the node type
    description: str           # Human-readable description of the node
    parameters: Dict[str, Union[ParameterDefinition, Dict[str, Any], Any]] = field(default_factory=dict)
    inputs: Dict[str, Union[PortDefinition, Dict[str, Any], str]] = field(default_factory=dict)
    outputs: Dict[str, Union[PortDefinition, Dict[str, Any], str]] = field(default_factory=dict)
    methods: Dict[str, Union[MethodDefinition, Dict[str, Any], str]] = field(default_factory=dict)
```

Each node class must define a `NODE_DEFINITION` class variable using this schema. This definition is used to:

1. Auto-generate input and output ports
2. Initialize default parameter values
3. Document the node's capabilities
4. Validate connections between nodes

## Port System

Ports are the connection points between nodes. They define the data types that can flow through them and enforce type safety in the workflow.

### Port Types

The `PortType` enum defines the basic data types supported by the port system:

```python
class PortType(Enum):
    ANY = auto()      # Any type (no type checking)
    INT = auto()      # Integer values
    FLOAT = auto()    # Floating-point values
    STR = auto()      # String values
    BOOL = auto()     # Boolean values
    LIST = auto()     # List of values
    DICT = auto()     # Dictionary of values
    OBJECT = auto()   # Custom object
    FILE_PATH = auto()   # Generic file path
    CSV_FILE = auto()    # CSV data file
    JSON_FILE = auto()   # JSON configuration/data
    PICKLE_FILE = auto() # Python pickle file
    NUMPY_FILE = auto()  # NumPy array file
    HDF5_FILE = auto()   # HDF5 dataset file
```

### Port Definition

Ports are defined using the `PortDefinition` class:

```python
@dataclass
class PortDefinition:
    type: Union[PortType, Type] = PortType.ANY  # Data type (PortType enum or Python type)
    description: str = ""                       # Human-readable description
    optional: bool = False                      # Whether this port is required
```

### Input and Output Ports

The system distinguishes between input and output ports:

- **Input Ports**: Receive data from other nodes or external sources
- **Output Ports**: Send data to other nodes

Ports are automatically created from the `NODE_DEFINITION` schema when a node is instantiated.

### Port Compatibility

Ports enforce type compatibility when connecting nodes:

1. `ANY` type is compatible with any other type
2. A port can receive data of its own type or any subclass
3. Special handling for numeric types (int can be converted to float)

## Parameters

Parameters are configurable values that control a node's behavior. They are defined using the `ParameterDefinition` class:

```python
@dataclass
class ParameterDefinition:
    default_value: Any = None                # Default value for the parameter
    description: str = ""                    # Human-readable description
    constraints: Dict[str, Any] = field(default_factory=dict)  # Validation constraints
    optimizable: bool = False                # Whether this parameter can be tuned during optimization
    optimization_range: Optional[List] = None  # Range for parameter tuning [min, max]
    is_objective: bool = False               # Whether this parameter is an optimization objective/target
    objective_range: Optional[List] = None   # Acceptable range for the objective value [min, max]
    suggested_values: List[Dict[str, Any]] = field(default_factory=list)  # Optional suggestions
    secret: bool = False                 # If True, store a SecretRef — never a plaintext credential
```

### Parameter Constraints

Parameters can have constraints that validate their values:

- **min/max**: Minimum and maximum values for numeric parameters
- **allowed_values**: List of allowed values
- **min_length/max_length**: Constraints for list or string length

Parameters are accessed within node methods using the `self._parameters` dictionary.

Set `secret=True` for credentials (API keys, passwords). The GUI stores a named vault
reference, not the value. At runtime, call `from neuroworkflow.core.secrets import resolve`
and `resolve(self._parameters['password'])` before using the credential. Never put
secrets in `default_value`.

## Suggested Values (Optional)

You can attach optional AI or metadata suggestions to any parameter. These are **hints** and do not affect execution unless a user selects them.

```python
'x0': ParameterDefinition(
    default_value=-2.0,
    description='Epileptor parameter x0',
    constraints={'min': -4.0, 'max': 0.0},
    suggested_values=[
        {'value': -2.2, 'source': 'literature', 'species': 'human', 'brain_region': 'temporal_lobe'},
        {'value': -1.8, 'source': 'allen_brain', 'species': 'mouse'}
    ]
)
```

## Optimization Metadata (Optional)

`optimizable`, `optimization_range`, `is_objective`, and `objective_range` are **optional metadata** for downstream optimization tools. They do not change node behavior by themselves.

### Decision Variables (Parameters to Tune)

Use `optimizable` and `optimization_range` for parameters that should be tuned during optimization:

```python
'learning_rate': ParameterDefinition(
    default_value=0.01,
    description='Learning rate',
    constraints={'min': 0.0001, 'max': 0.1},
    optimizable=True,
    optimization_range=[0.0001, 0.1]
)
```

### Objectives (Targets to Achieve)

Use `is_objective` and `objective_range` for parameters that represent optimization targets or goals:

```python
'mean_firing_rate': ParameterDefinition(
    default_value=10.0,
    description='Target mean firing rate (Hz)',
    is_objective=True,
    objective_range=[5.0, 50.0]  # Acceptable range for the objective
)
```

This separation allows optimization nodes to distinguish between:
- **Decision variables** (`optimizable=True`): Parameters to tune (e.g., `I_e`, weights)
- **Objectives** (`is_objective=True`): Metrics to achieve (e.g., firing rate, error)

The `connect` method takes four arguments:

1. Source node name
2. Source port name
3. Target node name
4. Target port name

## Methods

Methods define the processing logic of a node. They are documented using the `MethodDefinition` class:

```python
@dataclass
class MethodDefinition:
    description: str = ""                     # Human-readable description
    inputs: List[str] = field(default_factory=list)  # Input names used by this method
    outputs: List[str] = field(default_factory=list) # Output names produced by this method
```

Methods are implemented as regular Python methods in the node class. The `MethodDefinition` provides documentation and helps with automatic process step creation.

## Process Steps

Process steps define the execution sequence within a node. Each step corresponds to a method call with defined inputs and outputs.

```python
class ProcessStep:
    def __init__(self, name: str, method: Callable, description: str = "",
                inputs: List[str] = None, outputs: List[str] = None, method_key: str = None):
        # ...
```

Process steps are typically defined in the `_define_process_steps` method of a node class. They can be created automatically from the `NODE_DEFINITION.methods` dictionary.

## Creating Custom Nodes

To create a custom node:

1. Subclass the `Node` base class
2. Define a `NODE_DEFINITION` class variable using `NodeDefinitionSchema`
3. Implement the required methods
4. Override `_define_process_steps` to set up the execution sequence

Basic structure of a custom node:

```python
class MyCustomNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type='my_custom_node',
        description='Description of my custom node',
        parameters={
            'param1': ParameterDefinition(default_value=10, description='Parameter 1'),
            # More parameters...
        },
        inputs={
            'input1': PortDefinition(type=PortType.INT, description='Input 1'),
            # More inputs...
        },
        outputs={
            'output1': PortDefinition(type=PortType.FLOAT, description='Output 1'),
            # More outputs...
        },
        methods={
            'process_data': MethodDefinition(
                description='Process the input data',
                inputs=['input1'],
                outputs=['output1']
            ),
            # More methods...
        }
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step(
            "process_data",
            self.process_data,
            method_key="process_data"
        )

    def process_data(self, input1: int) -> Dict[str, float]:
        # Process the input and return the output
        result = float(input1) * self._parameters['param1']
        return {'output1': result}
```

## Example Node Implementation

### Minimal Example (One Input → One Output)

```python
from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType

class MultiplyByFactorNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="multiply_by_factor",
        description="Multiply a number by a configurable factor",
        parameters={
            "factor": ParameterDefinition(
                default_value=2.0,
                description="Multiplication factor"
            )
        },
        inputs={
            "x": PortDefinition(type=PortType.FLOAT, description="Input value")
        },
        outputs={
            "y": PortDefinition(type=PortType.FLOAT, description="Output value")
        },
        methods={
            "compute": MethodDefinition(
                description="Compute y = x * factor",
                inputs=["x"],
                outputs=["y"]
            )
        }
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self):
        self.add_process_step("compute", self.compute, method_key="compute")

    def compute(self, x: float):
        return {"y": x * self._parameters["factor"]}
```

Here's a simplified example of a spike analysis node:

```python
class SpikeAnalysisNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type='spike_analysis',
        description='Analyzes spike trains from neural simulations',

        parameters={
            'time_window': ParameterDefinition(
                default_value=[0.0, 1000.0],
                description='Time window for analysis in milliseconds',
                constraints={'min_length': 2, 'max_length': 2}
            ),
            'bin_size': ParameterDefinition(
                default_value=10.0,
                description='Bin size for spike histograms in milliseconds',
                constraints={'min': 0.1, 'max': 1000.0},
                optimizable=True,
                optimization_range=[1.0, 50.0]
            ),
            'detection_threshold': ParameterDefinition(
                default_value=0.5,
                description='Threshold for spike detection',
                constraints={'min': 0.1, 'max': 0.9},
                optimizable=True,
                optimization_range=[0.2, 0.8]
            )
        },

        inputs={
            'spike_data': PortDefinition(
                type=PortType.OBJECT,
                description='Spike recorder data from simulation'
            )
        },

        outputs={
            'firing_rates': PortDefinition(
                type=PortType.DICT,
                description='Firing rates for each neuron'
            )
        },

        methods={
            'calculate_rates': MethodDefinition(
                description='Calculate firing rates',
                inputs=['spike_data'],
                outputs=['firing_rates']
            )
        }
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step(
            "calculate_rates",
            self.calculate_rates,
            method_key="calculate_rates"
        )

    def calculate_rates(self, spike_data: Dict[str, Any]) -> Dict[str, Dict[int, float]]:
        # Implementation of firing rate calculation
        # ...
        return {'firing_rates': calculated_rates}
```

This node schema provides a flexible yet structured way to define processing components for neural simulations, ensuring type safety and proper data flow between workflow elements.
