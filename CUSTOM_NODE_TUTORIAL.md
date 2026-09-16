# Custom Node Development Tutorial for NeuroWorkflow

This comprehensive tutorial will guide you through creating custom nodes for the NeuroWorkflow system. By the end of this tutorial, you'll be able to build sophisticated, reusable processing components that integrate seamlessly with the workflow system.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Understanding the Node Architecture](#understanding-the-node-architecture)
4. [Step-by-Step Node Creation](#step-by-step-node-creation)
5. [Optional: Suggested Values](#optional-suggested-values)
6. [Optional: Optimization Metadata](#optional-optimization-metadata)
6. [Best Practices](#best-practices)
7. [Testing Your Node](#testing-your-node)
8. [Integration with Workflows](#integration-with-workflows)
9. [Common Patterns and Examples](#common-patterns-and-examples)
10. [Troubleshooting](#troubleshooting)

## Overview

NeuroWorkflow uses a **node-based architecture** where each node represents a specific processing step in a neural simulation workflow. Nodes communicate through **typed ports** and can be connected to form complex processing pipelines.

### Key Concepts:

- **Nodes**: Self-contained processing units
- **Ports**: Typed input/output interfaces
- **Parameters**: Configurable values that control node behavior
- **Process Steps**: Ordered sequence of method executions
- **Workflows**: Connected networks of nodes

## Prerequisites

Before starting, ensure you have:

1. **Python 3.8+** installed
2. **NeuroWorkflow** library accessible in your Python path
3. Basic understanding of:
   - Python classes and inheritance
   - Type hints
   - Dataclasses
   - Neural simulation concepts (optional but helpful)

### Required Imports

Every custom node needs these core imports:

```python
from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition
)
from neuroworkflow.core.port import PortType
```

## Understanding the Node Architecture

### Node Components

Every NeuroWorkflow node consists of:

1. **NODE_DEFINITION**: Schema defining the node's interface
2. **Parameters**: Configurable values
3. **Input Ports**: Data entry points
4. **Output Ports**: Data exit points
5. **Methods**: Processing functions
6. **Process Steps**: Execution sequence

### Node Lifecycle

1. **Initialization**: Node is created with default parameters
2. **Configuration**: Parameters are set
3. **Connection**: Ports are connected to other nodes
4. **Validation**: Inputs and parameters are validated
5. **Execution**: Process steps run in sequence
6. **Output**: Results are made available through output ports

## Step-by-Step Node Creation

### Step 1: Define Your Node Class

Start by creating a class that inherits from `Node`:

```python
class MyCustomNode(Node):
    """
    Brief description of what your node does.

    Detailed description including:
    - Input requirements
    - Processing performed
    - Output produced
    - Dependencies or special requirements
"""
```

### Minimal Example (One Input → One Output)

Use this as the smallest working node:

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

### Minimal Workflow Example

```python
from neuroworkflow.core.workflow import WorkflowBuilder

node_a = MultiplyByFactorNode("multiply_by_2")
node_a.configure(factor=2.0)

node_b = MultiplyByFactorNode("multiply_by_3")
node_b.configure(factor=3.0)

workflow = (
    WorkflowBuilder("simple_multiply_workflow")
    .add_node(node_a)
    .add_node(node_b)
    .connect("multiply_by_2", "y", "multiply_by_3", "x")
    .build()
)

# Provide initial input and run
node_a.set_input("x", 5.0)
workflow.execute()

print(node_b.get_output("y"))  # 5 * 2 * 3 = 30
```

### Workflow Context Example (Optional)

You can define workflow-level context once and it will be injected into all nodes.

```python
from neuroworkflow.core.workflow import WorkflowBuilder

builder = WorkflowBuilder(
    "simple_multiply_workflow",
    context={
        "species": "human",
        "metadata_sources": ["literature"],
        "resource_requirements": {"cpus": 4, "memory_gb": 16}
    }
)

workflow = (
    builder
    .add_node(node_a)
    .add_node(node_b)
    .connect("multiply_by_2", "y", "multiply_by_3", "x")
    .build()
)
```

### Step 2: Create the NODE_DEFINITION

This is the **most important part** - it defines your node's interface:

```python
NODE_DEFINITION = NodeDefinitionSchema(
    type='my_custom_node',  # Unique identifier
    description='What this node does',

    parameters={
        # Define your parameters here
    },

    inputs={
        # Define input ports here
    },

    outputs={
        # Define output ports here
    },

    methods={
        # Document your methods here
    }
)
```

### Step 3: Define Parameters

Parameters control your node's behavior:

```python
parameters={
    # Simple parameter
    'threshold': ParameterDefinition(
        default_value=0.5,
        description='Threshold for processing'
    ),

    # Parameter with constraints
    'window_size': ParameterDefinition(
        default_value=100,
        description='Window size in samples',
        constraints={'min': 1, 'max': 1000}
    ),

    # Choice parameter
    'method': ParameterDefinition(
        default_value='linear',
        description='Processing method',
        constraints={'allowed_values': ['linear', 'nonlinear', 'adaptive']}
    )
}
```

## Optional: Suggested Values

If you want AI or metadata suggestions, you can attach them to parameters. These are **hints only** and do not affect execution unless a user selects them.

```python
'threshold': ParameterDefinition(
    default_value=0.5,
    description='Threshold for processing',
    constraints={'min': 0.0, 'max': 1.0},
    suggested_values=[
        {'value': 0.4, 'source': 'literature', 'species': 'human'},
        {'value': 0.6, 'source': 'allen_brain', 'species': 'mouse'}
    ]
)
```

## Optional: Optimization Metadata

Optimization metadata fields are optional and used by downstream optimization tools. If you are not using optimization, you can ignore them.

### Decision Variables (Parameters to Tune)

Use `optimizable` and `optimization_range` for parameters that should be tuned during optimization:

```python
'learning_rate': ParameterDefinition(
    default_value=0.01,
    description='Learning rate for adaptation',
    constraints={'min': 0.001, 'max': 0.1},
    optimizable=True,
    optimization_range=[0.001, 0.1]
)
```

### Objectives (Targets to Achieve)

A target — a measured output steered into a range — belongs to the optimization study, not to the node. It is declared on an `NW_Optimization` node (`objectives=[{"name": ..., "measures": "Analysis.firing_rate_hz.exc", "low": 40.0, "high": 50.0}]`), so do not add a parameter the simulation never reads just to hold a target value. `is_objective` / `objective_range` exist only as a hint that a genuine parameter states a set-point; see `docs/OPTIMIZATION.md`.

### Step 4: Define Input Ports

Input ports specify what data your node receives:

```python
inputs={
    # Required input
    'signal_data': PortDefinition(
        type=PortType.OBJECT,
        description='Input signal data'
    ),

    # Optional input
    'sampling_rate': PortDefinition(
        type=PortType.FLOAT,
        description='Sampling rate in Hz',
        optional=True
    ),

    # File input
    'config_file': PortDefinition(
        type=PortType.JSON_FILE,
        description='Configuration file path',
        optional=True
    )
}
```

### Step 5: Define Output Ports

Output ports specify what data your node produces:

```python
outputs={
    # Main output
    'processed_signal': PortDefinition(
        type=PortType.OBJECT,
        description='Processed signal data'
    ),

    # Statistics output
    'analysis_results': PortDefinition(
        type=PortType.DICT,
        description='Analysis statistics and metrics'
    ),

    # Optional output
    'debug_info': PortDefinition(
        type=PortType.DICT,
        description='Debug information',
        optional=True
    )
}
```

### Step 6: Implement the Constructor

```python
def __init__(self, name: str):
    """Initialize the node."""
    super().__init__(name)

    # Initialize any additional instance variables
    self._cache = {}
    self._state = 'initialized'

    # REQUIRED: Define process steps
    self._define_process_steps()
```

### Step 7: Define Process Steps

This determines the execution order of your methods:

```python
def _define_process_steps(self) -> None:
    """Define the sequence of processing steps."""

    self.add_process_step(
        "validate_inputs",
        self.validate_inputs,
        method_key="validate_inputs"
    )

    self.add_process_step(
        "process_signal",
        self.process_signal,
        method_key="process_signal"
    )

    self.add_process_step(
        "analyze_results",
        self.analyze_results,
        method_key="analyze_results"
    )
```

### Step 8: Implement Processing Methods

Each method should:

- Take inputs as parameters
- Return outputs as a dictionary
- Handle errors gracefully
- Provide informative logging

```python
def validate_inputs(self, signal_data: Any, sampling_rate: Optional[float] = None) -> Dict[str, Any]:
    """Validate input data and parameters."""
    print(f"[{self.name}] Validating inputs...")

    # Validation logic here
    if signal_data is None:
        raise ValueError("Signal data cannot be None")

    # Return validation results
    return {'validation_passed': True}

def process_signal(self, signal_data: Any, sampling_rate: Optional[float] = None) -> Dict[str, Any]:
    """Main signal processing method."""
    print(f"[{self.name}] Processing signal...")

    # Access parameters
    threshold = self._parameters['threshold']
    method = self._parameters['method']

    # Processing logic here
    processed_data = self._apply_processing(signal_data, threshold, method)

    return {'processed_signal': processed_data}

def analyze_results(self, processed_signal: Any) -> Dict[str, Any]:
    """Analyze processing results."""
    print(f"[{self.name}] Analyzing results...")

    # Analysis logic here
    stats = self._calculate_statistics(processed_signal)

    return {'analysis_results': stats}
```

## Advanced Features

### Parameter Optimization Support

Make parameters optimizable for automatic tuning:

```python
# Decision variable - parameter to tune
'amplitude': ParameterDefinition(
    default_value=1.0,
    description='Signal amplitude',
    constraints={'min': 0.1, 'max': 10.0},
    optimizable=True,  # Enable optimization
    optimization_range=[0.1, 5.0]  # Optimization bounds
)
```

Targets are not parameters: declare them on an `NW_Optimization` node (see `docs/OPTIMIZATION.md`).

### File I/O Operations

Handle file-based inputs and outputs:

```python
def load_data(self, data_file: str) -> Dict[str, Any]:
    """Load data from file."""
    import pandas as pd

    try:
        if data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
        elif data_file.endswith('.json'):
            import json
            with open(data_file, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")

        return {'loaded_data': data}

    except Exception as e:
        raise ValueError(f"Failed to load data from {data_file}: {e}")
```

## Best Practices

### 1. Clear Documentation

```python
class WellDocumentedNode(Node):
    """
    Clear, comprehensive description of the node.

    This node performs X operation on Y type of data, producing Z outputs.
    It is designed for use in neural simulation workflows where...

    Requirements:
    - Input data must be in format X
    - Requires Y dependency for full functionality

    Example Usage:
        node = WellDocumentedNode("processor")
        node.configure(parameter1=value1)
        # Connect to workflow...
    """
```

### 2. Robust Error Handling

```python
def process_data(self, input_data: Any) -> Dict[str, Any]:
    """Process data with robust error handling."""

    try:
        # Validate inputs
        if not self._validate_input_format(input_data):
            raise ValueError("Invalid input format")

        # Process data
        result = self._core_processing(input_data)

        # Validate outputs
        if not self._validate_output_format(result):
            raise RuntimeError("Processing produced invalid output")

        return {'processed_data': result}

    except Exception as e:
        # Log error with context
        print(f"[{self.name}] Error in process_data: {e}")
        print(f"[{self.name}] Input type: {type(input_data)}")
        print(f"[{self.name}] Parameters: {self._parameters}")

        # Re-raise with additional context
        raise RuntimeError(f"Data processing failed in {self.name}: {e}") from e
```

### 3. Parameter Validation

```python
def _validate_parameters(self) -> None:
    """Validate parameter values and combinations."""

    # Check individual parameters
    if self._parameters['threshold'] < 0:
        raise ValueError("Threshold must be non-negative")

    # Check parameter combinations
    if (self._parameters['method'] == 'adaptive' and
        self._parameters['learning_rate'] <= 0):
        raise ValueError("Adaptive method requires positive learning rate")

    # Check dependencies
    if (self._parameters['enable_advanced_features'] and
        not self._check_advanced_dependencies()):
        raise RuntimeError("Advanced features require additional dependencies")
```

### 4. Informative Logging

```python
def process_data(self, input_data: Any) -> Dict[str, Any]:
    """Process data with informative logging."""

    print(f"[{self.name}] Starting data processing")
    print(f"[{self.name}] Input shape: {getattr(input_data, 'shape', 'N/A')}")
    print(f"[{self.name}] Method: {self._parameters['method']}")

    start_time = time.time()

    # Processing logic
    result = self._do_processing(input_data)

    elapsed = time.time() - start_time
    print(f"[{self.name}] Processing completed in {elapsed:.2f}s")
    print(f"[{self.name}] Output shape: {getattr(result, 'shape', 'N/A')}")

    return {'processed_data': result}
```

### 5. Modular Design

Break complex processing into smaller, testable methods:

```python
def process_data(self, input_data: Any) -> Dict[str, Any]:
    """Main processing method - orchestrates sub-operations."""

    # Preprocessing
    cleaned_data = self._preprocess_data(input_data)

    # Core processing
    processed_data = self._apply_core_algorithm(cleaned_data)

    # Post-processing
    final_result = self._postprocess_data(processed_data)

    return {'processed_data': final_result}

def _preprocess_data(self, data: Any) -> Any:
    """Preprocess input data."""
    # Specific preprocessing logic
    pass

def _apply_core_algorithm(self, data: Any) -> Any:
    """Apply the main algorithm."""
    # Core algorithm implementation
    pass

def _postprocess_data(self, data: Any) -> Any:
    """Post-process results."""
    # Post-processing logic
    pass
```

### Integration Testing

Test your node in a workflow context:

```python
def test_node_in_workflow():
    """Test node integration in a workflow."""
    from neuroworkflow.core.workflow import WorkflowBuilder

    # Create nodes
    source_node = DataSourceNode("source")
    your_node = YourCustomNode("processor")
    sink_node = DataSinkNode("sink")

    # Create workflow
    workflow = (
        WorkflowBuilder("test_workflow")
        .add_node(source_node)
        .add_node(your_node)
        .add_node(sink_node)
        .connect("source", "data", "processor", "input_data")
        .connect("processor", "processed_data", "sink", "input_data")
        .build()
    )

    # Test workflow execution
    # ... workflow execution logic
```

## Integration with Workflows

### Creating Workflows

```python
from neuroworkflow.core.workflow import WorkflowBuilder

# Create nodes
data_loader = DataLoaderNode("loader")
processor = YourCustomNode("processor")
analyzer = AnalysisNode("analyzer")

# Configure nodes
data_loader.configure(file_path="data.csv")
processor.configure(threshold=0.7, method='advanced')
analyzer.configure(metrics=['mean', 'std', 'max'])

# Build workflow
workflow = (
    WorkflowBuilder("analysis_pipeline")
    .add_node(data_loader)
    .add_node(processor)
    .add_node(analyzer)
    .connect("loader", "data", "processor", "input_data")
    .connect("processor", "processed_data", "analyzer", "signal_data")
    .build()
)

# Execute workflow
workflow.execute()
```

### Parameter Optimization

```python
from neuroworkflow.nodes.optimization import OptimizationNode

# Create optimization workflow
optimizer = OptimizationNode("optimizer")
optimizer.configure(
    method='bayesian',
    max_iterations=50,
    objective='minimize_error'
)

# Connect to your node for parameter optimization
optimization_workflow = (
    WorkflowBuilder("parameter_optimization")
    .add_node(data_loader)
    .add_node(processor)  # Your node with optimizable parameters
    .add_node(evaluator)
    .add_node(optimizer)
    # ... connections
    .build()
)
```

## Common Patterns and Examples

### Pattern 1: Data Transformation Node

```python
class DataTransformNode(Node):
    """Transform data from one format to another."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type='data_transform',
        description='Transform data between formats',

        parameters={
            'transform_type': ParameterDefinition(
                default_value='normalize',
                description='Type of transformation',
                constraints={'allowed_values': ['normalize', 'standardize', 'scale']}
            )
        },

        inputs={
            'input_data': PortDefinition(type=PortType.OBJECT, description='Data to transform')
        },

        outputs={
            'transformed_data': PortDefinition(type=PortType.OBJECT, description='Transformed data')
        }
    )

    def transform_data(self, input_data: Any) -> Dict[str, Any]:
        transform_type = self._parameters['transform_type']

        if transform_type == 'normalize':
            result = self._normalize(input_data)
        elif transform_type == 'standardize':
            result = self._standardize(input_data)
        elif transform_type == 'scale':
            result = self._scale(input_data)

        return {'transformed_data': result}
```

### Pattern 2: Analysis Node

```python
class SignalAnalysisNode(Node):
    """Analyze signal properties and characteristics."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type='signal_analysis',
        description='Analyze signal properties',

        parameters={
            'analysis_window': ParameterDefinition(
                default_value=1000,
                description='Analysis window size in samples'
            ),
            'metrics': ParameterDefinition(
                default_value=['power', 'frequency'],
                description='Metrics to calculate'
            )
        },

        inputs={
            'signal': PortDefinition(type=PortType.OBJECT, description='Input signal'),
            'sampling_rate': PortDefinition(type=PortType.FLOAT, description='Sampling rate')
        },

        outputs={
            'analysis_results': PortDefinition(type=PortType.DICT, description='Analysis results'),
            'summary_stats': PortDefinition(type=PortType.DICT, description='Summary statistics')
        }
    )

    def analyze_signal(self, signal: Any, sampling_rate: float) -> Dict[str, Any]:
        metrics = self._parameters['metrics']
        window_size = self._parameters['analysis_window']

        results = {}

        if 'power' in metrics:
            results['power_spectrum'] = self._calculate_power_spectrum(signal, sampling_rate)

        if 'frequency' in metrics:
            results['dominant_frequency'] = self._find_dominant_frequency(signal, sampling_rate)

        summary = self._generate_summary(results)

        return {
            'analysis_results': results,
            'summary_stats': summary
        }
```

### Pattern 3: File I/O Node

```python
class FileProcessorNode(Node):
    """Process data from files and save results."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type='file_processor',
        description='Process data from files',

        inputs={
            'input_file': PortDefinition(type=PortType.CSV_FILE, description='Input data file')
        },

        outputs={
            'processed_data': PortDefinition(type=PortType.OBJECT, description='Processed data'),
            'output_file': PortDefinition(type=PortType.CSV_FILE, description='Output file path')
        }
    )

    def load_and_process(self, input_file: str) -> Dict[str, Any]:
        # Load data
        data = pd.read_csv(input_file)

        # Process data
        processed = self._process_dataframe(data)

        # Save results
        output_file = f"processed_{os.path.basename(input_file)}"
        processed.to_csv(output_file, index=False)

        return {
            'processed_data': processed,
            'output_file': output_file
        }
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Port Connection Errors

**Problem**: "Port not found" or "Type mismatch" errors

**Solution**:

```python
# Check port definitions
print("Input ports:", list(node._input_ports.keys()))
print("Output ports:", list(node._output_ports.keys()))

# Check port types
input_port = node.get_input_port('data')
print("Port type:", input_port.data_type)
```

#### Issue 2: Parameter Validation Errors

**Problem**: Parameters not being validated correctly

**Solution**:

```python
# Add explicit parameter validation
def _validate_parameters(self):
    for name, value in self._parameters.items():
        param_def = self.NODE_DEFINITION.parameters.get(name)
        if param_def and hasattr(param_def, 'constraints'):
            self._validate_parameter_constraints(name, value, param_def.constraints)
```

#### Issue 3: Process Step Execution Errors

**Problem**: Methods not being called in the right order

**Solution**:

```python
# Debug process steps
def _define_process_steps(self):
    print(f"[{self.name}] Defining process steps...")

    self.add_process_step("step1", self.method1)
    print(f"[{self.name}] Added step1")

    self.add_process_step("step2", self.method2)
    print(f"[{self.name}] Added step2")

    print(f"[{self.name}] Total steps: {len(self._process_steps)}")
```

### Debugging Tips

1. **Add verbose logging**:

```python
def process_data(self, input_data: Any) -> Dict[str, Any]:
    if self._parameters.get('debug', False):
        print(f"[DEBUG] Input type: {type(input_data)}")
        print(f"[DEBUG] Input shape: {getattr(input_data, 'shape', 'N/A')}")
        print(f"[DEBUG] Parameters: {self._parameters}")

    # ... processing logic
```

2. **Use try-except blocks with detailed error messages**:

```python
try:
    result = self._complex_operation(data)
except Exception as e:
    error_msg = (
        f"Error in {self.name}.{self._complex_operation.__name__}: {e}\n"
        f"Input type: {type(data)}\n"
        f"Parameters: {self._parameters}\n"
        f"Node state: {self.__dict__}"
    )
    raise RuntimeError(error_msg) from e
```

3. **Implement state inspection methods**:

```python
def get_debug_info(self) -> Dict[str, Any]:
    """Get comprehensive debug information."""
    return {
        'name': self.name,
        'parameters': self._parameters,
        'input_ports': {name: port.value for name, port in self._input_ports.items()},
        'output_ports': {name: port.value for name, port in self._output_ports.items()},
        'process_steps': [step.name for step in self._process_steps],
        'cache_size': len(getattr(self, '_cache', {})),
        'validation_status': getattr(self, '_validation_status', 'unknown')
    }
```

## Conclusion

Creating custom nodes for NeuroWorkflow involves:

1. **Understanding the architecture** - nodes, ports, parameters, and workflows
2. **Following the template** - use the provided template as a starting point
3. **Implementing robust processing** - handle errors, validate inputs, log progress
4. **Testing thoroughly** - unit tests, integration tests, edge cases
5. **Documenting clearly** - help others understand and use your node

The template provided (`CustomNodeTemplate.py`) includes all these best practices and serves as a comprehensive starting point for your custom nodes. Modify it according to your specific needs while maintaining the core structure and patterns.
