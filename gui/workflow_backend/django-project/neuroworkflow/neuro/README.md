# NeuroWorkflow

A Python library for building and executing neural simulation workflows.

## Features

- Node-based workflow system for neural simulations
- Type-safe connections between workflow components
- Pre-built nodes for common neural simulation tasks
- Extensible architecture for custom nodes
- Command-line interface for workflow execution

## Installation

```bash
# Basic installation
pip install neuroworkflow

# With NEST integration
pip install neuroworkflow[nest]

# With visualization tools
pip install neuroworkflow[visualization]

# For development
pip install neuroworkflow[dev]
```

## Quick Start

```python
from neuroworkflow import WorkflowBuilder
from neuroworkflow.nodes.network import BuildSonataNetworkNode
from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode

# Create nodes
build_network = BuildSonataNetworkNode("NetworkBuilder")
build_network.configure(sonata_path="/path/to/config")

simulate = SimulateSonataNetworkNode("Simulator")
simulate.configure(simulation_time=1000.0)

# Create and execute workflow
workflow = (
    WorkflowBuilder("my_simulation")
        .add_node(build_network)
        .add_node(simulate)
        .connect("NetworkBuilder", "sonata_net", "Simulator", "sonata_net")
        .connect("NetworkBuilder", "node_collections", "Simulator", "node_collections")
        .build()
)

workflow.execute()
```

## Creating Custom Nodes

You can create custom nodes by subclassing the `Node` class. For a detailed step-by-step tutorial, see [Creating Custom Nodes](docs/creating_custom_nodes.md).

Here's a simple example:

```python
from neuroworkflow import Node, NodeDefinitionSchema, PortDefinition, PortType

class MyCustomNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type='custom_node',
        description='My custom node',
        inputs={
            'input_data': PortDefinition(
                type=PortType.OBJECT,
                description='Input data'
            )
        },
        outputs={
            'output_data': PortDefinition(
                type=PortType.OBJECT,
                description='Output data'
            )
        },
        methods={
            'process_data': {
                'description': 'Process the input data',
                'inputs': ['input_data'],
                'outputs': ['output_data']
            }
        }
    )
    
    def __init__(self, name):
        super().__init__(name)
        self._define_process_steps()
        
    def _define_process_steps(self):
        self.add_process_step(
            "process_data",
            self.process_data,
            method_key="process_data"
        )
        
    def process_data(self, input_data):
        # Process the data
        output_data = input_data  # Replace with actual processing
        return {"output_data": output_data}
```

For a complete example, see the [SpikeAnalysisNode](src/neuroworkflow/nodes/analysis/spike_analysis.py) implementation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.