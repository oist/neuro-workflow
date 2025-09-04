# Creating Custom Nodes in NeuroWorkflow

This tutorial provides a step-by-step guide to creating custom nodes for the NeuroWorkflow library. Custom nodes allow you to extend the framework with your own functionality, making it adaptable to a wide range of scientific workflows.

## Table of Contents

1. [Understanding Node Architecture](#understanding-node-architecture)
2. [Step 1: Planning Your Node](#step-1-planning-your-node)
3. [Step 2: Creating the Node Definition](#step-2-creating-the-node-definition)
4. [Step 3: Implementing Process Methods](#step-3-implementing-process-methods)
5. [Step 4: Defining Process Steps](#step-4-defining-process-steps)
6. [Step 5: Testing Your Node](#step-5-testing-your-node)
7. [Advanced Topics](#advanced-topics)
8. [Complete Example](#complete-example)

## Understanding Node Architecture

Before creating a custom node, it's important to understand the architecture of nodes in NeuroWorkflow:

- **Node Definition**: A schema that defines the node's type, parameters, inputs, outputs, and methods
- **Ports**: Input and output connections with type information
- **Parameters**: Configurable values that control the node's behavior
- **Process Steps**: Ordered execution steps that implement the node's functionality
- **Context**: A dictionary that passes data between process steps

## Step 1: Planning Your Node

Start by planning what your node will do:

1. **Purpose**: Define the specific task your node will perform
2. **Inputs**: What data will your node receive?
3. **Outputs**: What data will your node produce?
4. **Parameters**: What aspects of your node should be configurable?
5. **Process Steps**: What sequence of operations will your node perform?

### Example Planning

Let's plan a node that performs spike train analysis:

- **Purpose**: Analyze spike trains from neural simulations
- **Inputs**: Spike recorder data
- **Outputs**: Analysis results (firing rates, ISI histograms)
- **Parameters**: Analysis window, bin size
- **Process Steps**: 1) Extract spikes, 2) Calculate metrics, 3) Generate results

## Step 2: Creating the Node Definition

Create a new Python file for your node class. Start by defining the `NODE_DEFINITION` schema:

```python
from neuroworkflow import Node, NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType

class SpikeAnalysisNode(Node):
    """Node for analyzing spike trains from neural simulations."""
    
    NODE_DEFINITION = NodeDefinitionSchema(
        type='spike_analysis',  # Unique type identifier for this node
        description='Analyzes spike trains from neural simulations',
        
        # Define configurable parameters
        parameters={
            'time_window': ParameterDefinition(
                default_value=[0.0, 1000.0],  # [start_time, end_time] in ms
                description='Time window for analysis in milliseconds',
                constraints={'min_length': 2, 'max_length': 2}
            ),
            'bin_size': ParameterDefinition(
                default_value=10.0,  # ms
                description='Bin size for spike histograms in milliseconds',
                constraints={'min': 0.1, 'max': 1000.0}
            ),
            'metrics': ParameterDefinition(
                default_value=['rate', 'isi'],
                description='Metrics to calculate',
                constraints={'allowed_values': ['rate', 'isi', 'cv', 'fano']}
            )
        },
        
        # Define input ports
        inputs={
            'spike_data': PortDefinition(
                type=PortType.OBJECT,
                description='Spike recorder data from simulation'
            ),
            'neuron_ids': PortDefinition(
                type=PortType.LIST,
                description='List of neuron IDs to analyze',
                optional=True  # This input is optional
            )
        },
        
        # Define output ports
        outputs={
            'firing_rates': PortDefinition(
                type=PortType.DICT,
                description='Firing rates for each neuron'
            ),
            'isi_histograms': PortDefinition(
                type=PortType.DICT,
                description='Inter-spike interval histograms',
                optional=True  # This output may not always be produced
            ),
            'analysis_report': PortDefinition(
                type=PortType.DICT,
                description='Complete analysis report'
            )
        },
        
        # Define methods
        methods={
            'extract_spikes': MethodDefinition(
                description='Extract spikes from recorder data',
                inputs=['spike_data', 'neuron_ids'],
                outputs=['extracted_spikes']
            ),
            'calculate_metrics': MethodDefinition(
                description='Calculate analysis metrics',
                inputs=['extracted_spikes'],
                outputs=['firing_rates', 'isi_histograms']
            ),
            'generate_report': MethodDefinition(
                description='Generate complete analysis report',
                inputs=['firing_rates', 'isi_histograms'],
                outputs=['analysis_report']
            )
        }
    )
    
    # Node implementation will follow...
```

### Key Components of the Node Definition:

- **type**: A unique identifier for your node type
- **description**: A clear description of what your node does
- **parameters**: Configurable values with defaults and constraints
- **inputs**: Data that flows into your node
- **outputs**: Data that flows out of your node
- **methods**: The processing steps your node will perform

## Step 3: Implementing Process Methods

Next, implement the methods that will process your data. Each method should:

1. Accept inputs as defined in the method definition
2. Process the data
3. Return a dictionary with the outputs

```python
def extract_spikes(self, spike_data, neuron_ids=None):
    """Extract spikes from recorder data.
    
    Args:
        spike_data: Spike recorder data from simulation
        neuron_ids: Optional list of neuron IDs to analyze
        
    Returns:
        Dictionary with extracted spikes
    """
    # In a real implementation, you would extract spike times from the recorder
    # For this example, we'll create a simple simulation
    
    print(f"Extracting spikes from recorder data")
    
    # Get time window from parameters
    time_window = self._parameters['time_window']
    start_time, end_time = time_window
    
    # If neuron_ids is not provided, use all neurons in the data
    if neuron_ids is None:
        # In a real implementation, you would extract this from spike_data
        neuron_ids = list(range(1, 11))  # Simulate 10 neurons
        
    print(f"Analyzing {len(neuron_ids)} neurons in time window [{start_time}, {end_time}] ms")
    
    # Simulate extracted spikes (neuron_id -> list of spike times)
    import random
    extracted_spikes = {}
    for neuron_id in neuron_ids:
        # Simulate Poisson process with random rate between 5-15 Hz
        rate = random.uniform(5, 15)  # Hz
        # Convert to expected number of spikes in the time window
        expected_spikes = rate * (end_time - start_time) / 1000.0
        # Generate random spike times
        n_spikes = int(random.gauss(expected_spikes, expected_spikes**0.5))
        spike_times = sorted([random.uniform(start_time, end_time) for _ in range(max(0, n_spikes))])
        extracted_spikes[neuron_id] = spike_times
        
    return {'extracted_spikes': extracted_spikes}

def calculate_metrics(self, extracted_spikes):
    """Calculate analysis metrics.
    
    Args:
        extracted_spikes: Dictionary of extracted spikes (neuron_id -> spike times)
        
    Returns:
        Dictionary with calculated metrics
    """
    print(f"Calculating metrics: {self._parameters['metrics']}")
    
    # Get parameters
    time_window = self._parameters['time_window']
    start_time, end_time = time_window
    bin_size = self._parameters['bin_size']
    metrics = self._parameters['metrics']
    
    # Calculate firing rates
    firing_rates = {}
    for neuron_id, spike_times in extracted_spikes.items():
        # Calculate rate as spikes per second
        duration_sec = (end_time - start_time) / 1000.0
        rate = len(spike_times) / duration_sec if duration_sec > 0 else 0
        firing_rates[neuron_id] = rate
        
    # Calculate ISI histograms if requested
    isi_histograms = {}
    if 'isi' in metrics:
        for neuron_id, spike_times in extracted_spikes.items():
            if len(spike_times) >= 2:
                # Calculate inter-spike intervals
                isis = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
                
                # Create histogram
                import numpy as np
                max_isi = max(isis) if isis else bin_size
                bins = np.arange(0, max_isi + bin_size, bin_size)
                hist, edges = np.histogram(isis, bins=bins)
                
                isi_histograms[neuron_id] = {
                    'counts': hist.tolist(),
                    'bin_edges': edges.tolist(),
                    'mean_isi': np.mean(isis) if isis else 0,
                    'std_isi': np.std(isis) if isis else 0
                }
            else:
                isi_histograms[neuron_id] = {
                    'counts': [],
                    'bin_edges': [],
                    'mean_isi': 0,
                    'std_isi': 0
                }
    
    return {
        'firing_rates': firing_rates,
        'isi_histograms': isi_histograms if 'isi' in metrics else {}
    }

def generate_report(self, firing_rates, isi_histograms):
    """Generate complete analysis report.
    
    Args:
        firing_rates: Dictionary of firing rates
        isi_histograms: Dictionary of ISI histograms
        
    Returns:
        Dictionary with analysis report
    """
    print("Generating analysis report")
    
    # Create a comprehensive report
    report = {
        'parameters': {
            'time_window': self._parameters['time_window'],
            'bin_size': self._parameters['bin_size'],
            'metrics': self._parameters['metrics']
        },
        'summary': {
            'n_neurons': len(firing_rates),
            'mean_firing_rate': sum(firing_rates.values()) / len(firing_rates) if firing_rates else 0,
            'min_firing_rate': min(firing_rates.values()) if firing_rates else 0,
            'max_firing_rate': max(firing_rates.values()) if firing_rates else 0
        },
        'neuron_data': {
            neuron_id: {
                'firing_rate': firing_rates.get(neuron_id, 0),
                'isi_stats': isi_histograms.get(neuron_id, {})
            }
            for neuron_id in firing_rates.keys()
        }
    }
    
    return {'analysis_report': report}
```

## Step 4: Defining Process Steps

Now, implement the constructor and define the process steps that link your methods to the node definition:

```python
def __init__(self, name: str):
    """Initialize the SpikeAnalysisNode.
    
    Args:
        name: Name of the node
    """
    super().__init__(name)
    self._define_process_steps()
    
def _define_process_steps(self):
    """Define the process steps for this node."""
    # Add process steps in the order they should be executed
    self.add_process_step(
        "extract_spikes",
        self.extract_spikes,
        method_key="extract_spikes"
    )
    
    self.add_process_step(
        "calculate_metrics",
        self.calculate_metrics,
        method_key="calculate_metrics"
    )
    
    self.add_process_step(
        "generate_report",
        self.generate_report,
        method_key="generate_report"
    )
```

The `_define_process_steps` method links your implementation methods to the method definitions in `NODE_DEFINITION`. The `method_key` parameter tells the framework which method definition to use for each process step.

## Step 5: Testing Your Node

Create a simple script to test your node:

```python
from neuroworkflow import WorkflowBuilder
from your_module import SpikeAnalysisNode

# Create a spike analysis node
analysis_node = SpikeAnalysisNode("SpikeAnalyzer")
analysis_node.configure(
    time_window=[0.0, 2000.0],
    bin_size=5.0,
    metrics=['rate', 'isi']
)

# Create a simple workflow with just this node
workflow = WorkflowBuilder("test_workflow").add_node(analysis_node).build()

# Set input values directly (in a real workflow, these would come from connections)
analysis_node.get_input_port("spike_data").set_value({
    "format": "spike_recorder",
    "data": {
        "senders": [1, 2, 3, 1, 2, 1],
        "times": [10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    }
})

# Execute the workflow
workflow.execute()

# Get the results
report = analysis_node.get_output_port("analysis_report").value
print("\nAnalysis Report Summary:")
print(f"Number of neurons: {report['summary']['n_neurons']}")
print(f"Mean firing rate: {report['summary']['mean_firing_rate']:.2f} Hz")
```

## Common Pitfalls and Best Practices

When creating custom nodes, be aware of these common issues and follow these best practices:

### 1. Class Definition Structure

Always define your node class in a single, cohesive block:

```python
class MyCustomNode(Node):
    """Node documentation."""
    
    NODE_DEFINITION = NodeDefinitionSchema(...)
    
    def __init__(self, name):
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self):
        # Define process steps here
        
    def process_method1(self, input1, input2):
        # Method implementation
        
    def process_method2(self, input1):
        # Method implementation
```

Avoid defining methods separately and attaching them to the class later, as this can lead to issues with method binding and inheritance.

### 2. Variable Scope in Notebooks

When using nodes in Jupyter notebooks:

- Be careful with variable scope between cells
- Don't rely on variables defined in previous cells unless necessary
- Make each cell as self-contained as possible
- Use `globals()` to check if variables exist before using them

Example of defensive coding in notebooks:

```python
# Check if the node exists before using it
if 'my_node' not in globals():
    my_node = MyCustomNode("NewNode")
    my_node.configure(param1=value1, param2=value2)
```

### 3. Error Handling

Implement robust error handling in your node methods:

```python
def process_data(self, data):
    try:
        # Process data
        result = self._complex_calculation(data)
        return {'output': result}
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        # Return a default or partial result
        return {'output': None, 'error': str(e)}
```

### 4. Documentation

Document your node thoroughly:

- Class docstring explaining the node's purpose
- Method docstrings with parameter and return value descriptions
- Examples of how to use the node
- Any assumptions or limitations

## Advanced Topics

### 1. Type Validation and Conversion

You can implement custom type validation in your methods:

```python
def extract_spikes(self, spike_data, neuron_ids=None):
    # Validate input types
    if not isinstance(spike_data, dict):
        raise TypeError(f"Expected spike_data to be a dictionary, got {type(spike_data)}")
    
    if neuron_ids is not None and not isinstance(neuron_ids, list):
        # Convert to list if possible
        try:
            neuron_ids = list(neuron_ids)
        except:
            raise TypeError(f"neuron_ids must be convertible to a list")
    
    # Continue with processing...
```

### 2. Progress Reporting

For long-running operations, you can add progress reporting:

```python
def calculate_metrics(self, extracted_spikes):
    print(f"Calculating metrics: {self._parameters['metrics']}")
    
    # Get total work to be done
    total_neurons = len(extracted_spikes)
    
    # Process with progress updates
    for i, (neuron_id, spike_times) in enumerate(extracted_spikes.items()):
        # Report progress every 10%
        if i % max(1, total_neurons // 10) == 0:
            print(f"Progress: {i/total_neurons*100:.1f}% ({i}/{total_neurons} neurons)")
        
        # Process this neuron...
```

### 3. Caching Results

For computationally intensive operations, you can implement caching:

```python
def calculate_metrics(self, extracted_spikes):
    # Create a cache key based on inputs and parameters
    import hashlib
    import json
    
    # Create a deterministic representation of inputs and parameters
    cache_data = {
        'spikes': {k: v for k, v in extracted_spikes.items()},
        'params': {
            'time_window': self._parameters['time_window'],
            'bin_size': self._parameters['bin_size'],
            'metrics': sorted(self._parameters['metrics'])
        }
    }
    
    cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    # Check if we have cached results
    if hasattr(self, '_cache') and cache_key in self._cache:
        print("Using cached results")
        return self._cache[cache_key]
    
    # Calculate metrics as before...
    results = {
        'firing_rates': firing_rates,
        'isi_histograms': isi_histograms
    }
    
    # Cache the results
    if not hasattr(self, '_cache'):
        self._cache = {}
    self._cache[cache_key] = results
    
    return results
```

## Building Workflows with Custom Nodes

Once you've created custom nodes, you can use them in workflows:

### 1. Creating a Workflow with Custom Nodes

```python
from neuroworkflow import WorkflowBuilder
from your_module import CustomNode1, CustomNode2

# Create instances of your custom nodes
node1 = CustomNode1("FirstNode")
node1.configure(param1=value1, param2=value2)

node2 = CustomNode2("SecondNode")
node2.configure(param1=value1, param2=value2)

# Create a workflow builder
workflow_builder = WorkflowBuilder("my_workflow")

# Add nodes to the workflow
workflow_builder.add_node(node1)
workflow_builder.add_node(node2)

# Connect nodes (output of node1 to input of node2)
workflow_builder.connect(
    "FirstNode", "output_port_name", 
    "SecondNode", "input_port_name"
)

# Build the workflow
workflow = workflow_builder.build()
```

### 2. Executing the Workflow

```python
# Set input values for the first node
node1.get_input_port("input_port_name").set_value(input_data)

# Execute the workflow
success = workflow.execute()

if success:
    # Get results from the last node
    result = node2.get_output_port("output_port_name").value
    print("Workflow execution completed successfully!")
    print(f"Result: {result}")
else:
    print("Workflow execution failed!")
```

### 3. Visualizing the Workflow

You can visualize your workflow using NetworkX:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes
for node_name in workflow.get_node_names():
    G.add_node(node_name)

# Add edges
for connection in workflow.get_connections():
    source_node = connection["source_node"]
    target_node = connection["target_node"]
    G.add_edge(
        source_node, 
        target_node, 
        label=f"{connection['source_port']} → {connection['target_port']}"
    )

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(
    G, pos, with_labels=True, 
    node_color="lightblue", 
    node_size=2000, 
    font_size=10, 
    font_weight="bold"
)

# Add edge labels
edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Workflow Structure")
plt.axis("off")
plt.tight_layout()
plt.show()
```

### 4. Saving and Loading Workflows

You can save your workflow configuration for later use:

```python
import json
import pickle

# Save workflow configuration as JSON
config = workflow.to_dict()
with open("workflow_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Save the entire workflow object using pickle
with open("workflow.pkl", "wb") as f:
    pickle.dump(workflow, f)

# Load the workflow later
with open("workflow.pkl", "rb") as f:
    loaded_workflow = pickle.load(f)
```

## Complete Example

Here's the complete implementation of our `SpikeAnalysisNode`:

```python
from neuroworkflow import Node, NodeDefinitionSchema, PortDefinition, ParameterDefinition, MethodDefinition
from neuroworkflow.core.port import PortType
import random
import numpy as np

class SpikeAnalysisNode(Node):
    """Node for analyzing spike trains from neural simulations."""
    
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
                constraints={'min': 0.1, 'max': 1000.0}
            ),
            'metrics': ParameterDefinition(
                default_value=['rate', 'isi'],
                description='Metrics to calculate',
                constraints={'allowed_values': ['rate', 'isi', 'cv', 'fano']}
            )
        },
        
        inputs={
            'spike_data': PortDefinition(
                type=PortType.OBJECT,
                description='Spike recorder data from simulation'
            ),
            'neuron_ids': PortDefinition(
                type=PortType.LIST,
                description='List of neuron IDs to analyze',
                optional=True
            )
        },
        
        outputs={
            'firing_rates': PortDefinition(
                type=PortType.DICT,
                description='Firing rates for each neuron'
            ),
            'isi_histograms': PortDefinition(
                type=PortType.DICT,
                description='Inter-spike interval histograms',
                optional=True
            ),
            'analysis_report': PortDefinition(
                type=PortType.DICT,
                description='Complete analysis report'
            )
        },
        
        methods={
            'extract_spikes': MethodDefinition(
                description='Extract spikes from recorder data',
                inputs=['spike_data', 'neuron_ids'],
                outputs=['extracted_spikes']
            ),
            'calculate_metrics': MethodDefinition(
                description='Calculate analysis metrics',
                inputs=['extracted_spikes'],
                outputs=['firing_rates', 'isi_histograms']
            ),
            'generate_report': MethodDefinition(
                description='Generate complete analysis report',
                inputs=['firing_rates', 'isi_histograms'],
                outputs=['analysis_report']
            )
        }
    )
    
    def __init__(self, name: str):
        """Initialize the SpikeAnalysisNode."""
        super().__init__(name)
        self._define_process_steps()
    
    def _define_process_steps(self):
        """Define the process steps for this node."""
        self.add_process_step(
            "extract_spikes",
            self.extract_spikes,
            method_key="extract_spikes"
        )
        
        self.add_process_step(
            "calculate_metrics",
            self.calculate_metrics,
            method_key="calculate_metrics"
        )
        
        self.add_process_step(
            "generate_report",
            self.generate_report,
            method_key="generate_report"
        )
    
    def extract_spikes(self, spike_data, neuron_ids=None):
        """Extract spikes from recorder data."""
        print(f"Extracting spikes from recorder data")
        
        # Get time window from parameters
        time_window = self._parameters['time_window']
        start_time, end_time = time_window
        
        # If neuron_ids is not provided, use all neurons in the data
        if neuron_ids is None:
            neuron_ids = list(range(1, 11))  # Simulate 10 neurons
            
        print(f"Analyzing {len(neuron_ids)} neurons in time window [{start_time}, {end_time}] ms")
        
        # Simulate extracted spikes (neuron_id -> list of spike times)
        extracted_spikes = {}
        for neuron_id in neuron_ids:
            # Simulate Poisson process with random rate between 5-15 Hz
            rate = random.uniform(5, 15)  # Hz
            # Convert to expected number of spikes in the time window
            expected_spikes = rate * (end_time - start_time) / 1000.0
            # Generate random spike times
            n_spikes = int(random.gauss(expected_spikes, expected_spikes**0.5))
            spike_times = sorted([random.uniform(start_time, end_time) for _ in range(max(0, n_spikes))])
            extracted_spikes[neuron_id] = spike_times
            
        return {'extracted_spikes': extracted_spikes}
    
    def calculate_metrics(self, extracted_spikes):
        """Calculate analysis metrics."""
        print(f"Calculating metrics: {self._parameters['metrics']}")
        
        # Get parameters
        time_window = self._parameters['time_window']
        start_time, end_time = time_window
        bin_size = self._parameters['bin_size']
        metrics = self._parameters['metrics']
        
        # Calculate firing rates
        firing_rates = {}
        for neuron_id, spike_times in extracted_spikes.items():
            # Calculate rate as spikes per second
            duration_sec = (end_time - start_time) / 1000.0
            rate = len(spike_times) / duration_sec if duration_sec > 0 else 0
            firing_rates[neuron_id] = rate
            
        # Calculate ISI histograms if requested
        isi_histograms = {}
        if 'isi' in metrics:
            for neuron_id, spike_times in extracted_spikes.items():
                if len(spike_times) >= 2:
                    # Calculate inter-spike intervals
                    isis = [spike_times[i+1] - spike_times[i] for i in range(len(spike_times)-1)]
                    
                    # Create histogram
                    max_isi = max(isis) if isis else bin_size
                    bins = np.arange(0, max_isi + bin_size, bin_size)
                    hist, edges = np.histogram(isis, bins=bins)
                    
                    isi_histograms[neuron_id] = {
                        'counts': hist.tolist(),
                        'bin_edges': edges.tolist(),
                        'mean_isi': np.mean(isis) if isis else 0,
                        'std_isi': np.std(isis) if isis else 0
                    }
                else:
                    isi_histograms[neuron_id] = {
                        'counts': [],
                        'bin_edges': [],
                        'mean_isi': 0,
                        'std_isi': 0
                    }
        
        return {
            'firing_rates': firing_rates,
            'isi_histograms': isi_histograms if 'isi' in metrics else {}
        }
    
    def generate_report(self, firing_rates, isi_histograms):
        """Generate complete analysis report."""
        print("Generating analysis report")
        
        # Create a comprehensive report
        report = {
            'parameters': {
                'time_window': self._parameters['time_window'],
                'bin_size': self._parameters['bin_size'],
                'metrics': self._parameters['metrics']
            },
            'summary': {
                'n_neurons': len(firing_rates),
                'mean_firing_rate': sum(firing_rates.values()) / len(firing_rates) if firing_rates else 0,
                'min_firing_rate': min(firing_rates.values()) if firing_rates else 0,
                'max_firing_rate': max(firing_rates.values()) if firing_rates else 0
            },
            'neuron_data': {
                neuron_id: {
                    'firing_rate': firing_rates.get(neuron_id, 0),
                    'isi_stats': isi_histograms.get(neuron_id, {})
                }
                for neuron_id in firing_rates.keys()
            }
        }
        
        return {'analysis_report': report}
```

This example demonstrates a complete, functional node that can be integrated into any NeuroWorkflow. By following these steps, you can create custom nodes for your specific scientific needs.