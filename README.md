# NeuroWorkflow

A Python library for building and executing neural simulation workflows.

## Features

- Node-based workflow system for neural simulations
- Type-safe connections between workflow components
- Pre-built nodes for common neural simulation tasks
- Extensible architecture for custom nodes
- Parameter optimization support for tuning simulation parameters

## Documentation

- [NODE_SCHEMA.md](NODE_SCHEMA.md) - Comprehensive documentation of the node schema and parameter optimization features

## Current status

- The `src` folder contains the core functionality and sample nodes
- In the examples folder:
  - `simple_simulation.py` - Basic simulation example
  - `parameter_optimization.py` - Example of parameter optimization
  - `modular_optimization.py` - Example of modular parameter optimization with custom objective functions
  - `connected_optimization_workflow.py` - Example of a workflow with optimization at the workflow level
  - `properly_connected_workflow.py` - Example of a fully connected workflow with proper node connections
- In the notebooks folder:
  - `01_Basic_Simulation.ipynb` - Interactive example of basic simulation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
