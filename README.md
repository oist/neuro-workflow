# NeuroWorkflow

A Python library for building and executing neural simulation workflows.

## Features

- Node-based workflow system for neural simulations
- Type-safe connections between workflow components
- Pre-built nodes for common neural simulation tasks
- Extensible architecture for custom nodes
- Parameter optimization support for tuning simulation parameters

## Current status

- The `src` folder contains the core functionality and sample nodes
- In the examples folder:
  - `sonata_simulation.py` - Basic simulation example
  - `neuron_optimization.py` - Example of parameter optimization (not yet completed, but running with some bugs)
  - `epilepsy_rs.py` - Example of epileptic resting state using the virtual brain TVB
- In the notebooks folder:
  - `01_Basic_Simulation.ipynb` - Interactive example of basic simulation
  - `epilepsy_rs.ipynb` - Interactive example of epileptic resting state using the virtual brain TVB
  - `SNNbuilder_example1.ipynb` - Interactive example of Spiking Neural Network building using SNNbuilder custom nodes

## License

This project is licensed under the MIT License - see the LICENSE file for details.
