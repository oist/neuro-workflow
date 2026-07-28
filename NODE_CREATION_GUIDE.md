# Node Creation Guide

A living reference for creating NeuroWorkflow nodes from existing brain models and tools.
Update the stage list and semantic guides as the library grows.

---

## Core Principles

1. **One node = one scientific operation at one stage.** Not one GitHub repo, not one line of code.
2. **Stage defines role.** The stage tells an agent *what this node does* in the modeling pipeline, even without reading the implementation.
3. **Tool defines context.** The tool/simulator tells an agent *what dependency is required* and what idioms to expect.
4. **Descriptions are for agents too.** Write `description`, port descriptions, and parameter descriptions as if an AI will read them to decide whether to use this node — because it will.
5. **Flexibility within stages.** Port names and types are not mandated across nodes. The stage semantic guide describes *what should flow*, not the exact variable names.

---

## Required Metadata Fields

Every node **must** declare these fields in `NODE_DEFINITION`:

```python
NODE_DEFINITION = NodeDefinitionSchema(
    type="unique_snake_case_identifier",   # globally unique
    stage="population",                    # see stage list below
    tool="NEST",                           # simulator, library, or "custom"
    model_source="https://github.com/...", # origin of the model/algorithm
    description="One clear sentence describing what this node does scientifically.",
    parameters={...},
    inputs={...},
    outputs={...},
    methods={...},
)
```

- `stage` — must match one of the stages listed below (or propose a new one)
- `tool` — the simulator or Python library this node wraps (e.g. `NEST`, `TVB`, `Brian2`, `NEURON`, `SNNBuilder`, `SciPy`, `custom`)
- `model_source` — URL to the paper, GitHub repo, or documentation this node implements

---

## Stage List

Stages represent steps in the brain modeling process. The list grows over time.

### `database`
Query an external reference database and bring its data into the workflow. These
are the entry points that read from remote neuroscience catalogs, as opposed to
`io` nodes that read/write the local filesystem.
- Typical inputs: none (source nodes) — query is set via parameters
- Typical outputs: fetched records (list of dicts) plus a fetch-envelope dict
- Parameters: query/search terms, result `limit`/`offset`, optional API key, enrichment toggle
- Examples: list CBS (RIKEN) resources, fetch Brain/MINDS Beyond human MRI datasets

There are two ways to reach a catalog, and a new node should pick deliberately:

- **Direct** (`neuroworkflow.utils.remote_catalogs`) — one client per upstream API,
  queried live. Always current, but slow, one source per node, and it fails when the
  upstream does. `CBSQueryNode` and `BMBHumanQueryNode` work this way.
- **Via mdb** (`neuroworkflow.utils.mdb_client`) — queries a running
  [bm_mindsdb](https://github.com/oist/bm_mindsdb) service, which keeps a synced copy of
  the remote catalogs plus an index of local BIDS trees. Fast, searchable across every
  source at once, reproducible between runs, and the only route to local datasets.
  Needs mdb reachable at `MDB_BASE_URL` and is only as current as its last sync.
  `MDBCatalogSearchNode`, `MDBCatalogLookupNode` and `MDBLocalCatalogNode` work this way.

Either way, keep the envelope contract: never raise out of the node. Return the records
plus a `metadata` dict carrying `status`, `count` and — on failure — `error`, so a dead
network degrades the run instead of aborting it.

### `io`
Load and save data between the filesystem and the workflow.
- Typical inputs: file paths (connectomes, time series, imaging data)
- Typical outputs: in-memory data structures (arrays, dataframes, objects)
- Parameters: file paths, format options, subject/session IDs
- Examples: load DTI connectome, load fMRI time series, export simulation results

### `neuron`
Define a single neuron model — its dynamics, morphology, and intrinsic parameters.
- Typical inputs: configuration dicts or file paths
- Typical outputs: a neuron model object or parameter set
- Parameters: membrane time constant, firing threshold, adaptation, ion channels
- Examples: IAF neuron definition, Hodgkin-Huxley cell, morphologically detailed cell

### `population`
Define a group of neurons of the same or mixed types.
- Typical inputs: neuron model definition
- Typical outputs: population object (group of cells with shared properties)
- Parameters: number of cells, cell type distribution, spatial layout
- Examples: excitatory population, inhibitory interneuron pool, cortical column layer

### `synapse`
Define synaptic connection rules, dynamics, and plasticity.
- Typical inputs: pre/post population definitions
- Typical outputs: synapse specification object
- Parameters: synaptic weight, delay, receptor type, plasticity rule (STDP, etc.)
- Examples: AMPA synapse, GABA_B synapse, STDP rule, short-term depression

### `connectivity`
Wire populations together into a network using a connectivity rule or matrix.
- Typical inputs: populations, synapse specs, optional connectome data
- Typical outputs: connected network or adjacency structure
- Parameters: connection probability, weight distribution, topology (random, small-world, data-driven)
- Examples: random connectivity, structural connectome-based wiring, topographic map

### `stimulus`
Design external input signals delivered to the network.
- Typical inputs: population or target definition
- Typical outputs: stimulus object or signal array
- Parameters: amplitude, frequency, duration, waveform type, target region
- Examples: DC current injection, Poisson spike train, sinusoidal drive, sensory input protocol

### `simulation`
Execute the model and record its activity.
- Typical inputs: network (with connectivity and stimulus), simulation config
- Typical outputs: raw recorded data (spikes, voltages, LFP, BOLD)
- Parameters: simulation duration, time step (dt), recording devices, random seed
- Examples: NEST simulation run, TVB simulator, Brian2 run, NEURON simulation

### `analysis`
Compute scientific metrics from raw simulation or experimental data.
- Typical inputs: raw recorded data
- Typical outputs: derived metrics, statistics, figures
- Parameters: time window, bin size, metric selection, frequency bands
- Examples: firing rate calculation, power spectrum, spike synchrony, BOLD signal GLM

### `optimization`
Search or fit parameters to match a target or objective.
- Typical inputs: any stage node's parameters + objective metric
- Typical outputs: optimized parameter set, optimization trajectory
- Parameters: algorithm (Bayesian, evolutionary, grid), budget, objective function
- Examples: Bayesian optimization of synaptic weights, fitting to experimental firing rates

---

## Proposing a New Stage

The stage list above is a starting point, not a closed set. If a model from a source repository does not map cleanly to any existing stage, **propose a new stage** rather than forcing a bad fit.

A new stage is warranted when:
- The scientific operation is meaningfully distinct from all existing stages
- It would apply to multiple models or tools (not just one specific script)
- A neuroscientist would naturally describe it as a separate step in the modeling process

To propose a new stage:
1. Pick a short lowercase name (e.g. `morphology`, `plasticity`, `decoding`)
2. Write a one-paragraph semantic guide following the format above (role, typical in/out, typical parameters, examples)
3. Add it to this file in the appropriate position in the pipeline order
4. Create the folder `src/neuroworkflow/nodes/<new_stage>/` with an `__init__.py`

When in doubt, propose and document — a well-described new stage is better than a misclassified node.

---

## Granularity Rules

| Too fine | Too coarse | Right |
|---|---|---|
| `SetLeakConductance` | `RunFullBGModel` | `NESTStriatumPopulationNode` |
| `ApplyThreshold` | `BuildAndSimulateNetwork` | `TVBEpileptorNode` |
| `LoadOneFile` | `LoadAllSubjects` | `VNMLoadSubjectTimeSeriesNode` |

**Rule of thumb:** A node is the right size if a neuroscientist would describe it in one sentence and it produces a self-contained scientific object that another node can consume.

---

## Process: GitHub Model → NeuroWorkflow Node

1. **Read the model.** Understand what it computes, what parameters it exposes, what it produces.
2. **Identify the stage.** Which step of brain modeling does this represent?
3. **Identify the tool.** What simulator or library does it depend on?
4. **Define the scientific interface.** What are the meaningful inputs/outputs from a neuroscience perspective — not implementation details.
5. **Set parameters.** Extract every scientifically meaningful parameter with its default, range, and description.
6. **Mark optimizable params.** Which parameters would a researcher want to tune or fit?
7. **Write the node class.** Use `/create-node` skill as scaffold.
8. **Write for agents.** Descriptions must be specific enough that an AI can decide whether to connect this node to another without reading the implementation.

---

## Writing Descriptions for Agent Readability

Bad (too vague):
```python
description="Runs the simulation."
```

Good (scientifically specific):
```python
description="Executes a NEST spiking network simulation and records spike times and membrane voltages from designated populations."
```

Bad port description:
```python
"data": PortDefinition(type=PortType.OBJECT, description="Data")
```

Good port description:
```python
"spike_recorder": PortDefinition(type=PortType.OBJECT, description="NEST spike recorder device containing spike times and neuron IDs for all recorded populations")
```

---

## Output File Convention

Nodes that write files to disk (plots, CSVs, HDF5, spike recordings, simulator output directories) must follow this pattern:

```python
import os

def my_method(self, ...) -> Dict[str, Any]:
    # 1. Context first, node parameter as fallback
    data_path = self._context.get("results_path", self._parameters["data_path"])

    # 2. Create the directory BEFORE any simulator initialization
    os.makedirs(data_path, exist_ok=True)

    # 3. Pass the path to the simulator after initialization
    #    (some simulators require absolute paths — check their docs)
    simulator.setup(output_path=data_path)
```

**Why this order matters:** The output directory must exist before the simulator tries to write to it. Reading from context first allows a single `results_path` set at the workflow level to propagate automatically to all nodes, without each node needing to independently configure the same path.

`WorkflowBuilder` defaults `results_path` to `"results/"` (relative to the working directory where `wf.build()` is called), and creates that folder automatically at build time. Users can set `results_path` to any path in the workflow context — the node always reads from context, so it adapts automatically. Nodes that only pass data in memory do not need to follow this convention.

---

## Checklist Before Committing a Node

- [ ] `stage` field is set and matches the stage list
- [ ] `tool` field identifies the simulator or library
- [ ] `model_source` URL is valid and points to the origin
- [ ] `description` is one clear scientific sentence
- [ ] All parameters have `description` and `default_value`
- [ ] Scientifically tunable parameters have `optimizable=True` and `optimization_range`
- [ ] `is_objective` / `objective_range` are only on `ParameterDefinition`, never on `PortDefinition`
- [ ] All port descriptions are specific enough for an agent to understand the data
- [ ] NEST nodes: `nest.ResetKernel()` is inside a process step method, not at import or `__init__`
- [ ] Nodes that write files use `self._context.get("results_path", "results/")` as the output directory
- [ ] Node is placed in the correct `src/neuroworkflow/nodes/<stage>/` folder
- [ ] Import added to the stage `__init__.py`
