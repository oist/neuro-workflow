---
name: create-node
description: Create a new NeuroWorkflow node (NEST / TVB / Brian2 / custom computation) following the NODE_CREATION_GUIDE.md conventions, then generate the file. Use when the user wants to build or add a workflow node.
---

# Create NeuroWorkflow Node

Guide the user through creating a new node following the NODE_CREATION_GUIDE.md conventions, then generate the file.

## Environment Detection (do this first, before anything else)

Determine the output path using this priority order:

1. **User-specified folder** — if the user has indicated a folder (e.g. `my_nodes/`, `./nodes/`, or any path), always use it. The user's choice overrides everything.
2. **Repo sandbox** — if `src/neuroworkflow/nodes/sandbox/` exists relative to the current directory and the user has not specified a folder, use it as the default output.
3. **`./nodes/`** — fallback if neither of the above applies. Create the folder if it does not exist.

Similarly, determine the source code path:
- Use the folder the user indicates as their existing source code.
- If none indicated and repo structure exists, browse `src/neuroworkflow/nodes/` for reference.
- Otherwise skip browsing existing nodes.

Read `NODE_CREATION_GUIDE.md` from:
- The repo root if repo structure is detected.
- The current directory if present in standalone mode.
- Skip stage guidance if not found anywhere.

Ask the user for the output folder and source code folder **together with the Step 1 questions** (not as a separate message) if they have not already been specified.

**Always report the full absolute path of every file created at the end of the skill.**

---

Before starting, read `NODE_CREATION_GUIDE.md` to apply the current stage list and semantic guides. If the node will have multiple methods or is structurally complex, also browse the existing nodes folder for a comparable existing node to use as a reference.

---

## Design philosophy — composability across models and simulators

Every node you build may one day be connected to a node from a completely different model, simulator, brain region, or scientific team. A NEST basal ganglia node may feed into a TVB cortical node. A single-neuron node may feed into an ensemble node that wraps an entire workflow. A workflow (model/project) is itself a node — its inputs and outputs are its public scientific API.

This means nodes must be designed not just to work today, but to be **connectable by an AI agent tomorrow** without human intervention. An AI connecting two nodes will read port descriptions to determine compatibility — so descriptions must carry scientific meaning, not just data shape.

**Keep this in mind when designing every port:**

- Write port descriptions with **scientific units and semantics**: `"Mean firing rate per nucleus in Hz, dict keyed by population name"` not `"firing rate dict"`.
- Prefer `DICT` with well-named keys over opaque `OBJECT` — a dict's contents are inspectable; an object is a black box.
- Use standard neuroscience quantities in key names (`firing_rate_hz`, `membrane_potential_mv`, `spike_times_ms`) so ports from different simulators can be matched semantically.
- Ask: *"Could an AI agent determine whether this output is scientifically compatible with an input from a node it has never seen?"* If not, the description needs more precision.
- A node that produces a clean, well-described `DICT` output is more composable than one that returns a simulator-specific object, because any downstream node — regardless of simulator — can read from it.

---

## Step 0 — Audit existing nodes first

Before writing anything new, search the existing nodes folder (determined in Environment Detection above) to understand what already exists.

Ask yourself:

1. **Does an existing node already cover this?** If so, stop — point the user to it.
2. **Could an existing node be made configurable to cover both the old and new case?**
   - If yes, propose extending it with a `model_type`-style string parameter (see *Configurable node pattern* below) rather than creating a new node.
   - Only create a new node if the scientific interface (inputs/outputs) is meaningfully different, not just the parameters.
3. **Is a family of similar nodes being requested at once?** (e.g., from a notebook or paper)
   - Map all requested operations to candidate nodes first.
   - Identify which can share a single configurable node before deciding how many files to create.

When the source is a **Jupyter notebook** (.ipynb):
- Read the notebook cell by cell.
- Each coherent group of cells that produces a self-contained scientific object is a candidate node.
- Check existing nodes for each candidate before creating a new one.
- The notebook's parameter values become `default_value` in the node's `ParameterDefinition`.

---

## Step 1 — Gather requirements

Ask the user (all at once, in a single message) for the **required** fields. The optional fields below can be filled in later or inferred from the implementation — do not block on them.

### Required (always ask)

1. **Node name** — CamelCase class name (e.g. `NESTStriatumPopulationNode`)
2. **Stage** — determine the stage by reasoning from the node's role, not by matching against the list:
   - What does this node **consume**? (What must already exist for it to run?)
   - What does it **produce**? (What self-contained scientific object becomes available after it runs?)
   - Is that output something another node can consume without knowing how it was built?
   - Is this step conceptually independent from adjacent steps — i.e., could a different tool produce the same output and be swapped in?
   
   If yes to the last two, it is a distinct stage. Check `NODE_CREATION_GUIDE.md` to see if an existing stage matches. If no existing stage fits cleanly, propose a new one — a well-described new stage is better than a forced fit.
3. **Tool** — simulator or library (e.g. `NEST`, `TVB`, `Brian2`, `NEURON`, `SciPy`, `custom`)
4. **Model source** — URL to the GitHub repo, paper, or documentation this node implements
5. **Description** — one scientifically specific sentence (written for an AI agent to understand, not just a human)
6. **Parameters** — for each: `name`, `default_value`, `description`
7. **Inputs** — for each: `name`, `PortType`, description of the data
8. **Outputs** — for each: `name`, `PortType`, description of the data
9. **Methods** — processing method names, which inputs/outputs each uses, and what it does. If multiple methods are needed, list them in execution order (each method's outputs can be used as inputs to later methods).

Available PortTypes: `ANY`, `INT`, `FLOAT`, `STR`, `BOOL`, `LIST`, `DICT`, `OBJECT`, `FILE_PATH`, `CSV_FILE`, `JSON_FILE`, `PICKLE_FILE`, `NUMPY_FILE`, `HDF5_FILE`

### Choosing the right PortType

**Decision rule:** Ask "Could this data be serialized to JSON without losing information?"
- Yes → use `DICT`, `LIST`, `STR`, `FLOAT`, `BOOL`, or `INT`
- No (simulator object, model instance, NodeCollection) → use `OBJECT`

**Type-to-data mapping:**

| Data | PortType |
|------|----------|
| NEST NodeCollection, TVB model/connectivity object, any simulator-specific instance | `OBJECT` |
| Parameters, metadata, biological properties, summaries | `DICT` |
| Neuron/synapse/population name, generated code, file path | `STR` |
| Spike times, position arrays, result collections | `LIST` |
| Simulation time, weight, rate, threshold | `FLOAT` |
| Execution flag, enable/disable toggle | `BOOL` |

**Warning — `OBJECT` overuse:** `OBJECT` is compatible with every other port type in the framework, so it never causes a connection error. This makes it tempting to use everywhere — but it hides what the port actually carries and makes workflows harder to understand. Treat `OBJECT` as a last resort, only when the data genuinely cannot be expressed as a simpler type.

### Optional / inferred (fill in when known)

- **Parameter constraints** (`min`/`max` or `allowed_values`) — add when the scientific meaning implies a valid range or a fixed set of options. Leave out if the range is open-ended or unknown.
- **Optimizable parameters** (`optimizable=True`, `optimization_range`) — add for parameters a researcher would tune or fit to data.
- **Do not add a parameter to hold an optimization target.** A target (a measured output steered into a range) is declared on the `NW_Optimization` node's `objectives`, not on the model node. `is_objective` / `objective_range` are only a hint for a genuine set-point parameter the node itself reads, and **belong only on `ParameterDefinition`, never on `PortDefinition` — placing them on a port raises `TypeError` at class definition time.**
- **Optional inputs** (`optional=True`) — always infer this from the code, do not ask. If the method uses the input unconditionally, the port is required (default). If the method signature has `= None` for that parameter, or the body guards it with `if input is None:`, mark the port `optional=True`. The rule: used unconditionally → required; guarded or defaulted to None → optional.

If the user provides existing code or a GitHub repo, extract as much of the optional metadata as possible from the implementation rather than asking for it.

---

## Configurable node pattern

When a family of related models exists (e.g. several TVB neural mass models, several NEST neuron types), prefer **one configurable node** over separate nodes per model.

**Use this pattern when:**
- Multiple models share the same scientific interface (same inputs/outputs/stage)
- The only differences are parameter values and which class gets instantiated
- Existing nodes cover one model and a new simulation adds another

**How to implement:**
- Add a `model_type` string parameter with `constraints={"allowed_values": [...]}` listing all supported classes.
- Add a `model_params` dict parameter for scalar constructor arguments (values become `np.array([v])` internally).
- Add a `region_params` dict parameter if per-region overrides are needed (e.g. epilepsy heatmaps).
- Make any input that is only needed for some configurations `optional=True`.
- Resolve the class at runtime via `getattr(module, model_type)` with a fallback for non-standard imports.

**Example (TVB model node):**
```python
"model_type": ParameterDefinition(
    default_value="Generic2dOscillator",
    description="TVB model class: 'Generic2dOscillator' (resting-state), 'EpileptorRestingState' (seizure/resting hybrid), ...",
    constraints={"allowed_values": ["Generic2dOscillator", "EpileptorRestingState", ...]},
),
"model_params": ParameterDefinition(
    default_value={"a": 1.74},
    description="Dict of scalar parameters passed to the model constructor as numpy arrays.",
),
```

This lets one node replace multiple model-specific nodes and serve both existing and new workflows by reconfiguring parameters.

---

## Step 2 — Handle new stage (if applicable)

If no existing stage fits, confirm the new stage name with the user before proceeding. A new stage is valid when:
- The operation is conceptually independent (different tool could produce the same output)
- It would apply to more than one possible node, not just this specific script
- A neuroscientist would describe it as a distinct step in the modeling process

Then:
1. Add the new stage entry to `NODE_CREATION_GUIDE.md` following the existing format (name, role, typical in/out, parameters, examples) — only if in repo mode and the file is present.
2. In repo mode: create `src/neuroworkflow/nodes/<new_stage>/__init__.py`
3. Inform the user the new stage has been registered

---

## Step 3 — Generate the node file

### Where to place it

Use the output path determined in Environment Detection:
- **User-specified folder**: `<user_folder>/<NodeName>.py`
- **Repo mode default**: `src/neuroworkflow/nodes/sandbox/<NodeName>.py`
- **Standalone fallback**: `./nodes/<NodeName>.py`

In repo mode, the sandbox is a staging area. Once tested end-to-end, the node can be promoted to its final stage folder:
```
src/neuroworkflow/nodes/<stage>/<NodeName>.py`
```

In standalone mode, the output folder itself is the staging area — promotion to the repo happens separately when the user uploads the node via the GUI.

### File structure

```python
from typing import Dict, Any, Optional

# Domain-specific imports go here (e.g. import nest, from tvb.simulator import simulator)

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import (
    NodeDefinitionSchema,
    PortDefinition,
    ParameterDefinition,
    MethodDefinition,
)
from neuroworkflow.core.port import PortType


class <NodeName>(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="<snake_case_type>",
        stage="<stage>",
        tool="<tool>",
        model_source="<url>",
        description="<one scientifically specific sentence written for an AI agent>",
        parameters={
            "<param>": ParameterDefinition(
                default_value=<default>,
                description="<specific description>",
                # constraints={"min": ..., "max": ...},  # or "allowed_values": [...]
                # optimizable=True,
                # optimization_range=[..., ...],
            ),
        },
        inputs={
            "<port>": PortDefinition(
                type=PortType.<TYPE>,
                description="<specific description of the data, not just the name>",
                # optional=True,  # add if this input is not required
            ),
        },
        outputs={
            "<port>": PortDefinition(
                type=PortType.<TYPE>,
                description="<specific description of the data, not just the name>",
            ),
        },
        methods={
            "<method>": MethodDefinition(
                description="<what this method computes>",
                inputs=["<port>", ...],
                outputs=["<port>", ...],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        # Add one call per method, in execution order.
        self.add_process_step("<method>", self.<method>, method_key="<method>")

    def <method>(self, <inputs>) -> Dict[str, Any]:
        # IMPORTANT: return dict keys must exactly match the output port names in NODE_DEFINITION.
        # A mismatch silently leaves the output port as None.
        # TODO: implement
        raise NotImplementedError
```

### Reading parameters inside methods

Use `self._parameters["key"]` — there is no `get_parameter()` method:

```python
def my_method(self, some_input) -> Dict[str, Any]:
    bin_size = float(self._parameters["bin_size"])
    title    = str(self._parameters["title"])
```

### Passing data between steps

The framework puts each step's return dict into `self._context`. The next step receives values from it by matching parameter names. No instance variables needed for inter-step data:

```python
def step_one(self, some_input) -> Dict[str, Any]:
    result = compute(some_input)
    return {"intermediate": result}      # key name must match step_two's parameter name

def step_two(self, intermediate) -> Dict[str, Any]:  # name matches → wired automatically
    final = process(intermediate)
    return {"output_port": final}        # key must match declared output port name
```

Use instance variables (`self._x`) only for truly internal bookkeeping, not for passing data between steps.

### Re-exporting a parameter as an output port

A node can expose one of its own parameter values as an output port by including the parameter's key in the method's return dict. This lets downstream nodes receive the value automatically without the user having to configure it twice. Useful when multiple nodes in a chain need the same dict (e.g. `neuron_params`):

```python
outputs={
    "neuron_params": PortDefinition(
        type=PortType.DICT,
        description="Neuron parameter dict passed through for downstream nodes.",
    ),
},

def configure_network(self, ...) -> Dict[str, Any]:
    neuron_params = self._parameters["neuron_params"]
    # ... do work ...
    return {
        "population": population,
        "neuron_params": neuron_params,   # re-export parameter as output
    }
```

### Writing output files — use `results_path` from context

When a node writes files to disk (plots, CSVs, HDF5, spike recordings, simulator output directories), follow this pattern:

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
    ...
    return {"output_file": os.path.join(data_path, "my_output.csv")}
```

**Why this order matters:** The output directory must exist before the simulator tries to write to it. Reading from context first allows a single `results_path` set at the workflow level to propagate automatically to all nodes, without each node needing to independently configure the same path.

`self._context` is populated from `WorkflowBuilder.context`. The `results_path` key defaults to `"results/"`, but users can set it to any path they want in the workflow context — the node always reads from context, so it adapts automatically. Nodes that only pass data in memory (no file I/O) do not need to read `results_path`.

---

### NEST kernel reset

NEST is a global stateful kernel. **Never call `nest.ResetKernel()` at module import or in `__init__`** — it silently destroys kernel state if the node is imported or instantiated mid-workflow, which is especially problematic in Jupyter notebooks.

Place it inside a process step method. If the node design calls for it, consider exposing reset behaviour as a boolean parameter so the user can control whether the kernel is reset when the node runs — useful when chaining multiple NEST nodes in one session:

```python
"reset_kernel": ParameterDefinition(
    default_value=True,
    description="If True, call nest.ResetKernel() at the start of this step. Set to False when chaining multiple NEST nodes in one session.",
),

def build_network(self, ...) -> Dict[str, Any]:
    if self._parameters["reset_kernel"]:
        nest.ResetKernel()
    # ... rest of setup ...
```

---

## Step 4 — Register the node

No import registration needed. Import the node directly from its module:

**Repo mode:**
```python
from neuroworkflow.nodes.sandbox.<NodeName> import <NodeName>
```

**Standalone mode:**
```python
import sys
sys.path.insert(0, '<output_dir>')
from <NodeName> import <NodeName>
```

---

## Step 5 — Run the checklist

Before finishing, verify every item:

- [ ] `stage` is set and matches the stage list
- [ ] `tool` identifies the simulator or library
- [ ] `model_source` URL is provided
- [ ] `description` is one clear scientific sentence written for an AI agent
- [ ] All parameters have `description` and `default_value`
- [ ] Parameters with numeric bounds have `constraints` (`min`/`max`)
- [ ] Parameters with fixed options have `constraints` (`allowed_values`)
- [ ] Scientifically tunable parameters have `optimizable=True` and `optimization_range`
- [ ] No parameter exists only to hold an optimization target (targets go on `NW_Optimization.objectives`)
- [ ] Optional input ports are marked `optional=True`
- [ ] All port descriptions are specific enough for an agent to understand the data
- [ ] Every method's return dict keys exactly match the corresponding output port names in `NODE_DEFINITION` (mismatch silently leaves ports as `None`)
- [ ] `type` field is globally unique across all nodes (check existing nodes — do not reuse a type string)
- [ ] Node is in the correct output folder (user-specified, repo sandbox, or `./nodes/`)
- [ ] In repo mode: `__init__.py` exists in the output folder
- [ ] Smoke test passes:
  - Repo mode: `from neuroworkflow.nodes.sandbox.<NodeName> import <NodeName>; n = <NodeName>("test"); print(n.get_info())`
  - Standalone mode: `import sys; sys.path.insert(0, '<output_dir>'); from <NodeName> import <NodeName>; n = <NodeName>("test"); print(n.get_info())`
- [ ] If configurable node pattern was used: verify all supported `model_type` values can be instantiated
- [ ] `is_objective` / `objective_range` are only on `ParameterDefinition`, never on `PortDefinition`
- [ ] NEST nodes: `nest.ResetKernel()` is inside a process step method, not at import or `__init__`
- [ ] Nodes that write files use `self._context.get("results_path", "results/")` as the output directory (not a hardcoded path)

---

## Step 6 — Tell the user

Report:
- **Full absolute path of every file created** — always, regardless of mode
- Checklist status
- Which mode was used (repo / standalone) and which output folder
- What validation is needed before promotion to the repo or GUI

Note: syncing nodes to the GUI (`gui/workflow_backend/django-project/codes/nodes/`) is a separate production step done after the node is validated and promoted — it is outside the scope of this skill.

---

## Step 7 — Offer to create a companion workflow file

After one or more nodes are created and smoke-tested, offer to create a workflow `.py` file that wires them together and runs a simulation end-to-end.

A workflow file follows the pattern in `examples/epilepsy_rs.py`:

```python
from neuroworkflow import WorkflowBuilder
from neuroworkflow.nodes.sandbox.<NodeName> import <NodeName>

# 1. Instantiate nodes
node_a = NodeA("NodeA")
node_b = NodeB("NodeB")

# 2. Configure parameters
node_a.configure(param=value, ...)
node_b.configure(param=value, ...)

# 3. Build topology once
wf = WorkflowBuilder("WorkflowName")
wf.add_node(node_a)
wf.add_node(node_b)
wf.connect("NodeA", "output_port", "NodeB", "input_port")

# Set context BEFORE build() — build() reads self.context, it does not accept context as an argument
wf.context["results_path"] = os.path.join(os.getcwd(), "results")
workflow = wf.build()

# 4. Execute (reconfigure nodes between runs to vary parameters)
workflow.execute()

# 5. Reconfigure and re-execute for a second simulation variant
node_a.configure(param=other_value, ...)
workflow.execute()
```

**Key rules for workflow files:**
- Place in the output folder (determined by Environment Detection) while under development, or in `examples/` once validated (repo mode only).
- One `WorkflowBuilder` per topology. Re-execute the same `workflow` object after reconfiguring nodes — do not rebuild for parameter-only changes.
- Use descriptive node instance names (passed to the constructor) since these are used as keys in `connect()`.
- Build paths to data files with a try/except to handle both script and Jupyter notebook execution. Use the **same relative path in both branches** — `__file__` and `os.getcwd()` resolve to the same directory when the notebook CWD matches the script location:
  ```python
  try:
      _HERE = os.path.dirname(os.path.abspath(__file__))
  except NameError:
      _HERE = os.getcwd()  # Jupyter: CWD is typically the script's directory

  DATA_FILE = os.path.abspath(os.path.join(_HERE, "relative/path/to/data"))
  ```
