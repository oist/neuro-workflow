"""Dotted addresses into a workflow.

Two kinds of address are used by the optimizer, both rooted at a node name:

* a **parameter** address names something to set — ``"Model.nest_params.C_m"``
* a **measurement** address names something to read — ``"Analysis.firing_rate_hz.exc"``

The first segment is always the node name. The second is the parameter or output
port name. Any further segments index into a nested dict.
"""

from copy import deepcopy
from typing import Any, Dict, List, Tuple

# Deeper than this and an "objective" is almost certainly a data structure
# rather than a measured quantity.
MAX_LEAF_DEPTH = 4


def split_address(address: str) -> Tuple[str, str, List[str]]:
    """Split ``"Node.port.k1.k2"`` into ``("Node", "port", ["k1", "k2"])``."""
    parts = address.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Address {address!r} must be at least 'Node.name' "
            f"(e.g. 'Analysis.firing_rate_hz.exc')"
        )
    return parts[0], parts[1], parts[2:]


def _get_node(workflow, node_name: str):
    if node_name not in workflow.nodes:
        raise KeyError(
            f"No node named {node_name!r} in the workflow "
            f"(have: {', '.join(sorted(workflow.nodes))})"
        )
    return workflow.nodes[node_name]


def set_parameter(workflow, address: str, value: Any) -> None:
    """Set a parameter by dotted address, going through ``Node.configure()``.

    ``configure()`` only accepts whole parameters, so a nested key is applied by
    copying the current dict, updating the key, and configuring the result. The
    copy matters: mutating the live dict in place would edit the node's state
    even if ``configure()`` then rejected the value.
    """
    node_name, param, keys = split_address(address)
    node = _get_node(workflow, node_name)

    if param not in node._parameters:
        raise KeyError(
            f"Node {node_name!r} has no parameter {param!r} "
            f"(have: {', '.join(sorted(node._parameters))})"
        )

    if not keys:
        node.configure(**{param: value})
        return

    current = deepcopy(node._parameters[param])
    if not isinstance(current, dict):
        raise TypeError(
            f"Address {address!r} indexes into {param!r}, but it holds "
            f"{type(current).__name__}, not a dict"
        )

    target = current
    for k in keys[:-1]:
        if not isinstance(target.get(k), dict):
            raise KeyError(f"Address {address!r}: {k!r} is not a nested dict")
        target = target[k]
    target[keys[-1]] = value

    node.configure(**{param: current})


def read_output(workflow, address: str) -> Any:
    """Read a value from an output port by dotted address."""
    node_name, port, keys = split_address(address)
    node = _get_node(workflow, node_name)

    if port not in node._output_ports:
        raise KeyError(
            f"Node {node_name!r} has no output port {port!r} "
            f"(have: {', '.join(sorted(node._output_ports))})"
        )

    value = node._output_ports[port].value
    for k in keys:
        if not isinstance(value, dict):
            raise TypeError(
                f"Address {address!r}: cannot index {type(value).__name__} with {k!r}"
            )
        if k not in value:
            raise KeyError(
                f"Address {address!r}: key {k!r} not in "
                f"{', '.join(map(str, list(value)[:8]))}"
            )
        value = value[k]
    return value


def _is_number(value: Any) -> bool:
    # bool is an int subclass but is never a measured quantity.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def discover_measurables(workflow) -> Dict[str, float]:
    """Walk every output port and collect the numeric leaves it currently holds.

    This is what makes ``measures`` usable: the candidate targets come from the
    data a baseline run actually produced, not from a declaration a node author
    had to anticipate. Lists and arrays are skipped on purpose — a target that
    needs a reduction belongs in an analysis node as its own named output.
    """
    found: Dict[str, float] = {}

    def walk(prefix: str, value: Any, depth: int) -> None:
        if _is_number(value):
            found[prefix] = float(value)
            return
        if isinstance(value, dict) and depth < MAX_LEAF_DEPTH:
            for k, v in value.items():
                # A leading underscore marks a key as internal plumbing a node passes
                # to another node — not something anyone would target. Without this,
                # a builder's bookkeeping shows up in the menu of measurable values
                # alongside the real measurements.
                if isinstance(k, str) and not k.startswith("_"):
                    walk(f"{prefix}.{k}", v, depth + 1)

    for node_name, node in workflow.nodes.items():
        for port_name, port in node._output_ports.items():
            walk(f"{node_name}.{port_name}", port.value, 1)

    return found
