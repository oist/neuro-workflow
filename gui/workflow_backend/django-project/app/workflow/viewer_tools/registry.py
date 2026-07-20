"""Tool-name -> vendored callable registry for the chat dispatch endpoint.

Keys are the logical tool names the MCP wrappers send (the ``viewer_`` prefix on
the MCP side is dropped here). The spiking stubs (mean_firing_rate / isi_cv /
fano_factor) are intentionally NOT registered — they raise NotImplementedError
and would only confuse the LLM.

Every callable takes ``vd`` (a ViewerData) as its first argument followed by
keyword args; ``call_registered_tool`` filters the caller's ``args`` to the
parameters each function actually accepts.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from . import metrics, regions, structure, timeseries, viewer_actions

# name -> callable(vd, **kwargs)
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    # Group 1 — region semantics
    "search_regions": regions.search_regions,
    "get_region": regions.get_region,
    "list_groups": regions.list_groups,
    # Group 2 — structure
    "get_connections": structure.get_connections,
    "node_strength": structure.node_strength,
    # Group 3 — timeseries
    "list_signals": timeseries.list_signals,
    "get_activity": timeseries.get_activity,
    # Group 4 — metrics + semantics
    "compute_metrics": metrics.compute_metrics,
    "explain_activity": metrics.explain_activity,
    "functional_connectivity": metrics.functional_connectivity,
    # Group 5 — viewer control (returns an {"action": ...} dict)
    "highlight_region": viewer_actions.highlight_region,
    "focus_region": viewer_actions.focus_region,
    "set_time_window": viewer_actions.set_time_window,
    "show_trace": viewer_actions.show_curve,
    "clear_selection": viewer_actions.clear_selection,
}


class UnknownTool(Exception):
    """The requested tool name is not registered."""


def call_registered_tool(tool_name: str, vd, args: dict | None) -> Any:
    """Dispatch ``tool_name`` against ``vd`` with the (filtered) ``args``."""
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        raise UnknownTool(tool_name)
    args = args or {}
    params = inspect.signature(fn).parameters
    kwargs = {k: v for k, v in args.items() if k in params and k != "vd"}
    return fn(vd, **kwargs)
