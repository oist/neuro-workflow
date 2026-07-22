"""Vendored brain-viewer chatbot tool library + Django glue.

The Group 1-5 function modules (data/regions/structure/timeseries/metrics/
viewer_actions) are copied VERBATIM from the upstream draft at
``viewer_chatbot/functions/`` (numpy-only, provider-agnostic, returns
JSON-serialisable dicts). They are re-synced by diffing against upstream; the
only local change is a two-line provenance header at the top of each file.

Local glue added here (not from upstream):
- ``resolver.load_project_viewer_data`` — resolve a FlowProject to a ViewerData
  (find its run's connectivity JSON on disk + the species' region descriptions).
- ``registry`` — map a tool name to the vendored callable for the HTTP dispatch
  view (``ViewerChatToolView``).
"""

from .data import (  # noqa: F401
    SIGNAL_BOLD,
    SIGNAL_SPIKING,
    SIGNAL_TEMPORAL_AVERAGE,
    ViewerData,
    detect_signal_type,
    load_viewer_data,
)
from .metrics import (  # noqa: F401
    compute_metrics,
    explain_activity,
    functional_connectivity,
)
from .regions import get_region, list_groups, search_regions  # noqa: F401
from .registry import TOOL_REGISTRY, call_registered_tool  # noqa: F401
from .resolver import load_project_viewer_data  # noqa: F401
from .structure import get_connections, node_strength  # noqa: F401
from .timeseries import get_activity, list_signals  # noqa: F401
from .viewer_actions import (  # noqa: F401
    clear_selection,
    focus_region,
    highlight_region,
    set_time_window,
    show_curve,
)
