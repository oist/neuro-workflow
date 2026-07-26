"""Database node: search the bm_mindsdb (mdb) dataset catalog.

Searches mdb's synced catalog across DANDI, CBS, Brain/MINDS, BMB Human and the
local BIDS catalog in one call. Unlike the per-source nodes (DANDIQueryNode and
friends), which query each upstream API live, this reads mdb's local copy: it is
fast, spans every source at once, and returns the same records on re-runs. The
trade-off is that it reflects the catalog as of mdb's last sync.
"""

from typing import Any, Dict

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)
from neuroworkflow.utils.mdb_client import MDBClient


class MDBCatalogSearchNode(Node):
    """Search the mdb catalog and output matching dataset records."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="mdb_catalog_search",
        description=(
            "Searches the bm_mindsdb (mdb) dataset catalog across all synced "
            "sources (DANDI, CBS, Brain/MINDS, BMB Human, local BIDS) and "
            "outputs the matching dataset records. Reads mdb's local catalog, "
            "so results are fast and reproducible but only as current as the "
            "last catalog sync."
        ),
        stage="database",
        tool="mdb",
        model_source="https://github.com/oist/bm_mindsdb",
        parameters={
            "query": ParameterDefinition(
                default_value="",
                description=(
                    "Full-text search term. Empty string lists the catalog "
                    "instead of searching."
                ),
            ),
            "source": ParameterDefinition(
                default_value="",
                description=(
                    "Restrict to one source. Empty string searches all of them. "
                    "'aws' is the local BIDS catalog (SRPBS_TS)."
                ),
                constraints={
                    "allowed_values": [
                        "",
                        "dandi",
                        "cbs",
                        "brainminds",
                        "bmb_human",
                        "aws",
                    ]
                },
            ),
            "limit": ParameterDefinition(
                default_value=50,
                description="Maximum number of records to output.",
                constraints={"min": 1},
            ),
            "base_url": ParameterDefinition(
                default_value="",
                description=(
                    "mdb base URL. Empty string uses the MDB_BASE_URL "
                    "environment variable, then http://mdb:8004."
                ),
            ),
            "timeout": ParameterDefinition(
                default_value=30,
                description="HTTP request timeout in seconds.",
                constraints={"min": 1},
            ),
        },
        outputs={
            "datasets": PortDefinition(
                type=PortType.LIST,
                description=(
                    "List of dataset record dicts, each carrying source, "
                    "dataset_id, name, description, metadata, DOI and related "
                    "publications."
                ),
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description="Fetch envelope: status, count, total, query, source.",
            ),
        },
        methods={
            "fetch": MethodDefinition(
                description="Search the mdb catalog.",
                inputs=[],
                outputs=["datasets", "metadata"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("fetch", self.fetch, method_key="fetch")

    def fetch(self) -> Dict[str, Any]:
        p = self._parameters
        client = MDBClient(base_url=p["base_url"] or None, timeout=p["timeout"])
        env = client.search_datasets(p["query"], source=p["source"], limit=p["limit"])

        metadata = {
            "status": env.get("status"),
            "count": env.get("count"),
            "total": env.get("total"),
            "source": env.get("source", p["source"] or "all"),
            "query": p["query"],
            "base_url": env.get("base_url"),
        }
        if env.get("status") != "success":
            metadata["error"] = env.get("error")
        return {"datasets": env.get("datasets", []), "metadata": metadata}
