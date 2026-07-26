"""Database node: resolve one dataset in the bm_mindsdb (mdb) catalog by ID.

Use this when a workflow already knows which dataset it needs — pinning a
specific dandiset for a reproducible run, for example — instead of searching.
mdb normalises the ID per source, so a bare number resolves to the stored form.
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


class MDBCatalogLookupNode(Node):
    """Look up a single dataset record in the mdb catalog by ID."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="mdb_catalog_lookup",
        description=(
            "Looks up one dataset in the bm_mindsdb (mdb) catalog by its ID and "
            "outputs the full record. Use it to pin a known dataset for a "
            "reproducible workflow. mdb normalises the ID per source, so a bare "
            "number resolves to the stored identifier."
        ),
        stage="database",
        tool="mdb",
        model_source="https://github.com/oist/bm_mindsdb",
        parameters={
            "dataset_id": ParameterDefinition(
                default_value="",
                description=(
                    "Dataset identifier, e.g. '000004' or 'DANDI:000004' for "
                    "DANDI. Required."
                ),
            ),
            "source": ParameterDefinition(
                default_value="dandi",
                description="Catalog source the ID belongs to.",
                constraints={
                    "allowed_values": [
                        "dandi",
                        "cbs",
                        "brainminds",
                        "bmb_human",
                        "aws",
                    ]
                },
            ),
            "table": ParameterDefinition(
                default_value="api_datasets",
                description=(
                    "Catalog table to search. 'api_datasets' holds remote "
                    "catalogs, 'local_catalog_datasets' the local BIDS "
                    "datasets, 'metadata_entries' the legacy context import."
                ),
                constraints={
                    "allowed_values": [
                        "api_datasets",
                        "local_catalog_datasets",
                        "metadata_entries",
                    ]
                },
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
            "dataset": PortDefinition(
                type=PortType.DICT,
                description=(
                    "The matched dataset record, or an empty dict when the ID "
                    "is not in the catalog."
                ),
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Lookup envelope: status, count, source, table, "
                    "requested_id, normalized_id."
                ),
            ),
        },
        methods={
            "fetch": MethodDefinition(
                description="Look up a dataset record by ID.",
                inputs=[],
                outputs=["dataset", "metadata"],
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
        env = client.lookup(p["dataset_id"], source=p["source"], table=p["table"])

        metadata = {
            "status": env.get("status"),
            "count": env.get("count", 0),
            "source": env.get("source", p["source"]),
            "table": env.get("table", p["table"]),
            "requested_id": env.get("requested_id", p["dataset_id"]),
            "normalized_id": env.get("normalized_id"),
            "base_url": env.get("base_url"),
        }
        if env.get("status") != "success":
            metadata["error"] = env.get("error")
        return {"dataset": env.get("record") or {}, "metadata": metadata}
