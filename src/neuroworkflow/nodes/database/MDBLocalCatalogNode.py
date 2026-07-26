"""Database node: read the local BIDS catalog indexed by bm_mindsdb (mdb).

mdb indexes on-disk BIDS trees (currently SRPBS_TS, source key ``aws``) into
normalised participant / session / site tables. This node exposes those tables
so a workflow can select subjects or sessions by site, age or diagnosis without
touching the imaging files themselves — only metadata is indexed.

Requires ``POST /api/ingest_local_catalog`` to have been run on the mdb side,
which in turn needs a per-developer ``srpbs-ts.local.json`` pointing at the BIDS
paths. Without it this node reports an error in ``metadata`` rather than raising.
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


class MDBLocalCatalogNode(Node):
    """Read participants, sessions or per-site summaries from mdb's local catalog."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="mdb_local_catalog",
        description=(
            "Reads the local BIDS catalog indexed by bm_mindsdb (mdb) — "
            "participant, session and per-site tables for an on-disk dataset "
            "such as SRPBS_TS. Use it to select subjects or sessions by site or "
            "participant before downstream processing. Only metadata is "
            "indexed; no imaging files are read."
        ),
        stage="database",
        tool="mdb",
        model_source="https://github.com/oist/bm_mindsdb",
        parameters={
            "source": ParameterDefinition(
                default_value="aws",
                description=(
                    "Local catalog source key. 'aws' is SRPBS_TS, the only "
                    "dataset mdb currently ships a definition for."
                ),
            ),
            "dataset_id": ParameterDefinition(
                default_value="srpbs-ts",
                description="Local catalog dataset identifier.",
            ),
            "view": ParameterDefinition(
                default_value="participants",
                description=(
                    "Which table to read. 'participants' and 'sessions' return "
                    "rows; 'sites' returns a per-site session summary; 'index' "
                    "returns the dataset record with its counts."
                ),
                constraints={
                    "allowed_values": ["participants", "sessions", "sites", "index"]
                },
            ),
            "site_code": ParameterDefinition(
                default_value="",
                description=(
                    "Filter sessions to one acquisition site. Applies to the "
                    "'sessions' view only; empty string means no filter."
                ),
            ),
            "participant_id": ParameterDefinition(
                default_value="",
                description=(
                    "Filter sessions to one participant. Applies to the "
                    "'sessions' view only; empty string means no filter."
                ),
            ),
            "limit": ParameterDefinition(
                default_value=500,
                description=(
                    "Maximum number of session rows to fetch. Applies to the "
                    "'sessions' view only."
                ),
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
            "records": PortDefinition(
                type=PortType.LIST,
                description=(
                    "Rows for the selected view. Empty for the 'index' view, "
                    "which reports through the index output instead."
                ),
            ),
            "index": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Dataset index (participant/session counts, sites) for the "
                    "'index' view; empty dict for the other views."
                ),
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description="Fetch envelope: status, count, view, source, dataset_id.",
            ),
        },
        methods={
            "fetch": MethodDefinition(
                description="Read a table from mdb's local BIDS catalog.",
                inputs=[],
                outputs=["records", "index", "metadata"],
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
        env = client.local_catalog(
            source=p["source"],
            dataset_id=p["dataset_id"],
            view=p["view"],
            site_code=p["site_code"],
            participant_id=p["participant_id"],
            limit=p["limit"],
        )

        metadata = {
            "status": env.get("status"),
            "count": env.get("count", 0),
            "view": p["view"],
            "source": p["source"],
            "dataset_id": p["dataset_id"],
            "base_url": env.get("base_url"),
        }
        if env.get("status") != "success":
            metadata["error"] = env.get("error")
        return {
            "records": env.get("datasets", []),
            "index": env.get("index") or {},
            "metadata": metadata,
        }
