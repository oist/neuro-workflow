"""Database node: query the DANDI Archive catalog.

Fetches dandiset records from the DANDI Archive and emits them for downstream
nodes. With ``include_download_urls`` enabled, each record is enriched with the
preferred version metadata, per-asset download URLs, DOI, and related
publications (extra network calls per dataset).
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
from neuroworkflow.utils.remote_catalogs import DANDIAPIClient, enrichment


class DANDIQueryNode(Node):
    """Query the DANDI Archive and output dataset records."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="dandi_query",
        description=(
            "Queries the DANDI Archive catalog and outputs dandiset records "
            "(id, name, versions, metadata) for downstream nodes. Supports "
            "full-text search and optional enrichment with per-asset download "
            "URLs and related publications."
        ),
        stage="database",
        tool="DANDI",
        model_source="https://api.dandiarchive.org/api",
        parameters={
            "search": ParameterDefinition(
                default_value="",
                description=(
                    "Full-text search query. Empty string lists all dandisets "
                    "via cursor pagination up to `limit`."
                ),
            ),
            "limit": ParameterDefinition(
                default_value=20,
                description="Maximum number of datasets to fetch.",
                constraints={"min": 1},
            ),
            "include_download_urls": ParameterDefinition(
                default_value=False,
                description=(
                    "If true, enrich each record with DOI, related publications, "
                    "and per-asset download URLs (one extra network call per "
                    "dataset — keep `limit` small)."
                ),
            ),
            "api_key": ParameterDefinition(
                default_value="",
                description="Optional DANDI API key (sent as Bearer token).",
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
                    "List of dandiset record dicts (raw DANDI objects). With "
                    "include_download_urls, each record also carries data_urls, "
                    "dataset_doi, and related_publications."
                ),
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description="Fetch envelope: status, count, total, source.",
            ),
        },
        methods={
            "fetch": MethodDefinition(
                description="Fetch dandiset records from the DANDI Archive.",
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
        client = DANDIAPIClient(api_key=p["api_key"] or None, timeout=p["timeout"])
        if p["search"]:
            env = client.search_dandisets(p["search"], limit=p["limit"])
        else:
            env = client.get_dandisets(limit=p["limit"])

        datasets = env.get("datasets", [])
        if p["include_download_urls"]:
            for ds in datasets:
                enrichment.enrich_dandi_dataset(ds, client)
                ds.update(enrichment.build_publication_columns("dandi", ds))

        metadata = {
            "status": env.get("status"),
            "count": env.get("count"),
            "total": env.get("total"),
            "source": "dandi",
        }
        if env.get("status") != "success":
            metadata["error"] = env.get("error")
        return {"datasets": datasets, "metadata": metadata}
