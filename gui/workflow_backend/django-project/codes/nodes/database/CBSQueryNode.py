"""Database node: query the CBS (RIKEN neurodata) catalog.

Fetches dataset metadata from the CBS Dataportal via its ResourceSync XML
catalog. With ``include_download_urls`` enabled, each record is enriched with
neurodata API v3 file download URLs and publication columns (extra network
calls per dataset).
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
from neuroworkflow.utils.remote_catalogs import CBSAPIClient, enrichment


class CBSQueryNode(Node):
    """Query the CBS Dataportal (RIKEN) and output dataset records."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="cbs_query",
        description=(
            "Queries the CBS Dataportal (RIKEN neurodata) ResourceSync catalog "
            "and outputs dataset records (title, description, DOI, dates) for "
            "downstream nodes. Optional enrichment adds neurodata v3 file "
            "download URLs."
        ),
        stage="database",
        tool="CBS",
        model_source="https://neurodata.riken.jp/rs",
        parameters={
            "limit": ParameterDefinition(
                default_value=20,
                description="Maximum number of datasets to fetch.",
                constraints={"min": 1},
            ),
            "offset": ParameterDefinition(
                default_value=0,
                description="Number of catalog entries to skip (client-side slice).",
                constraints={"min": 0},
            ),
            "include_download_urls": ParameterDefinition(
                default_value=False,
                description=(
                    "If true, enrich each record with DOI, related publications, "
                    "and neurodata v3 file download URLs (extra network calls per "
                    "dataset — keep `limit` small)."
                ),
            ),
            "api_key": ParameterDefinition(
                default_value="",
                description="Optional CBS API key (sent as Bearer token).",
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
                    "List of CBS dataset record dicts (identifier, title, "
                    "description, doi, landing_page, file_urls). With "
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
                description="Fetch dataset records from the CBS ResourceSync catalog.",
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
        client = CBSAPIClient(api_key=p["api_key"] or None, timeout=p["timeout"])
        env = client.get_datasets(limit=p["limit"], offset=p["offset"])

        datasets = env.get("datasets", [])
        if p["include_download_urls"]:
            for ds in datasets:
                enrichment.enrich_cbs_dataset(ds, client)
                ds.update(enrichment.build_publication_columns("cbs", ds))

        metadata = {
            "status": env.get("status"),
            "count": env.get("count"),
            "total": env.get("total"),
            "source": "cbs",
        }
        if env.get("status") != "success":
            metadata["error"] = env.get("error")
        return {"datasets": datasets, "metadata": metadata}
