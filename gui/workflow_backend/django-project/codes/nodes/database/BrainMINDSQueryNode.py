"""Database node: query the Brain/MINDS Dataportal catalog.

Fetches schema.org dataset metadata from the Brain/MINDS Dataportal WordPress
REST API. With ``include_download_urls`` enabled, download URLs are extracted
from the already-present schema.org distribution tree (no extra network) and
publication columns are attached.
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
from neuroworkflow.utils.remote_catalogs import BrainMINDSAPIClient, enrichment


class BrainMINDSQueryNode(Node):
    """Query the Brain/MINDS Dataportal and output dataset records."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="brainminds_query",
        description=(
            "Queries the Brain/MINDS Dataportal catalog (schema.org metadata via "
            "WordPress REST) and outputs dataset records (name, description, DOI, "
            "keywords) for downstream nodes. Optional enrichment adds download "
            "URLs from the schema.org distribution tree."
        ),
        stage="database",
        tool="BrainMINDS",
        model_source="https://dataportal.brainminds.jp/wp-json/bminds/datasetmeta/",
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
                    "and download URLs from the schema.org distribution tree "
                    "(no extra network calls)."
                ),
            ),
            "api_key": ParameterDefinition(
                default_value="",
                description="Optional Brain/MINDS API key (sent as Bearer token).",
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
                    "List of Brain/MINDS dataset record dicts (portal_id, name, "
                    "description, doi, url, keywords, schema_org). With "
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
                description="Fetch dataset records from the Brain/MINDS Dataportal.",
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
        client = BrainMINDSAPIClient(api_key=p["api_key"] or None, timeout=p["timeout"])
        env = client.get_datasets(limit=p["limit"], offset=p["offset"])

        datasets = env.get("datasets", [])
        if p["include_download_urls"]:
            for ds in datasets:
                enrichment.enrich_brainminds_dataset(ds)
                ds.update(enrichment.build_publication_columns("brainminds", ds))

        metadata = {
            "status": env.get("status"),
            "count": env.get("count"),
            "total": env.get("total"),
            "source": "brainminds",
        }
        if env.get("status") != "success":
            metadata["error"] = env.get("error")
        return {"datasets": datasets, "metadata": metadata}
