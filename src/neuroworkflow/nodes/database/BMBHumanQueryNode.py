"""Database node: query the Brain/MINDS Beyond Human MRI portal.

Fetches dataset metadata from the Brain/MINDS Beyond Human MRI portal, which has
no API — the client scrapes the index for dataset slugs and reads a per-slug
BMB_META.json plus the landing page. With ``include_download_urls`` enabled,
download URLs are extracted from the schema.org distribution tree (no extra
network) and publication columns are attached. No authentication.
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
from neuroworkflow.utils.remote_catalogs import BMBHumanAPIClient, enrichment


class BMBHumanQueryNode(Node):
    """Query the Brain/MINDS Beyond Human MRI portal and output dataset records."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="bmb_human_query",
        description=(
            "Queries the Brain/MINDS Beyond Human MRI portal (no API; scrapes "
            "dataset slugs and reads per-slug BMB_META.json) and outputs dataset "
            "records (name, description, DOI, citation) for downstream nodes. "
            "Optional enrichment adds download URLs from the schema.org "
            "distribution tree. No authentication required."
        ),
        stage="database",
        tool="BMBHuman",
        model_source="https://mridata-brainminds-beyond.atr.jp",
        parameters={
            "limit": ParameterDefinition(
                default_value=20,
                description="Maximum number of datasets to fetch.",
                constraints={"min": 1},
            ),
            "offset": ParameterDefinition(
                default_value=0,
                description="Number of dataset slugs to skip (client-side slice).",
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
                    "List of BMB Human dataset record dicts (identifier, name, "
                    "description, doi, landing_page, page_citation, schema_org). "
                    "With include_download_urls, each record also carries "
                    "data_urls, dataset_doi, and related_publications."
                ),
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description="Fetch envelope: status, count, total, source.",
            ),
        },
        methods={
            "fetch": MethodDefinition(
                description="Fetch dataset records from the BMB Human MRI portal.",
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
        client = BMBHumanAPIClient(timeout=p["timeout"])
        env = client.get_datasets(limit=p["limit"], offset=p["offset"])

        datasets = env.get("datasets", [])
        if p["include_download_urls"]:
            for ds in datasets:
                enrichment.enrich_bmb_human_dataset(ds)
                ds.update(enrichment.build_publication_columns("bmb_human", ds))

        metadata = {
            "status": env.get("status"),
            "count": env.get("count"),
            "total": env.get("total"),
            "source": "bmb_human",
        }
        if env.get("status") != "success":
            metadata["error"] = env.get("error")
        return {"datasets": datasets, "metadata": metadata}
