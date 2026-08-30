"""Database node: harvest metadata records from an OAI-PMH repository.

The repository address and API key are configured on the NeuroWorkflow backend
(``OAI_PMH_*`` in ``gui/workflow_backend/.env``). The node reaches the
repository through the backend proxy (``/api/harvest/oai/``) and never sees the
key. See docs/OAI_PMH_HARVEST.md.
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
from neuroworkflow.utils.oai_pmh import OAIPMHClient


class OAIPMHHarvestNode(Node):
    """Run ListRecords through the backend proxy and output record dicts."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="oai_pmh_harvest",
        description=(
            "Harvests metadata records from the OAI-PMH repository configured on "
            "the NeuroWorkflow backend (repository address and API key are held "
            "server-side and are not node parameters). Outputs one dict per "
            "record; with metadata_prefix 'mdrs' each record also lists its data "
            "files for OAIPMHDownloadNode."
        ),
        stage="database",
        tool="OAI-PMH",
        model_source="https://www.openarchives.org/OAI/openarchivesprotocol.html",
        parameters={
            "metadata_prefix": ParameterDefinition(
                default_value="mdrs",
                description=(
                    "OAI-PMH metadataPrefix. 'mdrs' (default) carries folder "
                    "details and the file ids needed for downloading; 'oai_dc' is "
                    "Dublin Core (title/publisher/date only; the download node then "
                    "resolves file lists per record with GetRecord)."
                ),
            ),
            "set_spec": ParameterDefinition(
                default_value="",
                description=(
                    "Optional OAI set to restrict the harvest (e.g. 'dataset', "
                    "'public', 'project:bm2.0'). Empty harvests all sets."
                ),
            ),
            "from_date": ParameterDefinition(
                default_value="",
                description=(
                    "Optional lower datestamp bound (UTC), 'YYYY-MM-DD' or "
                    "'YYYY-MM-DDThh:mm:ssZ'. Empty for no lower bound."
                ),
            ),
            "until_date": ParameterDefinition(
                default_value="",
                description="Optional upper datestamp bound, same format as from_date.",
            ),
            "max_records": ParameterDefinition(
                default_value=100,
                description=(
                    "Stop after this many records (result pages are followed via "
                    "resumptionToken until the cap is reached)."
                ),
                constraints={"min": 1},
            ),
            "timeout": ParameterDefinition(
                default_value=30,
                description="HTTP timeout per request in seconds.",
                constraints={"min": 1},
            ),
        },
        outputs={
            "records": PortDefinition(
                type=PortType.LIST,
                description=(
                    "List of record dicts: identifier, datestamp, set_specs, "
                    "deleted, metadata_prefix, metadata (oai_dc: {field: [values]}; "
                    "mdrs: folder dict with name, path, size, parent, metadata, "
                    "files), files [{id, name, mime_type, size}]."
                ),
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Harvest envelope: status ('success'|'error'), count, total "
                    "(completeListSize when reported), error, error_code."
                ),
            ),
        },
        methods={
            "fetch": MethodDefinition(
                description=(
                    "Run ListRecords through the backend proxy and follow "
                    "resumption tokens up to max_records."
                ),
                inputs=[],
                outputs=["records", "metadata"],
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
        envelope = OAIPMHClient(timeout=p["timeout"]).list_records(
            metadata_prefix=p["metadata_prefix"],
            set_spec=p["set_spec"],
            from_date=p["from_date"],
            until_date=p["until_date"],
            max_records=p["max_records"],
        )
        records = envelope.pop("records")
        return {"records": records, "metadata": envelope}
