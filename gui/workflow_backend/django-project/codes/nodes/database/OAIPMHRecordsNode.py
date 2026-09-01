"""Database node: fetch harvested OAI-PMH records from the backend.

The NeuroWorkflow backend harvests the repository on a schedule
(``manage.py harvest_oai``) into its database; this node reads the records
selected in the GUI dataset search box by identifier. The repository address
and API key stay on the backend. See docs/OAI_PMH_HARVEST.md.
"""

import re
from typing import Any, Dict, List

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)
from neuroworkflow.utils.oai_pmh import fetch_backend_records

_CHUNK = 100  # request cap of /api/harvest/records/


def _split_identifiers(raw: str) -> List[str]:
    """Split a comma/newline separated identifier list, deduplicated in order."""
    identifiers: List[str] = []
    for part in re.split(r"[,\n]", str(raw or "")):
        ident = part.strip()
        if ident and ident not in identifiers:
            identifiers.append(ident)
    return identifiers


class OAIPMHRecordsNode(Node):
    """Fetch the listed identifiers from the backend's harvested record store."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="oai_pmh_records",
        description=(
            "Fetches metadata records from the repository copy that the "
            "NeuroWorkflow backend harvests periodically over OAI-PMH "
            "(repository address and API key are held server-side and are not "
            "node parameters). Fill 'identifiers' with the GUI dataset search "
            "box; each record lists its data files for OAIPMHDownloadNode."
        ),
        stage="database",
        tool="OAI-PMH",
        model_source="https://www.openarchives.org/OAI/openarchivesprotocol.html",
        parameters={
            "identifiers": ParameterDefinition(
                default_value="",
                description=(
                    "Comma- or newline-separated OAI identifiers to fetch "
                    "(the GUI dataset search box fills this in)."
                ),
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
                    "deleted, metadata_prefix, metadata (folder dict with name, "
                    "path, size, parent, metadata, files), files "
                    "[{id, name, mime_type, size}]."
                ),
            ),
            "metadata": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Fetch envelope: status ('success'|'error'), count, total "
                    "(the identifier count), error, error_code ('not_found' "
                    "when identifiers are missing from the harvested copy)."
                ),
            ),
        },
        methods={
            "fetch": MethodDefinition(
                description=(
                    "Fetch each listed identifier from the backend's harvested "
                    "record store."
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
        identifiers = _split_identifiers(p["identifiers"])
        records: List[Dict[str, Any]] = []
        failures: List[str] = []
        error_code = None
        for start in range(0, len(identifiers), _CHUNK):
            result = fetch_backend_records(
                identifiers[start : start + _CHUNK], timeout=p["timeout"]
            )
            records.extend(result["records"])
            if result["status"] == "error":
                failures.append(result["error"])
                error_code = error_code or result["error_code"]
        return {
            "records": records,
            "metadata": {
                "status": "error" if failures else "success",
                "count": len(records),
                "total": len(identifiers),
                "error": "; ".join(failures) if failures else None,
                "error_code": error_code,
            },
        }
