"""Database node: download the data files referenced by OAI-PMH records.

Files are fetched through the NeuroWorkflow backend proxy (which holds the
repository API key) and written under the workflow results path. See
docs/OAI_PMH_HARVEST.md.
"""

import os
import re
from typing import Any, Dict, List, Optional

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)
from neuroworkflow.utils.oai_pmh import (
    OAIPMHClient,
    OAIPMHError,
    fetch_backend_records,
)


def _safe_name(value: Any) -> str:
    """Reduce a folder/file name to a single safe path component."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return name or "unnamed"


class OAIPMHDownloadNode(Node):
    """Download every file listed in the input records into the results path."""

    NODE_DEFINITION = NodeDefinitionSchema(
        type="oai_pmh_download",
        description=(
            "Downloads the data files referenced by OAI-PMH records (from "
            "OAIPMHRecordsNode) into the workflow results directory through the "
            "backend proxy. Records without a file list are resolved from the "
            "backend's harvested record store first."
        ),
        stage="database",
        tool="OAI-PMH",
        model_source="https://www.openarchives.org/OAI/openarchivesprotocol.html",
        parameters={
            "subdir": ParameterDefinition(
                default_value="oai_pmh",
                description=(
                    "Sub-directory under the results path; files are written to "
                    "<subdir>/<record folder name or id>/<file name>."
                ),
            ),
            "max_files_per_record": ParameterDefinition(
                default_value=10,
                description="Maximum files to download per record; 0 downloads all.",
                constraints={"min": 0},
            ),
            "timeout": ParameterDefinition(
                default_value=60,
                description="HTTP timeout per request in seconds.",
                constraints={"min": 1},
            ),
            "skip_existing": ParameterDefinition(
                default_value=True,
                description="Skip files that already exist at the destination path.",
            ),
        },
        inputs={
            "records": PortDefinition(
                type=PortType.LIST,
                description=(
                    "Record dicts from OAIPMHRecordsNode (identifier, metadata, "
                    "files [{id, name, mime_type, size}])."
                ),
            ),
        },
        outputs={
            "file_paths": PortDefinition(
                type=PortType.LIST,
                description=(
                    "Paths of the files present after the run (downloaded or "
                    "already existing), in record order."
                ),
            ),
            "download_metadata": PortDefinition(
                type=PortType.DICT,
                description=(
                    "Download envelope: status, downloaded, skipped, failed, "
                    "records_without_files, errors [{identifier, file_id, error}]."
                ),
            ),
        },
        methods={
            "download": MethodDefinition(
                description=(
                    "Resolve each record's file list and stream every file to disk."
                ),
                inputs=["records"],
                outputs=["file_paths", "download_metadata"],
            ),
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("download", self.download, method_key="download")

    def _make_client(self) -> OAIPMHClient:
        return OAIPMHClient(timeout=self._parameters["timeout"])

    def download(
        self, records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        p = self._parameters
        base_dir = os.path.join(
            self._context.get("results_path", "results"), p["subdir"]
        )
        client = self._make_client()
        limit = int(p["max_files_per_record"]) or None
        file_paths: List[str] = []
        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "records_without_files": 0}
        errors: List[Dict[str, Any]] = []
        claimed: Dict[str, str] = {}  # destination path -> file id

        for record in records or []:
            if not isinstance(record, dict) or record.get("deleted"):
                continue
            identifier = str(record.get("identifier") or "")
            files = record.get("files") or []
            if not files and identifier:
                resolved = fetch_backend_records([identifier], timeout=p["timeout"])[
                    "records"
                ]
                if resolved:
                    record = resolved[0]
                    files = record.get("files") or []
            if not files:
                stats["records_without_files"] += 1
                continue

            metadata = record.get("metadata")
            folder_name = metadata.get("name") if isinstance(metadata, dict) else None
            folder = os.path.join(
                base_dir, _safe_name(folder_name or identifier.rsplit(":", 1)[-1])
            )
            for file_info in files[:limit]:
                file_id = str(file_info.get("id") or "")
                if not file_id:
                    continue
                name = _safe_name(file_info.get("name") or file_id)
                dest = os.path.join(folder, name)
                if claimed.get(dest, file_id) != file_id:
                    dest = os.path.join(folder, f"{file_id}_{name}")
                claimed[dest] = file_id
                if p["skip_existing"] and os.path.exists(dest):
                    stats["skipped"] += 1
                    file_paths.append(dest)
                    continue
                try:
                    file_paths.append(client.download_file(file_id, dest))
                    stats["downloaded"] += 1
                except OAIPMHError as e:
                    stats["failed"] += 1
                    errors.append(
                        {"identifier": identifier, "file_id": file_id, "error": str(e)}
                    )

        download_metadata = {
            "status": "error" if stats["failed"] else "success",
            **stats,
            "errors": errors,
        }
        return {"file_paths": file_paths, "download_metadata": download_metadata}
