"""Shared harvest logic: OAI-PMH client loading, record upserts, search text."""

import importlib.util
import os
from pathlib import Path

from .models import HarvestedRecord, HarvestRun

# The repository serves some ListRecords pages extremely slowly (folders with
# large file lists take 1-2 minutes each to serialize), so the harvester needs
# a far larger per-request timeout than the download proxy's OAI_PMH_TIMEOUT.
DEFAULT_HARVEST_TIMEOUT = 300.0

_OAI_PMH_PATH = (
    Path(__file__).resolve().parents[2] / "codes/neuroworkflow/utils/oai_pmh.py"
)
_oai_pmh_module = None


def oai_pmh():
    """Load the stdlib OAI-PMH client shipped for the kernels (``codes/`` copy).

    Loaded by file path because ``codes/`` is not an importable package in the
    backend, and importing through the package would execute kernel-oriented
    ``__init__`` modules. ``oai_pmh.py`` itself is stdlib-only and self-contained.
    """
    global _oai_pmh_module
    if _oai_pmh_module is None:
        spec = importlib.util.spec_from_file_location(
            "app.harvest._oai_pmh", _OAI_PMH_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _oai_pmh_module = module
    return _oai_pmh_module


def make_client():
    """Build the upstream client (direct mode; ``OAI_PMH_BASE_URL`` is set).

    Separate function so tests can monkeypatch the upstream in one place.
    """
    try:
        timeout = float(
            os.environ.get("OAI_PMH_HARVEST_TIMEOUT", DEFAULT_HARVEST_TIMEOUT)
        )
    except ValueError:
        timeout = DEFAULT_HARVEST_TIMEOUT
    return oai_pmh().OAIPMHClient(timeout=timeout)


def build_search_text(record):
    """Lower-cased haystack over the fields the search box should match."""
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    parts = [
        record.get("identifier", ""),
        str(metadata.get("name", "")),
        str(metadata.get("description", "")),
        str(metadata.get("laboratory_name", "")),
        str(metadata.get("path", "")),
    ]
    parts.extend(str(f.get("name", "")) for f in record.get("files", []))
    return " ".join(parts).lower()


def upsert_records(records):
    """Write parsed records into ``HarvestedRecord``.

    A deleted record arrives without metadata, so it only flips the flag and
    datestamp, keeping whatever content an earlier harvest stored.
    """
    for record in records:
        identifier = record.get("identifier", "")
        if not identifier:
            continue
        defaults = {
            "datestamp": record.get("datestamp", ""),
            "set_specs": record.get("set_specs", []),
            "deleted": bool(record.get("deleted")),
        }
        if not record.get("deleted"):
            defaults.update(
                metadata=record.get("metadata") or {},
                files=record.get("files") or [],
                search_text=build_search_text(record),
            )
        HarvestedRecord.objects.update_or_create(
            oai_identifier=identifier, defaults=defaults
        )


def record_payload(row):
    """Project a ``HarvestedRecord`` row onto the ``parse_record`` dict shape."""
    return {
        "identifier": row.oai_identifier,
        "datestamp": row.datestamp,
        "set_specs": row.set_specs,
        "deleted": row.deleted,
        "metadata_prefix": "mdrs",
        "metadata": row.metadata or None,
        "files": row.files,
    }


def latest_success_run():
    """The newest successful harvest run (watermark and freshness source)."""
    return (
        HarvestRun.objects.filter(status=HarvestRun.Status.SUCCESS)
        .order_by("-finished_at")
        .first()
    )
