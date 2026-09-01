"""Tests for the ``harvest_oai`` management command.

The upstream is faked at the ``services.make_client`` seam; what is exercised
is the watermark bookkeeping, the upserts and the run history.
"""

import pytest
from app.harvest.models import HarvestedRecord, HarvestRun
from django.core.management import CommandError, call_command

pytestmark = pytest.mark.django_db


def _record(
    identifier,
    name="Dataset",
    datestamp="2026-01-02T00:00:00Z",
    deleted=False,
    files=(),
):
    if deleted:
        return {
            "identifier": identifier,
            "datestamp": datestamp,
            "set_specs": ["dataset"],
            "deleted": True,
            "metadata_prefix": "mdrs",
            "metadata": None,
            "files": [],
        }
    return {
        "identifier": identifier,
        "datestamp": datestamp,
        "set_specs": ["public", "dataset"],
        "deleted": False,
        "metadata_prefix": "mdrs",
        "metadata": {
            "name": name,
            "description": "",
            "laboratory_name": "Doya Lab",
            "path": f"/{name}",
            "size": 1,
        },
        "files": [
            {"id": f"f-{i}", "name": n, "mime_type": "text/plain", "size": 10}
            for i, n in enumerate(files)
        ],
    }


def _envelope(records, error=None, error_code=None):
    return {
        "status": "error" if error else "success",
        "records": records,
        "count": len(records),
        "total": len(records),
        "error": error,
        "error_code": error_code,
    }


class _FakeClient:
    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = []

    def list_records(self, **kwargs):
        self.calls.append(kwargs)
        return self.envelope


@pytest.fixture
def harvest_env(monkeypatch):
    monkeypatch.setenv("OAI_PMH_BASE_URL", "https://repo.example/api/oai/")


def _install(monkeypatch, envelope):
    fake = _FakeClient(envelope)
    monkeypatch.setattr("app.harvest.services.make_client", lambda: fake)
    return fake


def test_initial_run_upserts_records_and_watermark(harvest_env, monkeypatch):
    fake = _install(
        monkeypatch,
        _envelope(
            [
                _record("oai:repo:1", "Alpha", datestamp="2026-01-01T00:00:00Z"),
                _record(
                    "oai:repo:2",
                    "Beta",
                    datestamp="2026-01-03T00:00:00Z",
                    files=("spikes.csv",),
                ),
            ]
        ),
    )

    call_command("harvest_oai")

    assert fake.calls == [
        {"metadata_prefix": "mdrs", "from_date": "", "max_records": 10000}
    ]
    assert HarvestedRecord.objects.count() == 2
    row = HarvestedRecord.objects.get(oai_identifier="oai:repo:2")
    assert row.metadata["name"] == "Beta"
    assert row.files[0]["id"] == "f-0"
    assert "beta" in row.search_text and "spikes.csv" in row.search_text
    run = HarvestRun.objects.get()
    assert run.status == HarvestRun.Status.SUCCESS
    assert run.mode == HarvestRun.Mode.INCREMENTAL
    assert run.from_datestamp == ""
    assert run.watermark == "2026-01-03T00:00:00Z"
    assert run.records_seen == 2


def test_incremental_run_resumes_from_watermark(harvest_env, monkeypatch):
    _install(monkeypatch, _envelope([_record("oai:repo:1")]))
    call_command("harvest_oai")

    fake = _install(monkeypatch, _envelope([]))
    call_command("harvest_oai")

    assert fake.calls[0]["from_date"] == "2026-01-02T00:00:00Z"
    # An empty window inherits the previous watermark.
    latest = HarvestRun.objects.order_by("-finished_at").first()
    assert latest.watermark == "2026-01-02T00:00:00Z"


def test_update_rebuilds_search_text(harvest_env, monkeypatch):
    _install(monkeypatch, _envelope([_record("oai:repo:1", "Old name")]))
    call_command("harvest_oai")
    _install(monkeypatch, _envelope([_record("oai:repo:1", "New name")]))
    call_command("harvest_oai")

    assert HarvestedRecord.objects.count() == 1
    row = HarvestedRecord.objects.get()
    assert row.metadata["name"] == "New name"
    assert "new name" in row.search_text and "old name" not in row.search_text


def test_deleted_record_flips_flag_and_keeps_content(harvest_env, monkeypatch):
    _install(monkeypatch, _envelope([_record("oai:repo:1", "Alpha")]))
    call_command("harvest_oai")
    _install(
        monkeypatch,
        _envelope(
            [_record("oai:repo:1", deleted=True, datestamp="2026-02-01T00:00:00Z")]
        ),
    )
    call_command("harvest_oai")

    row = HarvestedRecord.objects.get()
    assert row.deleted is True
    assert row.datestamp == "2026-02-01T00:00:00Z"
    assert row.metadata["name"] == "Alpha"  # earlier content is kept


def test_error_envelope_records_run_and_stores_nothing(harvest_env, monkeypatch):
    _install(
        monkeypatch,
        _envelope(
            [_record("oai:repo:1")],
            error="Authentication required",
            error_code="badAuthentication",
        ),
    )

    with pytest.raises(CommandError):
        call_command("harvest_oai")

    assert HarvestedRecord.objects.count() == 0
    run = HarvestRun.objects.get()
    assert run.status == HarvestRun.Status.ERROR
    assert run.watermark == ""
    assert "badAuthentication" in run.error


def test_full_run_marks_unseen_records_deleted(harvest_env, monkeypatch):
    _install(
        monkeypatch,
        _envelope([_record("oai:repo:1", "Kept"), _record("oai:repo:2", "Gone")]),
    )
    call_command("harvest_oai")

    fake = _install(monkeypatch, _envelope([_record("oai:repo:1", "Kept")]))
    call_command("harvest_oai", "--full")

    # --full ignores the watermark and re-harvests everything.
    assert fake.calls[0]["from_date"] == ""
    assert HarvestedRecord.objects.get(oai_identifier="oai:repo:1").deleted is False
    assert HarvestedRecord.objects.get(oai_identifier="oai:repo:2").deleted is True
    latest = HarvestRun.objects.order_by("-finished_at").first()
    assert latest.mode == HarvestRun.Mode.FULL
    assert latest.records_deleted == 1


def test_unconfigured_environment_is_a_quiet_noop(monkeypatch):
    monkeypatch.delenv("OAI_PMH_BASE_URL", raising=False)

    def fail_client():
        raise AssertionError("the upstream must not be contacted")

    monkeypatch.setattr("app.harvest.services.make_client", fail_client)
    call_command("harvest_oai")  # exit 0, no CommandError
    assert HarvestRun.objects.count() == 0


def test_run_history_is_pruned(harvest_env, monkeypatch):
    monkeypatch.setattr("app.harvest.management.commands.harvest_oai.KEPT_RUNS", 2)
    _install(monkeypatch, _envelope([]))
    for _ in range(4):
        call_command("harvest_oai")
    assert HarvestRun.objects.count() == 2
