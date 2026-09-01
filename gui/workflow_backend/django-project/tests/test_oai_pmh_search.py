"""Tests for the OAI-PMH keyword search endpoint (browser -> backend).

The view reads the harvested copy in the database (filled by ``manage.py
harvest_oai``), so tests seed ``HarvestedRecord`` rows directly. Keycloak
authentication is bypassed via the ``auth_client`` fixture.
"""

import pytest
from app.harvest import views as harvest_views
from app.harvest.models import HarvestedRecord, HarvestRun
from app.harvest.services import build_search_text
from django.urls import reverse

pytestmark = pytest.mark.django_db


def _url():
    return reverse("harvest-oai-search")


def _row(
    identifier,
    name,
    description="",
    laboratory="",
    files=(),
    set_specs=("public", "dataset"),
    deleted=False,
    datestamp="2026-01-02T00:00:00Z",
):
    record = {
        "identifier": identifier,
        "metadata": {
            "name": name,
            "description": description,
            "laboratory_name": laboratory,
            "path": f"/{name}",
            "size": 123,
        },
        "files": [
            {"id": f"f-{i}", "name": fname, "mime_type": "text/plain", "size": 10}
            for i, fname in enumerate(files)
        ],
    }
    return HarvestedRecord.objects.create(
        oai_identifier=identifier,
        datestamp=datestamp,
        set_specs=list(set_specs),
        deleted=deleted,
        metadata={} if deleted else record["metadata"],
        files=[] if deleted else record["files"],
        search_text="" if deleted else build_search_text(record),
    )


def test_requires_authentication(auth_client):
    _row("oai:repo:1", "Hippocampus recordings")
    resp = auth_client().get(_url(), {"q": "x"})
    assert resp.status_code in (401, 403)


def test_search_matches_name_case_insensitively(auth_client, user_alice):
    _row("oai:repo:1", "Hippocampus recordings")
    _row("oai:repo:2", "Cortex atlas")
    resp = auth_client(user_alice).get(_url(), {"q": "HIPPOCAMPUS"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert data["results"][0]["identifier"] == "oai:repo:1"
    assert data["results"][0]["name"] == "Hippocampus recordings"
    assert data["truncated"] is False


def test_multi_term_query_requires_all_terms(auth_client, user_alice):
    _row("oai:repo:1", "Mouse cortex atlas")
    _row("oai:repo:2", "Mouse hippocampus")
    _row("oai:repo:3", "Rat cortex")
    resp = auth_client(user_alice).get(_url(), {"q": "mouse cortex"})
    assert [r["identifier"] for r in resp.json()["results"]] == ["oai:repo:1"]


def test_search_covers_description_laboratory_and_file_names(auth_client, user_alice):
    _row("oai:repo:1", "A", description="two-photon imaging")
    _row("oai:repo:2", "B", laboratory="Doya Lab")
    _row("oai:repo:3", "C", files=("spikes.csv",))
    _row("oai:repo:4", "D")
    client = auth_client(user_alice)
    for query, expected in (
        ("two-photon", "oai:repo:1"),
        ("doya", "oai:repo:2"),
        ("spikes", "oai:repo:3"),
    ):
        data = client.get(_url(), {"q": query}).json()
        assert [r["identifier"] for r in data["results"]] == [expected], query


def test_browse_mode_returns_first_records_up_to_default_limit(auth_client, user_alice):
    for i in range(30):
        _row(f"oai:repo:{i:02d}", f"Dataset {i}")
    data = auth_client(user_alice).get(_url()).json()
    assert data["count"] == harvest_views.DEFAULT_SEARCH_LIMIT
    assert data["truncated"] is True
    # Same datestamp everywhere, so ordering falls back to the identifier.
    assert data["results"][0]["identifier"] == "oai:repo:00"


def test_results_ordered_by_datestamp_descending(auth_client, user_alice):
    _row("oai:repo:old", "Old", datestamp="2025-01-01T00:00:00Z")
    _row("oai:repo:new", "New", datestamp="2026-06-01T00:00:00Z")
    data = auth_client(user_alice).get(_url()).json()
    assert [r["identifier"] for r in data["results"]] == [
        "oai:repo:new",
        "oai:repo:old",
    ]


def test_set_filter_restricts_results(auth_client, user_alice):
    _row("oai:repo:1", "Alpha", set_specs=("public", "dataset"))
    _row("oai:repo:2", "Beta", set_specs=("public", "project:bm2.0"))
    data = auth_client(user_alice).get(_url(), {"q": "", "set": "project:bm2.0"}).json()
    assert [r["identifier"] for r in data["results"]] == ["oai:repo:2"]
    assert data["set"] == "project:bm2.0"


def test_limit_is_clamped_and_marks_truncation(auth_client, user_alice):
    _row("oai:repo:1", "A")
    _row("oai:repo:2", "B")
    data = auth_client(user_alice).get(_url(), {"limit": "0"}).json()
    assert data["count"] == 1
    assert data["truncated"] is True


def test_rejects_non_integer_limit(auth_client, user_alice):
    resp = auth_client(user_alice).get(_url(), {"limit": "abc"})
    assert resp.status_code == 400


def test_skips_deleted_records(auth_client, user_alice):
    _row("oai:repo:1", "Kept dataset")
    _row("oai:repo:2", "Gone", deleted=True)
    data = auth_client(user_alice).get(_url()).json()
    assert [r["identifier"] for r in data["results"]] == ["oai:repo:1"]
    assert data["scanned"] == 1


def test_summary_projects_record_fields(auth_client, user_alice):
    _row(
        "oai:repo:1",
        "Alpha",
        description="desc",
        laboratory="Doya Lab",
        files=("a.txt", "b.txt"),
    )
    result = auth_client(user_alice).get(_url()).json()["results"][0]
    assert result == {
        "identifier": "oai:repo:1",
        "name": "Alpha",
        "description": "desc",
        "laboratory_name": "Doya Lab",
        "datestamp": "2026-01-02T00:00:00Z",
        "set_specs": ["public", "dataset"],
        "file_count": 2,
        "size": 123,
    }


def test_empty_store_is_success_with_null_harvested_at(auth_client, user_alice):
    resp = auth_client(user_alice).get(_url(), {"q": "anything"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["results"] == []
    assert data["harvested_at"] is None


def test_harvested_at_reports_latest_successful_run(auth_client, user_alice):
    success = HarvestRun.objects.create(
        status=HarvestRun.Status.SUCCESS,
        mode=HarvestRun.Mode.INCREMENTAL,
        watermark="2026-01-02T00:00:00Z",
        started_at="2026-01-02T00:00:00Z",
    )
    HarvestRun.objects.create(
        status=HarvestRun.Status.ERROR,
        mode=HarvestRun.Mode.INCREMENTAL,
        error="boom",
        started_at="2026-01-03T00:00:00Z",
    )
    data = auth_client(user_alice).get(_url()).json()
    assert data["harvested_at"] == success.finished_at.isoformat()
