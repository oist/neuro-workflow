"""Offline tests for neuroworkflow.utils.oai_pmh and the OAI-PMH database nodes.

No network: responses are fixture XML strings modelled on the RIKEN MDRS
repository (``mdrs`` payload, ``errorCode`` attribute, HTTP-200 errors).
"""

import json

import pytest

from neuroworkflow.nodes.database.OAIPMHDownloadNode import OAIPMHDownloadNode
from neuroworkflow.nodes.database.OAIPMHRecordsNode import (
    OAIPMHRecordsNode,
    _split_identifiers,
)
from neuroworkflow.utils import oai_pmh
from neuroworkflow.utils.oai_pmh import OAIPMHClient, OAIPMHError

OAI_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
    "<responseDate>2026-08-30T16:37:05Z</responseDate>"
    '<request verb="ListRecords">https://repo.example/api/oai/</request>'
)


def _mdrs_record(identifier, name, files_xml, parent=True):
    parent_xml = "<parent><id>p-1</id><name>Parent</name></parent>" if parent else ""
    return (
        "<record><header>"
        f"<identifier>{identifier}</identifier>"
        "<datestamp>2023-06-26T00:27:16Z</datestamp>"
        "<setSpec>public</setSpec><setSpec>dataset</setSpec>"
        "</header><metadata>"
        ' <mdrs xmlns="https://www.ni.riken.jp/oai/mdrs/"><folder>'
        f"<id>{identifier.rsplit(':', 1)[-1]}</id><name>{name}</name>"
        "<description/><access_level>2</access_level>"
        "<laboratory_name>McHugh</laboratory_name>"
        "<updated_at>2023-06-26T00:27:16Z</updated_at>"
        "<created_at>2023-05-26T06:16:58Z</created_at>"
        f"<path>/Repository/{name}/</path><size>49652533367</size>"
        f"{parent_xml}"
        '<metadata>[{"no": 0, "key": "MDRS_00000842", "value": "A title"}]</metadata>'
        f"<files>{files_xml}</files>"
        "</folder></mdrs> </metadata></record>"
    )


def _file(file_id, name, size="1325"):
    return (
        f"<file><id>{file_id}</id><name>{name}</name><description/><type/>"
        f"<mime_type>text/plain</mime_type><size>{size}</size>"
        "<created_at>2023-06-23T05:39:40Z</created_at>"
        "<updated_at>2024-01-10T16:26:18Z</updated_at></file>"
    )


DELETED_RECORD = (
    '<record><header status="deleted">'
    "<identifier>oai:example.com:folder:gone</identifier>"
    "<datestamp>2024-01-01T00:00:00Z</datestamp></header></record>"
)

PAGE_1 = (
    OAI_HEAD
    + "<ListRecords>"
    + _mdrs_record(
        "oai:example.com:folder:a0ba497a",
        "pratap_stress",
        _file("f-1", "readme.txt") + _file("f-2", "data.bin", size="20"),
    )
    + DELETED_RECORD
    + '<resumptionToken expirationDate="2026-08-31T16:37:19Z" completeListSize="35">'
    "tok-1</resumptionToken></ListRecords></OAI-PMH>"
)

PAGE_2 = (
    OAI_HEAD
    + "<ListRecords>"
    + _mdrs_record("oai:example.com:folder:b1", "second", _file("f-3", "b.txt"))
    + _mdrs_record("oai:example.com:folder:c2", "third", "", parent=False)
    + "<resumptionToken/></ListRecords></OAI-PMH>"
)

OAI_DC_PAGE = (
    OAI_HEAD + "<ListRecords><record><header>"
    "<identifier>oai:example.com:folder:a0ba497a</identifier>"
    "<datestamp>2023-06-26T00:27:16Z</datestamp><setSpec>public</setSpec>"
    "</header><metadata> "
    '<oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/oai_dc/" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/">'
    "<dc:title>pratap_stress</dc:title><dc:publisher>McHugh</dc:publisher>"
    "<dc:type>Dataset</dc:type>"
    "<dc:identifier>oai:example.com:folder:a0ba497a</dc:identifier>"
    "</oai_dc:dc> </metadata></record></ListRecords></OAI-PMH>"
)

AUTH_ERROR = (
    OAI_HEAD
    + '<error errorCode="badAuthentication">Authentication required for OAI-PMH'
    "</error></OAI-PMH>"
)

NO_RECORDS = OAI_HEAD + '<error code="noRecordsMatch"/></OAI-PMH>'


@pytest.fixture
def proxy_env(monkeypatch):
    monkeypatch.delenv("OAI_PMH_BASE_URL", raising=False)
    monkeypatch.setenv("NEUROWORKFLOW_BACKEND_URL", "http://backend:3000/")
    monkeypatch.setenv("NEUROWORKFLOW_SERVICE_TOKEN", "svc-token")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_records_mdrs_page():
    records, token, total = oai_pmh.parse_records(
        oai_pmh.parse_response(PAGE_1.encode()), "mdrs"
    )

    assert token == "tok-1"
    assert total == 35
    assert [r["identifier"] for r in records] == [
        "oai:example.com:folder:a0ba497a",
        "oai:example.com:folder:gone",
    ]

    first = records[0]
    assert first["deleted"] is False
    assert first["set_specs"] == ["public", "dataset"]
    assert first["metadata_prefix"] == "mdrs"
    assert first["metadata"]["name"] == "pratap_stress"
    assert first["metadata"]["size"] == 49652533367
    assert first["metadata"]["parent"] == {"id": "p-1", "name": "Parent"}
    assert first["metadata"]["metadata"] == [
        {"no": 0, "key": "MDRS_00000842", "value": "A title"}
    ]
    assert first["files"] == [
        {"id": "f-1", "name": "readme.txt", "mime_type": "text/plain", "size": 1325},
        {"id": "f-2", "name": "data.bin", "mime_type": "text/plain", "size": 20},
    ]

    deleted = records[1]
    assert deleted["deleted"] is True
    assert deleted["metadata"] is None
    assert deleted["files"] == []


def test_parse_records_oai_dc_flattens_by_local_tag():
    records, token, total = oai_pmh.parse_records(
        oai_pmh.parse_response(OAI_DC_PAGE.encode()), "oai_dc"
    )
    assert token is None and total is None
    assert records[0]["metadata"] == {
        "title": ["pratap_stress"],
        "publisher": ["McHugh"],
        "type": ["Dataset"],
        "identifier": ["oai:example.com:folder:a0ba497a"],
    }
    assert records[0]["files"] == []


def test_parse_response_detects_error_element_in_http_200_body():
    with pytest.raises(OAIPMHError) as excinfo:
        oai_pmh.parse_response(AUTH_ERROR.encode())
    assert excinfo.value.code == "badAuthentication"
    assert "Authentication required" in str(excinfo.value)


def test_parse_response_non_xml_reports_bad_response_with_snippet():
    with pytest.raises(OAIPMHError) as excinfo:
        oai_pmh.parse_response(b'{"error": "Invalid service token"}')
    assert excinfo.value.code == "bad_response"
    assert "Invalid service token" in str(excinfo.value)


def test_mdrs_metadata_keeps_raw_text_when_json_is_invalid():
    page = (
        OAI_HEAD
        + "<GetRecord>"
        + _mdrs_record("oai:x:folder:1", "n", "")
        + "</GetRecord></OAI-PMH>"
    ).replace('[{"no": 0, "key": "MDRS_00000842", "value": "A title"}]', "not json")
    records, _, _ = oai_pmh.parse_records(oai_pmh.parse_response(page.encode()), "mdrs")
    assert records[0]["metadata"]["metadata"] == "not json"


# ---------------------------------------------------------------------------
# Endpoint resolution and request building
# ---------------------------------------------------------------------------


def test_resolve_endpoint_proxy_mode(proxy_env):
    oai_url, file_url, headers = oai_pmh.resolve_endpoint()
    assert oai_url == "http://backend:3000/api/harvest/oai/"
    assert file_url == "http://backend:3000/api/harvest/oai/files/{file_id}/download/"
    assert headers == {"X-Api-Key": "svc-token"}


def test_resolve_endpoint_direct_mode(monkeypatch):
    monkeypatch.setenv("OAI_PMH_BASE_URL", "https://repo.example/api/oai")
    monkeypatch.setenv("OAI_PMH_API_KEY", "k")
    monkeypatch.delenv("OAI_PMH_API_KEY_HEADER", raising=False)
    monkeypatch.setenv("OAI_PMH_FILE_DOWNLOAD_URL", "https://repo.example/f/{file_id}")
    oai_url, file_url, headers = oai_pmh.resolve_endpoint()
    assert oai_url == "https://repo.example/api/oai/"
    assert file_url == "https://repo.example/f/{file_id}"
    assert headers == {"X-MDRS-API-Key": "k"}


def test_request_omits_empty_arguments(proxy_env, monkeypatch):
    seen = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return PAGE_2.encode()

    client = OAIPMHClient()
    monkeypatch.setattr(client, "_open", lambda url: seen.append(url) or _Resp())
    client.request(
        "ListRecords", **{"metadataPrefix": "mdrs", "set": "", "from": "2020-01-01"}
    )
    assert seen == [
        "http://backend:3000/api/harvest/oai/"
        "?verb=ListRecords&metadataPrefix=mdrs&from=2020-01-01"
    ]


# ---------------------------------------------------------------------------
# list_records / get_record envelopes
# ---------------------------------------------------------------------------


def _fake_request(pages):
    calls = []

    def request(verb, **args):
        calls.append((verb, args))
        page = pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return oai_pmh.parse_response(page.encode())

    return request, calls


def test_list_records_follows_resumption_token_and_caps(proxy_env, monkeypatch):
    client = OAIPMHClient()
    request, calls = _fake_request([PAGE_1, PAGE_2])
    monkeypatch.setattr(client, "request", request)

    env = client.list_records("mdrs", set_spec="dataset", max_records=3)

    assert env["status"] == "success"
    assert env["count"] == 3 and len(env["records"]) == 3
    assert env["total"] == 35
    assert calls[0] == (
        "ListRecords",
        {"metadataPrefix": "mdrs", "set": "dataset", "from": "", "until": ""},
    )
    assert calls[1] == ("ListRecords", {"resumptionToken": "tok-1"})


def test_list_records_stops_at_cap_without_fetching_next_page(proxy_env, monkeypatch):
    client = OAIPMHClient()
    request, calls = _fake_request([PAGE_1])
    monkeypatch.setattr(client, "request", request)
    env = client.list_records("mdrs", max_records=2)
    assert env["count"] == 2 and len(calls) == 1


def test_list_records_no_records_match_is_empty_success(proxy_env, monkeypatch):
    client = OAIPMHClient()
    request, _ = _fake_request([OAIPMHError("none", "noRecordsMatch")])
    monkeypatch.setattr(client, "request", request)
    env = client.list_records("mdrs")
    assert env == {
        "status": "success",
        "records": [],
        "count": 0,
        "total": None,
        "error": None,
        "error_code": None,
    }


def test_list_records_error_keeps_partial_results(proxy_env, monkeypatch):
    client = OAIPMHClient()
    request, _ = _fake_request([PAGE_1, OAIPMHError("HTTP 503", "http_503")])
    monkeypatch.setattr(client, "request", request)
    env = client.list_records("mdrs", max_records=10)
    assert env["status"] == "error"
    assert env["error_code"] == "http_503"
    assert env["count"] == 2


def test_get_record_returns_error_envelope(proxy_env, monkeypatch):
    client = OAIPMHClient()
    request, _ = _fake_request([OAIPMHError("bad", "badAuthentication")])
    monkeypatch.setattr(client, "request", request)
    env = client.get_record("oai:x:folder:1")
    assert env["status"] == "error" and env["records"] == []


# ---------------------------------------------------------------------------
# fetch_backend_records
# ---------------------------------------------------------------------------


class _JSONResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


def test_fetch_backend_records_requests_backend_with_service_token(
    proxy_env, monkeypatch
):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["token"] = req.get_header("X-api-key")
        seen["timeout"] = timeout
        return _JSONResponse(
            {
                "status": "success",
                "records": [{"identifier": "oai:x:1"}],
                "count": 1,
                "missing": [],
            }
        )

    monkeypatch.setattr(oai_pmh.urllib.request, "urlopen", fake_urlopen)
    env = oai_pmh.fetch_backend_records(["oai:x:1", "oai:x:2"], timeout=7)

    assert seen["url"] == (
        "http://backend:3000/api/harvest/records/"
        "?identifiers=oai%3Ax%3A1%2Coai%3Ax%3A2"
    )
    assert seen["token"] == "svc-token" and seen["timeout"] == 7.0
    assert env["status"] == "success"
    assert env["count"] == 1 and env["total"] == 2


def test_fetch_backend_records_reports_missing_identifiers(proxy_env, monkeypatch):
    monkeypatch.setattr(
        oai_pmh.urllib.request,
        "urlopen",
        lambda req, timeout=None: _JSONResponse(
            {
                "status": "success",
                "records": [{"identifier": "oai:x:1"}],
                "count": 1,
                "missing": ["oai:x:bad"],
            }
        ),
    )
    env = oai_pmh.fetch_backend_records(["oai:x:1", "oai:x:bad"])
    assert env["status"] == "error"
    assert env["error_code"] == "not_found"
    assert "oai:x:bad" in env["error"]
    assert [r["identifier"] for r in env["records"]] == ["oai:x:1"]


def test_fetch_backend_records_transport_error_envelope(proxy_env, monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise oai_pmh.urllib.error.URLError("connection refused")

    monkeypatch.setattr(oai_pmh.urllib.request, "urlopen", fake_urlopen)
    env = oai_pmh.fetch_backend_records(["oai:x:1"])
    assert env["status"] == "error"
    assert env["error_code"] == "transport"
    assert env["records"] == []


# ---------------------------------------------------------------------------
# Transport resilience (slow MDRS pages)
# ---------------------------------------------------------------------------


def test_open_retries_transport_errors(proxy_env, monkeypatch):
    client = OAIPMHClient(max_retries=2)
    attempts = []
    ok = object()

    def fake_urlopen(req, timeout=None):
        attempts.append(req.full_url)
        if len(attempts) < 3:
            raise oai_pmh.urllib.error.URLError("timed out")
        return ok

    monkeypatch.setattr(oai_pmh.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(oai_pmh.time, "sleep", lambda s: None)
    assert client._open("http://backend:3000/x") is ok
    assert len(attempts) == 3


def test_open_raises_transport_after_retries_exhausted(proxy_env, monkeypatch):
    client = OAIPMHClient(max_retries=1)

    def fake_urlopen(req, timeout=None):
        raise oai_pmh.urllib.error.URLError("connection refused")

    monkeypatch.setattr(oai_pmh.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(oai_pmh.time, "sleep", lambda s: None)
    with pytest.raises(OAIPMHError) as excinfo:
        client._open("http://backend:3000/x")
    assert excinfo.value.code == "transport"


def test_request_wraps_stalled_body_read(proxy_env, monkeypatch):
    class _StallingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise TimeoutError("The read operation timed out")

    client = OAIPMHClient()
    monkeypatch.setattr(client, "_open", lambda url: _StallingResponse())
    with pytest.raises(OAIPMHError) as excinfo:
        client.request("ListRecords", metadataPrefix="mdrs")
    assert excinfo.value.code == "transport"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self, fail_ids=()):
        self.downloads = []
        self.fail_ids = set(fail_ids)

    def download_file(self, file_id, dest_path):
        if file_id in self.fail_ids:
            raise OAIPMHError("HTTP 404", "http_404")
        import os

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as fh:
            fh.write(b"x")
        self.downloads.append((file_id, dest_path))
        return dest_path


def _download_node(tmp_path, stub, **params):
    node = OAIPMHDownloadNode("dl")
    node._context["results_path"] = str(tmp_path)
    node._parameters.update(params)
    node._make_client = lambda: stub
    return node


def _fake_backend_fetch(calls, missing=()):
    missing = set(missing)

    def fake(identifiers, timeout=30.0):
        calls.append((list(identifiers), timeout))
        records = [{"identifier": i} for i in identifiers if i not in missing]
        absent = [i for i in identifiers if i in missing]
        return {
            "status": "error" if absent else "success",
            "records": records,
            "count": len(records),
            "total": len(identifiers),
            "error": "not found: " + ", ".join(absent) if absent else None,
            "error_code": "not_found" if absent else None,
        }

    return fake


def test_records_node_definition_has_no_address_or_key_parameters():
    params = OAIPMHRecordsNode.NODE_DEFINITION.parameters
    assert set(params) == {"identifiers", "timeout"}
    assert OAIPMHRecordsNode("r").get_info()["name"] == "r"
    assert OAIPMHDownloadNode("d").get_info()["name"] == "d"


def test_split_identifiers_handles_separators_and_duplicates():
    raw = "oai:x:1, oai:x:2\noai:x:1\n , oai:x:3,"
    assert _split_identifiers(raw) == ["oai:x:1", "oai:x:2", "oai:x:3"]
    assert _split_identifiers("") == []


def test_records_node_fetches_identifiers_from_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "neuroworkflow.nodes.database.OAIPMHRecordsNode.fetch_backend_records",
        _fake_backend_fetch(calls),
    )
    node = OAIPMHRecordsNode("r")
    node._parameters.update({"identifiers": "oai:x:1, oai:x:2\noai:x:1"})
    out = node.fetch()

    assert calls == [(["oai:x:1", "oai:x:2"], 30)]
    assert [r["identifier"] for r in out["records"]] == ["oai:x:1", "oai:x:2"]
    assert out["metadata"] == {
        "status": "success",
        "count": 2,
        "total": 2,
        "error": None,
        "error_code": None,
    }


def test_records_node_chunks_backend_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "neuroworkflow.nodes.database.OAIPMHRecordsNode.fetch_backend_records",
        _fake_backend_fetch(calls),
    )
    node = OAIPMHRecordsNode("r")
    node._parameters.update({"identifiers": ",".join(f"oai:x:{i}" for i in range(150))})
    out = node.fetch()

    assert [len(identifiers) for identifiers, _ in calls] == [100, 50]
    assert out["metadata"]["count"] == 150 and out["metadata"]["total"] == 150


def test_records_node_reports_missing_identifiers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "neuroworkflow.nodes.database.OAIPMHRecordsNode.fetch_backend_records",
        _fake_backend_fetch(calls, missing={"oai:x:bad"}),
    )
    node = OAIPMHRecordsNode("r")
    node._parameters.update({"identifiers": "oai:x:1\noai:x:bad"})
    out = node.fetch()

    metadata = out["metadata"]
    assert metadata["status"] == "error"
    assert metadata["error_code"] == "not_found"
    assert metadata["error"] == "not found: oai:x:bad"
    assert [r["identifier"] for r in out["records"]] == ["oai:x:1"]
    assert metadata["count"] == 1 and metadata["total"] == 2


def test_records_node_empty_identifiers_is_empty_success(monkeypatch):
    def fail_fetch(identifiers, timeout=30.0):
        raise AssertionError("the backend must not be called without identifiers")

    monkeypatch.setattr(
        "neuroworkflow.nodes.database.OAIPMHRecordsNode.fetch_backend_records",
        fail_fetch,
    )
    out = OAIPMHRecordsNode("r").fetch()
    assert out["records"] == []
    assert out["metadata"]["status"] == "success"
    assert out["metadata"]["count"] == 0 and out["metadata"]["total"] == 0


def test_download_node_writes_files_and_sanitizes_names(tmp_path):
    stub = _StubClient()
    node = _download_node(tmp_path, stub)
    records = [
        {
            "identifier": "oai:example.com:folder:a0ba497a",
            "deleted": False,
            "metadata": {"name": "My Folder/../x"},
            "files": [
                {"id": "f-1", "name": "../evil/name.csv"},
                {"id": "f-2", "name": "readme.txt"},
                {"id": "f-3", "name": "readme.txt"},  # same name, different id
            ],
        },
        {"identifier": "oai:example.com:folder:gone", "deleted": True, "files": []},
    ]

    out = node.download(records)

    base = tmp_path / "oai_pmh" / "My_Folder_.._x"
    assert out["download_metadata"]["status"] == "success"
    assert out["download_metadata"]["downloaded"] == 3
    assert out["file_paths"] == [
        str(base / "evil_name.csv"),
        str(base / "readme.txt"),
        str(base / "f-3_readme.txt"),
    ]
    for path in out["file_paths"]:
        assert path.startswith(str(tmp_path / "oai_pmh"))
        assert (tmp_path / path).exists()


def test_download_node_resolves_missing_file_lists_from_backend(tmp_path, monkeypatch):
    resolved = {
        "oai:example.com:folder:a0ba497a": {
            "identifier": "oai:example.com:folder:a0ba497a",
            "metadata": {"name": "resolved_folder"},
            "files": [{"id": "f-9", "name": "a.txt"}],
        }
    }
    calls = []

    def fake_fetch(identifiers, timeout=30.0):
        calls.append(list(identifiers))
        record = resolved.get(identifiers[0])
        return {
            "status": "success" if record else "error",
            "records": [record] if record else [],
            "count": 1 if record else 0,
            "total": len(identifiers),
            "error": None if record else "not found",
            "error_code": None if record else "not_found",
        }

    monkeypatch.setattr(
        "neuroworkflow.nodes.database.OAIPMHDownloadNode.fetch_backend_records",
        fake_fetch,
    )
    node = _download_node(tmp_path, _StubClient())
    records = [
        {"identifier": "oai:example.com:folder:a0ba497a", "metadata": {"title": ["t"]}},
        {"identifier": "oai:example.com:folder:unknown", "metadata": {}},
    ]

    out = node.download(records)

    assert calls == [
        ["oai:example.com:folder:a0ba497a"],
        ["oai:example.com:folder:unknown"],
    ]
    assert out["file_paths"] == [
        str(tmp_path / "oai_pmh" / "resolved_folder" / "a.txt")
    ]
    assert out["download_metadata"]["records_without_files"] == 1


def test_download_node_limits_skips_and_reports_failures(tmp_path):
    stub = _StubClient(fail_ids={"f-2"})
    node = _download_node(tmp_path, stub, max_files_per_record=2)
    existing = tmp_path / "oai_pmh" / "folder" / "have.txt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")
    records = [
        {
            "identifier": "oai:x:folder:1",
            "metadata": {"name": "folder"},
            "files": [
                {"id": "f-1", "name": "have.txt"},
                {"id": "f-2", "name": "broken.txt"},
                {"id": "f-3", "name": "never.txt"},
            ],
        }
    ]

    out = node.download(records)
    meta = out["download_metadata"]

    assert meta["status"] == "error"
    assert (meta["skipped"], meta["failed"], meta["downloaded"]) == (1, 1, 0)
    assert meta["errors"] == [
        {"identifier": "oai:x:folder:1", "file_id": "f-2", "error": "HTTP 404"}
    ]
    assert out["file_paths"] == [str(existing)]
    assert existing.read_bytes() == b"old"
    assert stub.downloads == []


def test_download_node_handles_no_input():
    out = OAIPMHDownloadNode("dl").download(None)
    assert out["file_paths"] == []
    assert out["download_metadata"]["status"] == "success"
