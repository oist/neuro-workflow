"""Palette listing: owner vs catalog, parse stubs, upload round-trip."""

import hashlib

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from app.box.models import PythonFile

pytestmark = pytest.mark.django_db

_VALID_NODE_SOURCE = """\
from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, PortDefinition
from neuroworkflow.core.port import PortType


class PaletteProbeNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type='palette_probe',
        description='probe node for palette tests',
        parameters={},
        inputs={},
        outputs={'ok': PortDefinition(type=PortType.STR, description='probe')},
        methods={},
    )
"""


def _media(settings, tmp_path):
    root = tmp_path / "nodes"
    (root / "analysis").mkdir(parents=True)
    settings.MEDIA_ROOT = str(root)
    return root


def _make_file(
    owner,
    *,
    name,
    hash_suffix,
    category="analysis",
    node_classes=None,
    is_analyzed=True,
    analysis_error=None,
):
    payload = f"# {hash_suffix}\n".encode("utf-8")
    upload = SimpleUploadedFile(name, payload, content_type="text/x-python")
    return PythonFile.objects.create(
        name=name,
        category=category,
        file=upload,
        file_content=payload.decode("utf-8"),
        uploaded_by=owner,
        file_size=len(payload),
        file_hash=hashlib.sha256(f"palette-test-{hash_suffix}".encode()).hexdigest(),
        is_analyzed=is_analyzed,
        analysis_error=analysis_error,
        node_classes=node_classes if node_classes is not None else {},
    )


def test_owner_valid_node_visible_bob_cannot_see(
    auth_client, user_alice, user_bob, tmp_path, settings
):
    _media(settings, tmp_path)
    own = _make_file(
        user_alice,
        name="alice.py",
        hash_suffix="alice-valid",
        node_classes={
            "Foo": {
                "description": "alice node",
                "inputs": {},
                "outputs": {},
                "parameters": {},
                "methods": {},
            }
        },
    )
    _make_file(
        user_bob,
        name="bob.py",
        hash_suffix="bob-valid",
        node_classes={
            "Bar": {
                "description": "bob node",
                "inputs": {},
                "outputs": {},
                "parameters": {},
                "methods": {},
            }
        },
    )

    url = reverse("box:uploaded-nodes")
    resp = auth_client(user_alice).get(url)
    assert resp.status_code == 200, resp.content
    body = resp.json()
    names = {n["file_name"] for n in body["nodes"]}
    assert own.name in names
    assert "bob.py" not in names

    foo = next(n for n in body["nodes"] if n["file_name"] == own.name)
    assert foo["is_own"] is True
    assert foo["parse_ok"] is True
    assert foo["draggable"] is True
    assert foo["category_key"] == "analysis"
    assert foo["label"] == "Foo"
    assert body["total_nodes"] == len(body["nodes"])


def test_owner_empty_classes_stub_catalog_empty_omitted(
    auth_client, user_alice, tmp_path, settings
):
    _media(settings, tmp_path)
    owner_fail = _make_file(
        user_alice,
        name="broken.py",
        hash_suffix="alice-empty",
        node_classes={},
        is_analyzed=True,
        analysis_error="",
    )
    catalog_empty = _make_file(
        None,
        name="__init__.py",
        hash_suffix="catalog-empty",
        node_classes={},
    )
    _make_file(
        None,
        name="catalog.py",
        hash_suffix="catalog-ok",
        node_classes={
            "Cat": {
                "description": "",
                "inputs": {},
                "outputs": {},
                "parameters": {},
                "methods": {},
            }
        },
    )

    resp = auth_client(user_alice).get(reverse("box:uploaded-nodes"))
    assert resp.status_code == 200, resp.content
    body = resp.json()
    names = {n["file_name"] for n in body["nodes"]}
    assert owner_fail.name in names
    assert catalog_empty.name not in names
    assert "catalog.py" in names

    stub = next(n for n in body["nodes"] if n["file_name"] == owner_fail.name)
    assert stub["parse_ok"] is False
    assert stub["draggable"] is False
    assert stub["is_own"] is True
    assert stub["category_key"] == "analysis"
    assert stub["label"] == "broken"
    assert "NODE_DEFINITION" in stub["description"]
    assert body["total_files"] == 2
    assert body["total_nodes"] == len(body["nodes"])


def test_upload_roundtrip_includes_class_and_category_label(
    auth_client, user_alice, tmp_path, settings
):
    _media(settings, tmp_path)
    client = auth_client(user_alice)
    source = _VALID_NODE_SOURCE + "\n# unique " + user_alice.username + "\n"
    upload = SimpleUploadedFile(
        "PaletteProbeNode.py",
        source.encode("utf-8"),
        content_type="text/x-python",
    )
    resp = client.post(
        reverse("box:python-file-upload"),
        {
            "file": upload,
            "name": "PaletteProbeNode.py",
            "category": "analysis",
        },
        format="multipart",
    )
    assert resp.status_code == 201, resp.content
    body = resp.json()
    assert body["node_classes_count"] == 1
    assert "PaletteProbeNode" in body["node_class_names"]
    assert body["category_label"]
    assert body["category"] == "analysis"
    assert body["analysis_error"] in (None, "")

    listed = client.get(reverse("box:uploaded-nodes"))
    assert listed.status_code == 200, listed.content
    labels = {n["label"] for n in listed.json()["nodes"]}
    assert "PaletteProbeNode" in labels
    probe = next(n for n in listed.json()["nodes"] if n["label"] == "PaletteProbeNode")
    assert probe["is_own"] is True
    assert probe["parse_ok"] is True
    assert probe["draggable"] is True
    assert probe["category_key"] == "analysis"


def test_owner_unanalyzed_leftover_classes_stub_bob_cannot_see(
    auth_client, user_alice, user_bob, tmp_path, settings
):
    _media(settings, tmp_path)
    leftover = {
        "Stale": {
            "description": "previous parse",
            "inputs": {},
            "outputs": {},
            "parameters": {},
            "methods": {},
        }
    }
    owner_stale = _make_file(
        user_alice,
        name="stale_reparse.py",
        hash_suffix="alice-stale",
        node_classes=leftover,
        is_analyzed=False,
        analysis_error="SyntaxError: failed re-analysis",
    )
    catalog_unanalyzed = _make_file(
        None,
        name="catalog_unanalyzed.py",
        hash_suffix="catalog-unanalyzed",
        node_classes={},
        is_analyzed=False,
        analysis_error="not scanned",
    )

    alice_resp = auth_client(user_alice).get(reverse("box:uploaded-nodes"))
    assert alice_resp.status_code == 200, alice_resp.content
    alice_body = alice_resp.json()
    alice_names = {n["file_name"] for n in alice_body["nodes"]}
    assert owner_stale.name in alice_names
    assert catalog_unanalyzed.name not in alice_names

    stubs = [n for n in alice_body["nodes"] if n["file_name"] == owner_stale.name]
    assert len(stubs) == 1
    stub = stubs[0]
    assert stub["parse_ok"] is False
    assert stub["draggable"] is False
    assert stub["is_own"] is True
    assert stub["label"] == "stale_reparse"
    assert stub["description"] == "SyntaxError: failed re-analysis"
    assert stub["class_name"] == ""

    bob_resp = auth_client(user_bob).get(reverse("box:uploaded-nodes"))
    assert bob_resp.status_code == 200, bob_resp.content
    bob_names = {n["file_name"] for n in bob_resp.json()["nodes"]}
    assert owner_stale.name not in bob_names
    assert catalog_unanalyzed.name not in bob_names
