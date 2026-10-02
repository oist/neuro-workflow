"""Issue #56: declared parameter types survive the trip to the GUI and codegen."""

import hashlib

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from app.box.models import PythonFile
from app.workflow.code_generation_service import CodeGenerationService


@pytest.mark.django_db
def test_palette_schema_carries_default_value_type(user_alice, tmp_path, settings):
    settings.MEDIA_ROOT = str(tmp_path)
    payload = b"# types\n"
    pf = PythonFile.objects.create(
        name="types.py",
        category="analysis",
        file=SimpleUploadedFile("types.py", payload, content_type="text/x-python"),
        file_content=payload.decode("utf-8"),
        uploaded_by=user_alice,
        file_size=len(payload),
        file_hash=hashlib.sha256(b"param-types").hexdigest(),
        is_analyzed=True,
        node_classes={
            "TypesNode": {
                "parameters": {
                    "weight": {"default_value": 8.0},
                    "flag": {"default_value": True},
                    "count": {"default_value": 100},
                    "label": {"default_value": "x"},
                    "items": {"default_value": [1.0, 2.0]},
                    "opts": {"default_value": {"a": 1}},
                    "unset": {"default_value": None},
                    "nodefault": {"description": "no default"},
                }
            }
        },
    )
    # Read back from PostgreSQL so a JSONB round-trip is part of the check.
    pf = PythonFile.objects.get(pk=pf.pk)
    [node] = pf.get_node_classes_for_frontend()
    params = node["schema"]["parameters"]

    assert params["weight"]["default_value_type"] == "float"
    assert params["flag"]["default_value_type"] == "bool"
    assert params["count"]["default_value_type"] == "int"
    assert params["label"]["default_value_type"] == "str"
    assert params["items"]["default_value_type"] == "list"
    assert params["opts"]["default_value_type"] == "dict"
    assert "default_value_type" not in params["unset"]
    assert "default_value_type" not in params["nodefault"]


@pytest.fixture
def service(tmp_path, settings):
    settings.MEDIA_ROOT = str(tmp_path)
    return CodeGenerationService()


def test_convert_keeps_bool(service):
    assert service._convert_parameter_value(False, "k") is False
    assert service._convert_parameter_value(True, "k", "bool") is True


def test_convert_restores_declared_float(service):
    value = service._convert_parameter_value(40, "k", "float")
    assert value == 40.0 and isinstance(value, float)


def test_convert_does_not_truncate_declared_int(service):
    assert service._convert_parameter_value(8.5, "k", "int") == 8.5


def test_convert_without_declared_type_unchanged(service):
    value = service._convert_parameter_value(40, "k")
    assert value == 40 and isinstance(value, int)


def test_configure_block_renders_declared_types(service):
    node_data = {
        "schema": {
            "parameters": {
                "syn_weight": {"default_value": 40, "default_value_type": "float"},
                "allow_multapses": {
                    "default_value": False,
                    "default_value_type": "bool",
                },
            }
        },
        "parameter_modifications": {
            "syn_weight": {
                "is_modified": True,
                "field_modifications": {"default_value_original": 25.0},
            },
            "allow_multapses": {
                "is_modified": True,
                "field_modifications": {"default_value_original": True},
            },
        },
    }
    block = service._generate_generic_configure_block("Conn", node_data)
    assert "syn_weight=40.0" in block
    assert "allow_multapses=False" in block
