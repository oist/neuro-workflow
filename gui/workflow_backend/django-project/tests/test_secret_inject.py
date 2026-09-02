"""Codegen SecretRef emission, context blocklist, jupyter wrap, slurm wrapper shred."""

from app.secrets.inject import (
    MissingRuntimeSecrets,
    collect_secret_names_from_data,
    redact_with_values,
    wrap_jupyter_code,
)
from app.secrets.redaction import WorkflowContextBlockedError, assert_context_has_no_secrets
from app.workflow.code_generation_service import CodeGenerationService
from app.workflow.execution.remote_slurm_executor import _TEMPLATE_PATH


FIXTURE_VALUE = "super-secret-fixture-value"


def test_configure_emits_secret_ref_not_value():
    service = CodeGenerationService()
    node_data = {
        "schema": {
            "parameters": {
                "password": {
                    "secret": True,
                    "default_value": {"__nw_secret": {"id": "abc", "name": "FOO"}},
                },
                "n": {"default_value": 1},
            }
        },
        "parameter_modifications": {
            "password": {"is_modified": True},
            "n": {
                "is_modified": True,
                "field_modifications": {"default_value_original": 0},
            },
        },
    }
    # n is modified 0 -> 1
    node_data["schema"]["parameters"]["n"]["default_value"] = 2
    block = service._generate_generic_configure_block("X", node_data)
    assert "SecretRef(" in block
    assert "FOO" in block
    assert "password=" in block
    assert FIXTURE_VALUE not in block
    assert "super-secret" not in block


def test_literal_secret_value_rejected():
    service = CodeGenerationService()
    node_data = {
        "schema": {
            "parameters": {
                "password": {"secret": True, "default_value": FIXTURE_VALUE},
            }
        },
        "parameter_modifications": {"password": {"is_modified": True}},
    }
    try:
        service._generate_generic_configure_block("X", node_data)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "vault" in str(exc).lower()
        assert FIXTURE_VALUE not in str(exc)


def test_context_blocklist():
    try:
        assert_context_has_no_secrets({"aspera_pass": "x", "species": "mouse"})
        assert False, "expected block"
    except WorkflowContextBlockedError as exc:
        assert "aspera_pass" in exc.keys


def test_wrap_jupyter_does_not_write_project_paths():
    wrapped = wrap_jupyter_code("print(1)\n", {"FOO": FIXTURE_VALUE})
    assert "install_runtime_secrets" in wrapped
    assert "codes/projects" not in wrapped
    assert FIXTURE_VALUE in wrapped  # in-process wrap only


def test_redact_with_values():
    assert FIXTURE_VALUE not in redact_with_values(f"pw={FIXTURE_VALUE}", [FIXTURE_VALUE])


def test_collect_secret_names():
    names = collect_secret_names_from_data(
        {"schema": {"parameters": {"password": {"default_value": {"__nw_secret": {"name": "FOO"}}}}}}
    )
    assert names == ["FOO"]


def test_slurm_wrapper_shreds_secrets_file():
    text = _TEMPLATE_PATH.read_text()
    assert "NW_SECRETS_FILE" in text
    assert "shred" in text
    assert ".nw-secrets" in text
    assert "trap" in text


def test_missing_runtime_secrets_names_ref():
    err = MissingRuntimeSecrets(["ASPERA_PASSWORD"])
    assert "ASPERA_PASSWORD" in str(err)
    assert "s3cret" not in str(err)
