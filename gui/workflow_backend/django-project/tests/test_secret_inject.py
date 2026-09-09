"""Codegen SecretRef emission, context blocklist, jupyter wrap, slurm wrapper shred."""

import pytest

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


@pytest.mark.django_db
def test_redact_run_text_uses_given_owner(user_alice, user_bob):
    from app.secrets.services import create_user_secret
    from app.workflow.models import FlowNode, FlowProject
    from app.workflow.views import _redact_run_text

    project = FlowProject.objects.create(name="run-redact", owner=user_alice)
    FlowNode.objects.create(
        id="n1",
        project=project,
        position_x=0,
        position_y=0,
        node_type="default",
        data={
            "schema": {
                "parameters": {
                    "password": {
                        "secret": True,
                        "default_value": {"__nw_secret": {"name": "FOO"}},
                    }
                }
            }
        },
    )
    create_user_secret(user_alice, name="FOO", value=FIXTURE_VALUE)
    create_user_secret(user_bob, name="FOO", value="bob-other-secret")
    assert FIXTURE_VALUE not in _redact_run_text(user_alice, project, f"out {FIXTURE_VALUE}")
    assert FIXTURE_VALUE in _redact_run_text(user_bob, project, f"out {FIXTURE_VALUE}")


@pytest.mark.django_db
def test_code_view_redacts_notebook_outputs(auth_client, user_alice, tmp_path, monkeypatch):
    import json as _json

    from django.urls import reverse

    from app.secrets.services import create_user_secret
    from app.workflow.models import FlowNode, FlowProject

    project = FlowProject.objects.create(name="nb-redact", owner=user_alice)
    FlowNode.objects.create(
        id="n1",
        project=project,
        position_x=0,
        position_y=0,
        node_type="default",
        data={
            "schema": {
                "parameters": {
                    "password": {
                        "secret": True,
                        "default_value": {"__nw_secret": {"name": "FOO"}},
                    }
                }
            }
        },
    )
    create_user_secret(user_alice, name="FOO", value=FIXTURE_VALUE)
    nb_path = tmp_path / "workflow.ipynb"
    nb_path.write_text(
        _json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["print(1)"],
                        "outputs": [
                            {"output_type": "stream", "text": f"{FIXTURE_VALUE}\n"}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.workflow.views.existing_project_dir", lambda project: tmp_path)
    monkeypatch.setattr("app.workflow.views.code_file_path", lambda project, create=False: tmp_path / "missing.py")
    monkeypatch.setattr("app.workflow.views.notebook_file_path", lambda project, create=False: nb_path)
    client = auth_client(user_alice)
    resp = client.get(reverse("workflow:workflow-code", kwargs={"workflow_id": project.id}))
    assert resp.status_code == 200
    body = resp.json()
    dumped = _json.dumps(body)
    assert FIXTURE_VALUE not in dumped
    assert body["notebook_outputs"][0]["outputs"] == ["••••"]


def test_jupyter_clears_secrets_if_kernel_create_fails(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "app.workflow.jupyter_execution_service.JUPYTERHUB_API_TOKEN", "test-token"
    )
    from app.secrets.logging_filter import _secret_values, clear_secret_values
    from app.workflow.jupyter_execution_service import JupyterExecutionService

    clear_secret_values()
    svc = JupyterExecutionService()
    svc._ensure_server_running = AsyncMock()
    svc._create_kernel = AsyncMock(side_effect=RuntimeError("kernel failed"))
    svc._delete_kernel = AsyncMock()

    async def consume():
        try:
            async for _ in svc.execute_code("print(1)", runtime_secrets={"FOO": FIXTURE_VALUE}):
                pass
        except RuntimeError:
            pass
        assert _secret_values.get() == ()

    asyncio.run(consume())
    svc._delete_kernel.assert_not_called()
