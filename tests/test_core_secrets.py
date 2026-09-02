"""Django-free tests for neuroworkflow.core.secrets and secret parameters."""

import os
from pathlib import Path

import pytest

from neuroworkflow.core.node import Node
from neuroworkflow.core.schema import NodeDefinitionSchema, ParameterDefinition
from neuroworkflow.core.secrets import (
    MissingSecretError,
    SecretRef,
    SecretStr,
    clear_runtime_secrets,
    install_runtime_secrets,
    load_runtime_secrets,
    resolve,
)


@pytest.fixture(autouse=True)
def _reset_runtime_secrets():
    clear_runtime_secrets()
    yield
    clear_runtime_secrets()


def test_resolve_passthrough():
    assert resolve(42) == 42
    assert resolve("plain") == "plain"


def test_resolve_secret_ref():
    install_runtime_secrets({"ASPERA_PASSWORD": "s3cret"})
    assert resolve(SecretRef("ASPERA_PASSWORD")) == "s3cret"
    assert resolve({"__nw_secret": {"id": "x", "name": "ASPERA_PASSWORD"}}) == "s3cret"


def test_missing_secret_names_ref_not_value():
    with pytest.raises(MissingSecretError, match="ASPERA_PASSWORD") as exc:
        resolve(SecretRef("ASPERA_PASSWORD"))
    assert "s3cret" not in str(exc.value)


def test_secret_str_masks_repr():
    wrapped = SecretStr("s3cret-value")
    assert str(wrapped) == "••••"
    assert "s3cret" not in repr(wrapped)
    assert wrapped.get_secret() == "s3cret-value"


def test_load_runtime_secrets_from_file(tmp_path: Path):
    path = tmp_path / "nw-secrets.json"
    path.write_text('{"FOO": "bar"}', encoding="utf-8")
    os.environ["NW_SECRETS_FILE"] = str(path)
    try:
        loaded = load_runtime_secrets()
        assert loaded["FOO"] == "bar"
        assert resolve(SecretRef("FOO")) == "bar"
    finally:
        os.environ.pop("NW_SECRETS_FILE", None)


def test_load_runtime_secrets_from_env():
    os.environ["NW_SECRET_TOKEN_X"] = "from-env"
    try:
        loaded = load_runtime_secrets()
        assert loaded["TOKEN_X"] == "from-env"
    finally:
        os.environ.pop("NW_SECRET_TOKEN_X", None)


class _SecretNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="secret_test",
        description="test",
        parameters={
            "password": ParameterDefinition(default_value="", description="pw", secret=True),
            "n": ParameterDefinition(default_value=1, description="count"),
        },
    )


def test_configure_resolves_secret_and_masks_str():
    install_runtime_secrets({"ASPERA_PASSWORD": "s3cret-value"})
    node = _SecretNode("n1")
    node.configure(password=SecretRef("ASPERA_PASSWORD"), n=2)
    assert node._parameters["n"] == 2
    assert str(node._parameters["password"]) == "••••"
    assert "s3cret-value" not in str(node)
    assert node._parameters["password"].get_secret() == "s3cret-value"


def test_configure_missing_secret_fails_closed():
    node = _SecretNode("n1")
    with pytest.raises(MissingSecretError, match="MISSING_NAME"):
        node.configure(password=SecretRef("MISSING_NAME"))
