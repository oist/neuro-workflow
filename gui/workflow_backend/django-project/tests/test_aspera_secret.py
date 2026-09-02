"""Aspera temp YAML is shredded; schema marks password as secret."""

import sys
from pathlib import Path
from unittest.mock import patch

from app.box.services.python_analyzer import PythonNodeAnalyzer
from neuroworkflow.core.secrets import SecretRef, clear_runtime_secrets, install_runtime_secrets

CODES = Path(__file__).resolve().parents[1] / "codes"
ASPERA_SRC = CODES / "nodes" / "io" / "AsperaSharesLoaderNode.py"


def test_analyzer_sees_secret_password(tmp_path):
    analyzer = PythonNodeAnalyzer(db_path=str(tmp_path / "nodes.db"))
    nodes = analyzer.analyze_file_content(ASPERA_SRC.read_text(encoding="utf-8"))
    params = nodes[0]["parameters"]
    assert params["password"].get("secret") is True
    assert params["username"].get("secret") in (None, False)


def test_aspera_temp_yaml_shredded(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    if str(CODES) not in sys.path:
        sys.path.insert(0, str(CODES))
    install_runtime_secrets({"ASPERA_PASSWORD": "aspera-fixture-secret"})
    try:
        from nodes.io.AsperaSharesLoaderNode import AsperaSharesLoaderNode

        node = AsperaSharesLoaderNode("aspera")
        node.configure(
            username="user",
            password=SecretRef("ASPERA_PASSWORD"),
            url="https://example.invalid",
            remote_path="/remote",
            local_path=str(tmp_path / "out"),
        )
        with patch("nodes.io.AsperaSharesLoaderNode.shutil.which", return_value="/bin/true"), patch(
            "nodes.io.AsperaSharesLoaderNode.subprocess.run"
        ) as run:
            node.download()
        leftovers = list(tmp_path.glob("nw-aspera-*"))
        assert leftovers == []
        assert "aspera-fixture-secret" not in str(run.call_args)
    finally:
        clear_runtime_secrets()
