"""Load files from an IBM Aspera Shares source.

Passwords are vault SecretRefs (secret=True). At runtime the node writes a
mode-0600 YAML under TMPDIR (never under results/), runs the transfer, then
overwrites and deletes the YAML. Operators may still set ASPERA_PASSWORD.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any

from neuroworkflow.core.node import Node
from neuroworkflow.core.port import PortType
from neuroworkflow.core.schema import (
    MethodDefinition,
    NodeDefinitionSchema,
    ParameterDefinition,
    PortDefinition,
)
from neuroworkflow.core.secrets import resolve


def _shred_file(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as handle:
            handle.write(b"\0" * size)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass
    try:
        os.remove(path)
    except OSError:
        pass


class AsperaSharesLoaderNode(Node):
    NODE_DEFINITION = NodeDefinitionSchema(
        type="aspera_shares_loader",
        description="Download from IBM Aspera Shares using a vault password reference",
        stage="io",
        tool="Aspera",
        parameters={
            "url": ParameterDefinition(
                default_value="",
                description="Aspera Shares base URL",
            ),
            "username": ParameterDefinition(
                default_value="",
                description="Aspera username (not secret)",
            ),
            "password": ParameterDefinition(
                default_value="",
                description="Vault secret for the Aspera password",
                secret=True,
            ),
            "remote_path": ParameterDefinition(
                default_value="",
                description="Remote path to download",
            ),
            "local_path": ParameterDefinition(
                default_value=".",
                description="Local destination directory",
            ),
            "config_path": ParameterDefinition(
                default_value="",
                description="Optional existing Aspera YAML (overrides username/password)",
            ),
        },
        outputs={
            "local_path": PortDefinition(
                type=PortType.STR,
                description="Local path of the downloaded files",
            )
        },
        methods={
            "download": MethodDefinition(
                description="Download via ascp/aspera using a shredded temp YAML",
                inputs=[],
                outputs=["local_path"],
            )
        },
    )

    def __init__(self, name: str):
        super().__init__(name)
        self._define_process_steps()

    def _define_process_steps(self) -> None:
        self.add_process_step("download", self.download, method_key="download")

    def _password(self) -> str:
        raw = self._parameters.get("password")
        value = resolve(raw)
        if value:
            return str(value)
        return os.environ.get("ASPERA_PASSWORD", "")

    def _write_temp_yaml(self, username: str, password: str, url: str) -> str:
        tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
        fd, path = tempfile.mkstemp(prefix="nw-aspera-", suffix=".yml", dir=tmpdir)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"user: {username}\n")
            handle.write("password: " + password.replace("\n", "") + "\n")
            handle.write(f"url: {url}\n")
        return path

    def _aspera_cmd(self, config_path: str, remote_path: str, local_path: str) -> list[str]:
        aspera = shutil.which("aspera")
        ascp = shutil.which("ascp")
        if aspera:
            return [aspera, "shares", "download", "--config", config_path, remote_path, local_path]
        if ascp:
            return [ascp, "-QT", remote_path, local_path]
        raise FileNotFoundError("Neither aspera nor ascp is on PATH")

    def download(self) -> dict[str, Any]:
        url = str(self._parameters.get("url") or "")
        username = str(self._parameters.get("username") or "")
        remote_path = str(self._parameters.get("remote_path") or "")
        local_path = str(self._parameters.get("local_path") or ".")
        config_path = str(self._parameters.get("config_path") or "").strip()
        owned_yaml = ""
        password = self._password()
        if not config_path:
            if not password:
                raise ValueError(
                    "Aspera password is missing. Bind a vault secret named in Settings → Secrets "
                    "or set ASPERA_PASSWORD."
                )
            owned_yaml = self._write_temp_yaml(username, password, url)
            config_path = owned_yaml
        try:
            cmd = self._aspera_cmd(config_path, remote_path, local_path)
            env = os.environ.copy()
            aspera = shutil.which("aspera")
            ascp = shutil.which("ascp")
            if not aspera and ascp and cmd and cmd[0] == ascp:
                env["ASPERA_PASSWORD"] = password
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        finally:
            if owned_yaml:
                _shred_file(owned_yaml)
        self._output_ports["local_path"].value = local_path
        return {"local_path": local_path}
