"""Runtime configuration for the in-notebook Claude chat agent.

Values come from the environment that the JupyterHub spawner injects into the
single-user container. Per-user MCP workflow tools need the user's Keycloak
token, which the kernel never holds: the app relays it to the backend while the
project's Jupyter tab is open, and the backend uses it on the kernel's behalf
(keyed by ``project_id``, derived from the notebook's project folder).
``user_token`` remains as a manual override.

The agent uses the Claude Agent SDK, which drives the ``claude`` CLI. The CLI
reaches Anthropic through the backend proxy (``ANTHROPIC_BASE_URL``) and presents
the shared service token as its API key, so the real Anthropic key never lives
in the user-accessible kernel.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class AgentConfig:
    backend_url: str
    service_token: str
    user_token: str | None
    skills_dir: str
    project_id: str | None
    anthropic_base_url: str
    anthropic_model: str | None
    workspace_root: str

    @property
    def has_mcp(self) -> bool:
        return bool(self.user_token) or bool(self.project_id and self.service_token)

    def cli_env(self) -> dict[str, str]:
        """Environment passed to the ``claude`` CLI subprocess.

        ``ANTHROPIC_BASE_URL`` points at the backend proxy and
        ``ANTHROPIC_API_KEY`` carries the shared service token (sent as
        ``x-api-key``). The backend validates it and swaps in the real key.
        """
        env = {
            "ANTHROPIC_BASE_URL": self.anthropic_base_url,
            "ANTHROPIC_API_KEY": self.service_token,
            # Keep the sandboxed kernel from making non-essential outbound calls.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
        if self.anthropic_model:
            env["ANTHROPIC_MODEL"] = self.anthropic_model
        return env


def _project_id_from_cwd(workspace_root: str) -> str | None:
    """Project id when the kernel runs inside ``<workspace_root>/projects/<uuid>``."""
    pattern = rf"{re.escape(workspace_root)}/projects/([0-9a-fA-F-]{{36}})(?:/|$)"
    match = re.match(pattern, os.getcwd())
    return match.group(1) if match else None


def get_config(
    *,
    user_token: str | None = None,
    project_id: str | None = None,
) -> AgentConfig:
    backend_url = os.environ.get(
        "NEUROWORKFLOW_BACKEND_URL", "http://backend:3000"
    ).rstrip("/")
    workspace_root = os.environ.get(
        "NEUROWORKFLOW_WORKSPACE_ROOT", "/home/jovyan/codes"
    )
    return AgentConfig(
        backend_url=backend_url,
        # The shared backend service token, injected by the spawner under a
        # non-reserved name (JUPYTERHUB_API_TOKEN in the kernel is the
        # per-server hub token, a different value).
        service_token=os.environ.get("NEUROWORKFLOW_SERVICE_TOKEN", ""),
        user_token=user_token or os.environ.get("NEUROWORKFLOW_USER_TOKEN") or None,
        skills_dir=os.environ.get(
            "NEUROWORKFLOW_SKILLS_DIR", "/home/jovyan/.claude/skills"
        ),
        project_id=(
            project_id
            or os.environ.get("NEUROWORKFLOW_PROJECT_ID")
            or _project_id_from_cwd(workspace_root)
        ),
        anthropic_base_url=(
            os.environ.get("ANTHROPIC_BASE_URL")
            or f"{backend_url}/api/chat/anthropic"
        ),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL") or None,
        workspace_root=workspace_root,
    )
