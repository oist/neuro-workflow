import re
import os
import json
from pathlib import Path
from django.conf import settings
from .models import FlowProject, FlowNode, FlowEdge
from .path_utils import code_file_path, projects_root
import logging
import traceback
import subprocess

logger = logging.getLogger(__name__)

class RunWorkflowService:
    """A service that run the Python code generated from the workflow"""

    def __init__(self):
        self.code_dir = projects_root()

    def run_workflow_code(self, workflow_id, project_name, runtime_secrets=None):
        project = FlowProject.objects.get(id=workflow_id)
        script_path = code_file_path(project)

        logger.info("Run Workflow [%s]", project_name)

        env = os.environ.copy()
        for name, value in (runtime_secrets or {}).items():
            env[f"NW_SECRET_{name}"] = str(value)

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )

            from app.secrets.inject import redact_with_values

            values = list((runtime_secrets or {}).values())
            return {
                "stdout": redact_with_values(result.stdout, values),
                "stderr": redact_with_values(result.stderr, values),
            }

        except subprocess.CalledProcessError as e:
            from app.secrets.inject import redact_with_values

            values = list((runtime_secrets or {}).values())
            return {
                "error": str(e),
                "stdout": redact_with_values(e.stdout, values),
                "stderr": redact_with_values(e.stderr, values),
            }

