"""Load git-tracked ``.claude`` skills and fold them into the system prompt.

Issue #52 requires the notebook agent to use the skills committed to the repo
(currently ``.claude/skills/create-node.md``). We do not reimplement Claude's
skill machinery — we read the markdown and inject it as guidance, as agreed.
"""

from __future__ import annotations

import glob
import logging
import os

logger = logging.getLogger(__name__)

_BASE_PROMPT = (
    "You are NeuroWorkflow Assistant running inside a Jupyter notebook for "
    "neuroscience simulation work. You help the user write, run, and debug "
    "code and build node-based workflows.\n\n"
    "You have these tools:\n"
    "- run_code: execute Python in the user's live kernel (shared namespace) "
    "and read back stdout/results. Prefer small, verifiable steps. Use this "
    "for Python rather than Bash.\n"
    "- Read / Write / Edit: inspect and edit files in the project workspace "
    "(under /home/jovyan/codes).\n"
    "- Bash: run shell commands when needed (not for Python — use run_code).\n"
    "The NeuroWorkflow library is importable as `neuroworkflow` (e.g. "
    "`from neuroworkflow.core.workflow import WorkflowBuilder`); node classes "
    "live under /home/jovyan/codes/nodes.\n\n"
    "Behavior:\n"
    "- Before claiming something works, run it with run_code and check the output.\n"
    "- Keep answers concise. Show the code you run.\n"
    "- Always respond in the same language the user uses.\n"
)

_MCP_PROMPT = (
    "\nYou also have workflow tools (add_node, get_flow, generate_code_batch, "
    "update_node_parameter, etc.) that operate on the user's saved workflow "
    "projects in the web app. When reading parameter values use the "
    "'default_value' field. Ask for confirmation before destructive actions "
    "(deleting nodes/edges).\n"
)

# Skills are authored against the repository layout, but this kernel only has
# the codes/ tree mounted. Correct the paths so node creation lands in the
# right place and the agent does not chase missing repo-root docs.
_ENV_PATHS_PROMPT = (
    "\n# Environment paths (this Jupyter kernel)\n"
    "Your working directory is a project folder, so always use absolute paths:\n"
    "- Node library (writable): /home/jovyan/codes/nodes/ "
    "(categories: analysis, network, simulation, stimulus, io, optimization)\n"
    "- Create new nodes under: /home/jovyan/codes/nodes/sandbox/\n"
    "- Core library: /home/jovyan/codes/neuroworkflow/\n"
    "- Workflow projects: /home/jovyan/codes/projects/\n"
    "- Skills: /home/jovyan/.claude/skills/\n"
    "If a skill refers to repository paths, map them to the above: "
    "`src/neuroworkflow/nodes/` -> `/home/jovyan/codes/nodes/`, "
    "`src/neuroworkflow/nodes/sandbox/` -> `/home/jovyan/codes/nodes/sandbox/`, "
    "`src/neuroworkflow/` -> `/home/jovyan/codes/neuroworkflow/`.\n"
    "Repository-root documents (e.g. NODE_CREATION_GUIDE.md, NODE_SCHEMA.md) are "
    "NOT mounted in this kernel — do not try to read them; rely on the skill text "
    "already provided in this prompt.\n"
)


def load_skills(skills_dir: str) -> list[tuple[str, str]]:
    """Return ``(name, markdown)`` for every ``*.md`` skill in ``skills_dir``."""
    if not skills_dir or not os.path.isdir(skills_dir):
        logger.info("Skills directory not found: %s (skipping skills)", skills_dir)
        return []
    skills = []
    for path in sorted(glob.glob(os.path.join(skills_dir, "*.md"))):
        try:
            with open(path, encoding="utf-8") as f:
                skills.append((os.path.basename(path), f.read()))
        except OSError as e:
            logger.warning("Could not read skill %s: %s", path, e)
    return skills


def build_system_prompt(
    skills_dir: str, *, with_mcp: bool, project_id: str | None = None
) -> str:
    prompt = _BASE_PROMPT
    if with_mcp:
        prompt += _MCP_PROMPT
        if project_id:
            # The relayed token only authorizes this project (workflow_id).
            prompt += (
                f"This notebook belongs to workflow project `{project_id}`; use it "
                "as `workflow_id`. The workflow tools are limited to this project.\n"
            )
    skills = load_skills(skills_dir)
    if skills:
        prompt += "\n# Available skills\n"
        prompt += (
            "The following skills are tracked in this repository. Apply the "
            "relevant one when the user's request matches it.\n"
        )
        for name, text in skills:
            prompt += f"\n## Skill: {name}\n{text}\n"
    prompt += _ENV_PATHS_PROMPT
    return prompt
