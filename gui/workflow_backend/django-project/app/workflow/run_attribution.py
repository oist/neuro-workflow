"""Attribute streamed run output to canvas nodes and persist figures.

The generated workflow script runs as a single Jupyter cell, so iopub
messages carry no node identity. The core execution loop prints
``Executing node: <var_name>`` before each node, and ipykernel flushes
stdout before publishing display messages, so tracking the latest marker
attributes every ``image`` event to the node that produced it.
"""

import base64
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FIGURES_SUBDIR = Path("results") / "figures"
MANIFEST_FILENAME = "manifest.json"
UNATTRIBUTED_DIR = "_unattributed"


def load_var_to_node(project_dir: Path) -> dict:
    """Load the var_name -> node_id map written at code-generation time.

    Missing or corrupt maps degrade to empty (all images unattributed).
    """
    node_map_file = project_dir / "node_map.json"
    try:
        with open(node_map_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        var_to_node = data.get("var_to_node", {})
        if not isinstance(var_to_node, dict):
            raise ValueError("var_to_node is not a dict")
        return var_to_node
    except FileNotFoundError:
        logger.warning(
            "node_map.json not found in %s; images will be unattributed", project_dir
        )
        return {}
    except Exception as e:
        logger.warning("Failed to load node_map.json from %s: %s", project_dir, e)
        return {}


class NodeAttributor:
    """Track the currently-executing node from ``Executing node:`` stdout
    markers, tag image events with the owning React Flow node id, and tee
    figures to ``results/figures/`` for reloads.

    ``process_event`` returns the list of events to emit for one upstream
    event (the original event, possibly augmented, plus any synthesized
    ``node_executing`` events).
    """

    # Anchored at line start so "Error executing node: X" never matches.
    MARKER_RE = re.compile(r"^Executing node: (.+?)\s*$")

    def __init__(self, var_to_node: dict, figures_dir: Path | None = None):
        self._var_to_node = var_to_node
        self._current_var = None
        self._current_node_id = None
        # Partial-line carry buffer: a marker can split across stream chunks.
        self._stdout_tail = ""
        # Per-node figure counters, keyed by node_id (None = unattributed).
        self._figure_counts: dict = {}
        self._figures_dir = figures_dir
        self._manifest_entries: list = []
        if self._figures_dir is not None:
            # Latest-run semantics: wipe the previous run's figures up front,
            # even if this run ends up producing none.
            shutil.rmtree(self._figures_dir, ignore_errors=True)

    def process_event(self, event: dict) -> list:
        etype = event.get("type")
        if etype == "stdout":
            return self._process_stdout(event)
        if etype == "image":
            self._process_image(event)
        return [event]

    def _process_stdout(self, event: dict) -> list:
        events = [event]
        text = self._stdout_tail + str(event.get("data", {}).get("content", ""))
        lines = text.split("\n")
        # The last element is either "" (text ended with a newline) or an
        # incomplete line; carry it over to the next chunk.
        self._stdout_tail = lines.pop()
        for line in lines:
            match = self.MARKER_RE.match(line)
            if not match:
                continue
            self._current_var = match.group(1)
            self._current_node_id = self._var_to_node.get(self._current_var)
            events.append(
                {
                    "type": "node_executing",
                    "data": {
                        "node_name": self._current_var,
                        "node_id": self._current_node_id,
                    },
                }
            )
        return events

    def _process_image(self, event: dict) -> None:
        data = event.setdefault("data", {})
        index = self._figure_counts.get(self._current_node_id, 0)
        self._figure_counts[self._current_node_id] = index + 1
        data["node_id"] = self._current_node_id
        data["node_name"] = self._current_var
        data["figure_index"] = index
        if self._figures_dir is not None:
            self._save_figure(data, index)

    def _save_figure(self, data: dict, index: int) -> None:
        try:
            node_dir = (
                re.sub(r"[^A-Za-z0-9_-]", "_", self._current_node_id)
                if self._current_node_id
                else UNATTRIBUTED_DIR
            )
            target_dir = self._figures_dir / node_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"fig_{index:03d}.png"
            target.write_bytes(base64.b64decode(data.get("content", "")))
            self._manifest_entries.append(
                {
                    "node_id": self._current_node_id,
                    "node_name": self._current_var,
                    "path": str(FIGURES_SUBDIR / node_dir / target.name),
                    "index": index,
                }
            )
        except Exception as e:
            logger.warning("Failed to persist run figure: %s", e)

    def write_manifest(self, status: str) -> None:
        """Write the figure manifest; called from the stream's finally block
        so it runs on normal completion and on client abort alike."""
        if self._figures_dir is None:
            return
        try:
            self._figures_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "version": 1,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "figures": self._manifest_entries,
            }
            with open(
                self._figures_dir / MANIFEST_FILENAME, "w", encoding="utf-8"
            ) as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write figure manifest: %s", e)
