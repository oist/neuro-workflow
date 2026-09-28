"""The ledger — the run's durable record and its steering seam.

Everything the engine knows is written to plain files in the run directory, and
everything that steers the engine is read from one. No sockets, no queues: an
agent, a notebook or a GUI panel participates by reading and writing files.

    run.json      manifest, written once
    trials.jsonl  one line per evaluation, append-only
    status.json   replaced each generation, the cheap poll
    control.json  written by a human or agent to steer the loop

Write discipline: whole-file writes go through a temp file and ``os.replace`` so a
reader never sees half a document; ``trials.jsonl`` is appended and flushed per
line, so the only partial read possible is a truncated last line.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


class Ledger:
    """Writes the run record; reads control commands."""

    def __init__(self, run_dir: str, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_path = self.run_dir / "run.json"
        self.trials_path = self.run_dir / "trials.jsonl"
        self.status_path = self.run_dir / "status.json"
        self.control_path = self.run_dir / "control.json"
        self._last_control_seq = 0

    # -- writes ----------------------------------------------------------
    def write_manifest(self, spec, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            **(extra or {}),
            **spec.to_dict(),
        }
        _write_atomic(self.run_path, payload)

    def append_trial(self, trial: Dict[str, Any]) -> None:
        with open(self.trials_path, "a") as fh:
            fh.write(
                json.dumps({"schema_version": SCHEMA_VERSION, **trial}, default=str)
                + "\n"
            )
            fh.flush()

    def write_status(self, status: Dict[str, Any]) -> None:
        _write_atomic(
            self.status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                **status,
            },
        )

    # -- reads -----------------------------------------------------------
    def read_trials(self) -> List[Dict[str, Any]]:
        """Read every logged trial, tolerating a truncated final line."""
        if not self.trials_path.exists():
            return []
        rows = []
        for line in self.trials_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially written tail line
        return rows

    def read_control(self) -> Optional[Dict[str, Any]]:
        """Return a control command if a new one has been written.

        A command is applied at most once: it is only returned when its ``seq``
        is higher than the last one acknowledged, which the writer can confirm by
        reading ``control_ack`` back from ``status.json``.
        """
        if not self.control_path.exists():
            return None
        try:
            command = json.loads(self.control_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None  # being written right now; try again next generation

        seq = command.get("seq", 0)
        if not isinstance(seq, int) or seq <= self._last_control_seq:
            return None

        self._last_control_seq = seq
        return command
