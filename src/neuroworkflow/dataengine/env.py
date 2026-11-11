"""Load the data engine's credentials from a .env file.

Only names starting with the prefix (``NW_`` by default) are loaded, so a shared
.env (e.g. ``gui/.env``, which also holds the LLM keys) does not leak its other
secrets into the Python session.
"""

from __future__ import annotations

import os
from typing import List


def _parse(path: str) -> dict:
    try:
        from dotenv import dotenv_values
        return {k: v for k, v in dotenv_values(path).items() if v is not None}
    except ImportError:
        values = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.removeprefix("export ").split("=", 1)
                values[k.strip()] = v.strip().strip("'\"")
        return values


def load_env(path: str, prefix: str = "NW_") -> List[str]:
    """Set the prefixed variables of a .env file in this process; returns their names
    (never their values)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    names = []
    for key, value in _parse(path).items():
        if key.startswith(prefix):
            os.environ[key] = value
            names.append(key)
    return names
