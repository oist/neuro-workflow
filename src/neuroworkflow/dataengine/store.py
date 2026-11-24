"""Where downloaded data lives, and the record kept beside it.

    <store>/<source>/<dataset key>/            the data, under the source's own names
    <store>/<source>/<dataset key>/.dataset.json  the record

The store root is, in order: the ``store`` argument, the ``NW_DATA_STORE``
environment variable, ``./data``. A dataset is downloaded once per store; every
workflow and node that names it uses the same folder.

The record says, per item, whether its download completed. Files left by an
interrupted download look exactly like finished ones; the record is how a later
reader tells them apart.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, Optional

ENV_STORE = "NW_DATA_STORE"
DEFAULT_STORE = "./data"
RECORD_NAME = ".dataset.json"


def store_root(store: Optional[str] = None) -> str:
    return os.path.abspath(os.path.expanduser(store or os.environ.get(ENV_STORE) or DEFAULT_STORE))


def dataset_dir(source: str, key: str, store: Optional[str] = None) -> str:
    return os.path.join(store_root(store), source, key)


def read_record(path: str) -> Dict[str, Any]:
    try:
        with open(os.path.join(path, RECORD_NAME)) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def write_record(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, RECORD_NAME + ".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(path, RECORD_NAME))


def local_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def missing(record: Dict[str, Any], include: Iterable[str]) -> list:
    """Items not recorded as complete. Empty include means the whole dataset."""
    include = list(include)
    if not include:
        return [] if record.get("whole_complete") else ["<whole dataset>"]
    items = record.get("items", {})
    return [n for n in include if not items.get(n, {}).get("complete")]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
