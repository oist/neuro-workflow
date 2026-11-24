"""ATR Aspera share (https://bmb-asp.atr.jp), through the aspera-client library.

    pip install "git+https://github.com/orrisroot/aspera-client"

Downloads run IBM's ``ascp`` program, which the client starts; listing does not need
it. Installing it is not the engine's job (``aspera setup``, see the client's
README); the engine uses ``ASPERA_ASCP`` when set (the app's Jupyter image sets it),
else the client's default location.

Credentials come from the environment, never from code::

    NW_ASPERA_USER, NW_ASPERA_PASSWORD, NW_ASPERA_URL (default https://bmb-asp.atr.jp/)

An Aspera share has paths only, no ids: a dataset is a folder on the share, named
by its path, e.g. ``{"path": "/SRPBS_OPEN"}``. Search matches folder names in the
share's top levels; there is no metadata to search.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from ..dataset import Dataset, Item
from .base import Source

DEFAULT_URL = "https://bmb-asp.atr.jp/"
ENV_URL = "NW_ASPERA_URL"
ENV_USER = "NW_ASPERA_USER"
ENV_PASSWORD = "NW_ASPERA_PASSWORD"
FOLDER_TYPES = ("directory", "container")


def _aspera():
    try:
        import aspera_client
    except ImportError as e:
        raise ImportError(
            "The ATR source needs the Aspera client: "
            'pip install "git+https://github.com/orrisroot/aspera-client"'
        ) from e
    return aspera_client


def _norm(path: str) -> str:
    return "/" + str(path).strip("/")


class AtrSource(Source):
    name = "atr"
    repository = "atr"

    def __init__(self, search_depth: int = 2):
        self.search_depth = search_depth       # folder levels searched by name
        self._client = None
        self._folders: Optional[List[Dict[str, Any]]] = None

    def ready(self) -> bool:
        return bool(os.environ.get(ENV_USER) and os.environ.get(ENV_PASSWORD))

    @property
    def client(self):
        if self._client is None:
            ac = _aspera()
            user, password = os.environ.get(ENV_USER), os.environ.get(ENV_PASSWORD)
            if not (user and password):
                raise PermissionError(f"atr needs {ENV_USER} and {ENV_PASSWORD} in the environment")
            config = ac.AsperaConfig.from_dict({"url": os.environ.get(ENV_URL) or DEFAULT_URL,
                                                "user": user, "password": password})
            self._client = ac.AsperaConnection(config=config, env=ac.AsperaEnvironment())
        return self._client

    def _list(self, path: str) -> List[Dict[str, Any]]:
        entries, _ = _aspera().browse(self.client, path=_norm(path))
        return entries

    def _size(self, path: str) -> int:
        entries, _ = _aspera().browse(self.client, path=_norm(path), recursive=True,
                                      type_filter="file")
        return sum(int(e.get("size") or 0) for e in entries)

    # -- Source -----------------------------------------------------------------

    def _all_folders(self) -> List[Dict[str, Any]]:
        if self._folders is None:
            found, level = [], [("/", 0)]
            while level:
                nxt = []
                for path, depth in level:
                    for e in self._list(path):
                        if e.get("type") in FOLDER_TYPES:
                            p = _norm(os.path.join(path, e["name"]))
                            found.append({"path": p, "name": e["name"]})
                            if depth + 1 < self.search_depth:
                                nxt.append((p, depth + 1))
                level = nxt
            self._folders = found
        return self._folders

    def search(self, text: str) -> List[Dataset]:
        words = [w for w in re.split(r"\W+", text.lower()) if w]
        return [self._dataset(f["path"], "atr folder names")
                for f in self._all_folders() if all(w in f["path"].lower() for w in words)]

    def _dataset(self, path: str, via: str = "") -> Dataset:
        return Dataset(source=self.name, ref={"path": _norm(path)}, title=_norm(path).strip("/"),
                       laboratory="ATR", size=None, access="account", found_via=via,
                       metadata={"host": os.environ.get(ENV_URL) or DEFAULT_URL})

    def resolve(self, ref: Dict[str, Any]) -> Dataset:
        path = ref.get("path")
        if not path:
            raise ValueError("ATR reference needs 'path'")
        parent, name = os.path.split(_norm(path))
        if not any(e.get("name") == name for e in self._list(parent or "/")):
            raise LookupError(f"ATR: no folder {path}")
        return self._dataset(path)

    def browse(self, dataset: Dataset) -> List[Item]:
        items = []
        for e in self._list(dataset.ref["path"]):
            kind = "folder" if e.get("type") in FOLDER_TYPES else "file"
            # Folder sizes are not reported by the listing; measured on request.
            size = int(e.get("size") or 0) if kind == "file" else None
            items.append(Item(e["name"], kind, size, _norm(os.path.join(dataset.ref["path"], e["name"]))))
        return items

    def measure(self, dataset: Dataset, item: Item) -> Optional[int]:
        return item.size if item.kind == "file" else self._size(item.id)

    def download(self, dataset: Dataset, items: List[Item], destination: str) -> None:
        os.makedirs(destination, exist_ok=True)
        if not items:
            items = self.browse(dataset)
        result = _aspera().download(
            self.client, remote_paths=[i.id for i in items], local_dest=destination,
            resume=True, ascp_path_override=os.environ.get("ASPERA_ASCP"))
        if result.get("status") != "success":
            errors = [f"{r['source']}: {r.get('error', r['status'])}"
                      for r in result.get("results", []) if r.get("status") != "success"]
            raise IOError("ATR download failed:\n  " + "\n  ".join(errors))
