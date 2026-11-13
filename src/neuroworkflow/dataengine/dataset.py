"""What the engine passes around: a dataset found by a search, the items inside it,
and a request to download some of them.

A dataset is named by its source's own reference (``ref``), because there is no id
shared across sources: CBS names a dataset by a folder UUID, an Aspera share by a
path. The DOI, title and laboratory travel along for display only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def human_size(n: Optional[int]) -> str:
    if n is None:
        return "?"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{n} B"


@dataclass
class Item:
    """One sub-folder or file directly inside a dataset."""

    name: str
    kind: str                      # "folder" | "file"
    size: Optional[int] = None     # bytes; a folder's size includes everything below it
    id: str = ""                   # the source's own id for it, if it has one
    access: str = ""


@dataclass
class Dataset:
    """A dataset as one source knows it."""

    source: str                    # e.g. "cbs"
    ref: Dict[str, Any]            # the source's own reference, e.g. {"folder_id": ...}
    title: str = ""
    doi: str = ""
    laboratory: str = ""
    size: Optional[int] = None     # bytes, whole dataset
    access: str = ""               # the source's access level, e.g. "Public", "CBS Open"
    found_via: str = ""            # where the search found it
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable local name: the source's own id, never the title or path (both can change)."""
        return str(self.ref.get("folder_id") or self.ref.get("path") or "").strip("/").replace("/", "_")

    def line(self) -> str:
        doi = f"  doi:{self.doi}" if self.doi else ""
        return (f"[{self.source}] {self.title[:80]}  ({self.laboratory}, "
                f"{human_size(self.size)}, {self.access or '?'}){doi}")


@dataclass
class DataRequest:
    """What to download: a dataset, optionally only some of its items, and where to."""

    dataset: Dataset
    include: List[str]             # item names directly under the dataset; [] = everything
    items: List[Item]              # the matching items, with sizes
    destination: str               # local folder for this dataset

    @property
    def size(self) -> Optional[int]:
        if not self.include:
            return self.dataset.size
        sizes = [i.size for i in self.items]
        return None if any(s is None for s in sizes) else sum(sizes)

    def summary(self) -> str:
        ds = self.dataset
        what = ", ".join(self.include) if self.include else "everything"
        lines = [
            f"dataset     : {ds.title}",
            f"source      : {ds.source}  {ds.ref}",
            f"doi         : {ds.doi or '-'}",
            f"access      : {ds.access or '?'}",
            f"download    : {what}",
        ]
        for i in self.items:
            lines.append(f"  - {i.name:<50} {i.kind:<6} {human_size(i.size)}")
        lines += [
            f"total       : {human_size(self.size)}",
            f"destination : {self.destination}",
        ]
        return "\n".join(lines)
