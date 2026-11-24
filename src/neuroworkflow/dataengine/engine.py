"""From a text query to data on disk, and a raw data node that points at it.

    eng  = DataEngine(store="./data")
    hits = eng.search("marmoset connectivity")     # every ready source, merged
    eng.show(hits)
    ds   = hits[0]
    eng.show_items(eng.browse(ds))                  # sub-folders and files, with sizes
    req  = eng.request(ds, include=["auxiliary_data"])
    print(req.summary())                            # nothing downloaded yet
    res  = eng.fetch(req)
    res.write_node("my_nodes/")                     # the raw data node

Later, from a node or a script, without searching again:

    path = ensure("cbs_public", {"folder_id": "..."}, include=["auxiliary_data"])

The engine finds, downloads, records, and writes the raw node. It does not
preprocess: that belongs to nodes downstream of the raw node.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import store as st
from .dataset import DataRequest, Dataset, Item, human_size
from .sources.atr import AtrSource
from .sources.base import Source
from .sources.cbs import CBSSource


def default_sources() -> Dict[str, Source]:
    return {
        "cbs_public": CBSSource(login=False),
        "cbs_project": CBSSource(login=True),
        "atr": AtrSource(),
    }


@dataclass
class FetchResult:
    path: str                     # local folder of the dataset
    record: Dict[str, Any]
    complete: bool                # every requested item downloaded in full
    dataset: Optional[Dataset] = None
    include: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        state = "complete" if self.complete else "INCOMPLETE"
        return f"{state}: {self.path}"

    def write_node(self, out_dir: str, class_name: Optional[str] = None) -> str:
        """Write the raw data node for this download; returns the .py path."""
        from .nodewriter import write_node
        return write_node(self.dataset, self.include, out_dir, class_name=class_name,
                          size=self.record.get("size"))


class DataEngine:
    def __init__(self, store: Optional[str] = None, sources: Optional[Dict[str, Source]] = None):
        self.store = st.store_root(store)
        self.sources = sources if sources is not None else default_sources()

    def _source(self, name: str) -> Source:
        if name not in self.sources:
            raise KeyError(f"unknown source {name!r}; available: {sorted(self.sources)}")
        return self.sources[name]

    def ready(self) -> Dict[str, bool]:
        """Which sources can be used here (credentials present where needed)."""
        return {name: s.ready() for name, s in self.sources.items()}

    # -- find -------------------------------------------------------------------

    def search(self, text: str, sources: Optional[List[str]] = None) -> List[Dataset]:
        """Search the named sources, or every ready one. A source that fails is
        reported and skipped, so one unreachable partner does not hide the others."""
        names = sources or [n for n, s in self.sources.items() if s.ready()]
        hits: List[Dataset] = []
        for name in names:
            try:
                hits += self._source(name).search(text)
            except Exception as e:
                print(f"[{name}] search failed: {type(e).__name__}: {e}")
        return hits

    def resolve(self, source: str, ref: Dict[str, Any]) -> Dataset:
        """The dataset a source reference names (a CBS folder UUID or DOI, an ATR path)."""
        return self._source(source).resolve(ref)

    def browse(self, dataset: Dataset) -> List[Item]:
        return self._source(dataset.source).browse(dataset)

    @staticmethod
    def show(hits: List[Dataset]) -> None:
        if not hits:
            print("no datasets found")
        for i, ds in enumerate(hits):
            print(f"{i:>3}  {ds.line()}")

    @staticmethod
    def show_items(items: List[Item]) -> None:
        for i in items:
            print(f"  {i.name:<60} {i.kind:<6} {human_size(i.size):>9}  {i.access}")

    # -- choose -----------------------------------------------------------------

    def path_of(self, dataset: Dataset) -> str:
        return st.dataset_dir(self._source(dataset.source).repository, dataset.key, self.store)

    def request(self, dataset: Dataset, include: Optional[List[str]] = None) -> DataRequest:
        """What would be downloaded. Checks the names exist and measures sizes the
        listing did not give; downloads nothing."""
        include = list(include or [])
        source = self._source(dataset.source)
        items = self.browse(dataset)
        by_name = {i.name: i for i in items}
        unknown = [n for n in include if n not in by_name]
        if unknown:
            raise KeyError(f"not in {dataset.title!r}: {unknown}; available: {sorted(by_name)}")
        chosen = [by_name[n] for n in include] if include else items
        for item in chosen:
            if item.size is None:
                item.size = source.measure(dataset, item)
        return DataRequest(dataset, include, chosen, self.path_of(dataset))

    # -- download ---------------------------------------------------------------

    def fetch(self, req: DataRequest) -> FetchResult:
        """Download the request, recording each item as it completes.

        Items already recorded as complete are not downloaded again. An item whose
        download fails is recorded as incomplete and the error is raised, so a
        half-downloaded item is never taken for a finished one.
        """
        ds, path = req.dataset, req.destination
        record = st.read_record(path) or {
            "source": ds.source, "ref": ds.ref, "title": ds.title, "doi": ds.doi,
            "laboratory": ds.laboratory, "access": ds.access, "size": ds.size,
            "metadata": ds.metadata, "items": {}, "whole_complete": False,
        }
        record["access"] = ds.access          # access can change at the source
        todo = [i for i in req.items if not record["items"].get(i.name, {}).get("complete")]
        for item in todo:
            entry = {"kind": item.kind, "remote_size": item.size, "complete": False,
                     "started_at": st.now()}
            record["items"][item.name] = entry
            st.write_record(path, record)
            self._source(ds.source).download(ds, [item], path)   # raises on any failure
            local = st.local_size(os.path.join(path, item.name))
            entry.update(complete=item.size is None or local == item.size,
                         local_size=local, fetched_at=st.now())
            st.write_record(path, record)
            if not entry["complete"]:
                raise IOError(f"{item.name}: {local} bytes on disk, {item.size} expected")
        if not req.include:
            record["whole_complete"] = all(record["items"].get(i.name, {}).get("complete")
                                           for i in req.items)
            st.write_record(path, record)
        missing = st.missing(record, req.include)
        return FetchResult(path, record, complete=not missing, dataset=ds,
                           include=list(req.include))


def ensure(source: str, ref: Dict[str, Any], include: Optional[List[str]] = None,
           store: Optional[str] = None, download: bool = False,
           engine: Optional[DataEngine] = None) -> str:
    """Local folder of a dataset, checked complete against the record and the disk.

    With ``download=False`` (the default, for use inside nodes) nothing is fetched:
    a dataset that is not complete on this machine raises with the reason, so a
    large transfer never starts in the middle of a workflow run. With
    ``download=True`` the missing items are fetched first.
    """
    include = list(include or [])
    eng = engine or DataEngine(store=store)
    repository = eng._source(source).repository
    key = str(ref.get("folder_id") or ref.get("path") or "").strip("/").replace("/", "_")
    missing = ["<unresolved reference>"]
    if key:
        path = st.dataset_dir(repository, key, eng.store)
        record = st.read_record(path)
        missing = st.missing(record, include)
        # The record says what finished; the disk says what is still there.
        names = include or list(record.get("items", {}))
        missing += [n for n in names
                    if n not in missing and not os.path.exists(os.path.join(path, n))]
        if not missing:
            return path
    if not download:
        raise FileNotFoundError(
            f"{source} {ref}: not complete in the data store {eng.store} "
            f"(missing: {', '.join(missing)}). Fetch it first with the data engine, "
            f"or call ensure(..., download=True)."
        )
    ds = eng.resolve(source, ref)
    result = eng.fetch(eng.request(ds, include))
    if not result.complete:
        raise IOError(f"{source} {ref}: download incomplete: {result.path}")
    return result.path
