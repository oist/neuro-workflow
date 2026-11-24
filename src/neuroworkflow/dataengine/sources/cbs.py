"""RIKEN CBS data repository (https://neurodata.riken.jp), through RIKEN's MDRS client.

    pip install "git+https://git.ni.riken.jp/niu/mdrs-client-python.git"

Two sources, one repository:

* ``cbs_public``: anonymous. Searches the published datasets (those with a DOI).
* ``cbs_project``: logged in with the shared Neuro-Workflow account, so it also
  sees BM2.0 internal data (CBS Open, 5Kikan Open, private). Searches every
  laboratory's ``/Repository/`` folders, so it finds datasets without a DOI too.
  Credentials come from the environment, never from code::

      NW_MDRS_USER, NW_MDRS_PASSWORD

Both download into the same ``cbs/`` folder of the data store, so a dataset is
never downloaded twice.

A dataset is named by its **folder UUID**: every folder has one, with or without a
DOI, public or internal, and it does not change when the folder is moved or
renamed. A DOI is resolved to its folder UUID.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ..dataset import Dataset, Item
from .base import Source

API_URL = "https://neurodata.riken.jp/api"
REMOTE = "neurodata"
ENV_USER = "NW_MDRS_USER"
ENV_PASSWORD = "NW_MDRS_PASSWORD"


def _client_modules():
    try:
        from mdrsclient.api import FoldersApi
        from mdrsclient.cache import InMemoryCache
        from mdrsclient.client import MdrsClient
        from mdrsclient.config import InMemoryConfig
        from mdrsclient.models.folder import FolderAccessLevel
    except ImportError as e:
        raise ImportError(
            "The CBS sources need RIKEN's MDRS client: "
            'pip install "git+https://git.ni.riken.jp/niu/mdrs-client-python.git"'
        ) from e
    return MdrsClient, InMemoryCache, InMemoryConfig, FoldersApi, FolderAccessLevel


def _words(text: str) -> List[str]:
    return [w for w in re.split(r"\W+", text.lower()) if w]


class CBSSource(Source):
    repository = "cbs"

    def __init__(self, login: bool = False):
        self.login = login
        self.name = "cbs_project" if login else "cbs_public"
        self._client = None
        self._doi_list: Optional[List[Dict[str, Any]]] = None
        self._lab_folders: Optional[List[Dict[str, Any]]] = None

    # -- client -----------------------------------------------------------------

    def ready(self) -> bool:
        return not self.login or bool(os.environ.get(ENV_USER) and os.environ.get(ENV_PASSWORD))

    @property
    def logged_in(self) -> bool:
        return self._client is not None and self.login

    @property
    def client(self):
        if self._client is None:
            MdrsClient, InMemoryCache, InMemoryConfig, _, _ = _client_modules()
            config = InMemoryConfig(REMOTE)      # nothing written to ~/.mdrs-client
            config.url = API_URL
            client = MdrsClient.from_remote(REMOTE, cache=InMemoryCache(), config=config)
            if self.login:
                user, password = os.environ.get(ENV_USER), os.environ.get(ENV_PASSWORD)
                if not (user and password):
                    raise PermissionError(
                        f"{self.name} needs {ENV_USER} and {ENV_PASSWORD} in the environment")
                client.login(user, password)
            self._client = client
        return self._client

    def _folder(self, folder_id: str):
        _, _, _, FoldersApi, _ = _client_modules()
        try:
            return FoldersApi(self.client.connection).retrieve(folder_id)
        except Exception as e:
            # A folder this account may list but not open (e.g. "PW Open", which
            # needs its own password): the server returns only a short record,
            # which the client cannot read.
            if type(e).__name__ == "ValidationError":
                raise PermissionError(
                    f"CBS folder {folder_id}: this account cannot open it (it needs "
                    "its folder password or wider access)") from e
            raise

    def _labs(self):
        if self.client.connection.laboratories.empty():
            self.client.get_laboratories()
        return self.client.connection.laboratories

    def _lab_name(self, laboratory_id: int) -> str:
        lab = self._labs().find_by_id(laboratory_id)
        return lab.name if lab else ""

    @staticmethod
    def _access(level: int) -> str:
        *_, FolderAccessLevel = _client_modules()
        return FolderAccessLevel.id2label(level) or str(level)

    # -- listings used by search ------------------------------------------------

    def _dois(self) -> List[Dict[str, Any]]:
        """Published datasets: DOI, folder UUID, title, laboratory, creators."""
        if self._doi_list is None:
            import requests
            r = requests.get(f"{API_URL}/v3/doi/", timeout=30)
            r.raise_for_status()
            self._doi_list = r.json()
        return self._doi_list

    def _repository_folders(self) -> List[Dict[str, Any]]:
        """Every dataset folder this account can see: each laboratory's /Repository/.

        One request per laboratory (about 60), run in parallel, once per session.
        """
        if self._lab_folders is None:
            labs = list(self._labs())

            def list_lab(lab):
                try:
                    repo = self.client.find_folder(lab, "/Repository/")
                except Exception:
                    return []                    # no Repository folder, or not visible
                return [{"lab": lab.name, "id": s.id, "name": s.name, "size": s.size,
                         "access": self._access(s.access_level),
                         "description": s.description} for s in repo.sub_folders]

            with ThreadPoolExecutor(max_workers=8) as pool:
                self._lab_folders = [f for fs in pool.map(list_lab, labs) for f in fs]
        return self._lab_folders

    # -- Source -----------------------------------------------------------------

    def search(self, text: str) -> List[Dataset]:
        words = _words(text)
        hits = []
        if not self.login:
            for rec in self._dois():
                meta = rec.get("folder", {}).get("metadata", {}) or {}
                haystack = " ".join([str(meta.get("Title", "")), str(meta.get("Laboratory", "")),
                                     rec.get("doi", ""), str(meta.get("Data Creator", ""))]).lower()
                if all(w in haystack for w in words):
                    hits.append(self._found(rec["folder"]["id"], "cbs doi list"))
        else:
            for f in self._repository_folders():
                haystack = " ".join([f["lab"], f["name"], f["description"] or ""]).lower()
                if all(w in haystack for w in words):
                    try:
                        hits.append(self._found(f["id"], "cbs lab repositories"))
                    except PermissionError:
                        hits.append(self._listed(f))
        return hits

    def _listed(self, f: Dict[str, Any]) -> Dataset:
        """A folder that cannot be opened, described from the listing alone."""
        return Dataset(source=self.name, ref={"folder_id": f["id"]},
                       title=" ".join(f["name"].split()), laboratory=f["lab"],
                       size=f["size"], access=f["access"],
                       found_via="cbs lab repositories (folder password needed)",
                       metadata={"folder_name": f["name"], "description": f["description"]})

    def _found(self, folder_id: str, via: str) -> Dataset:
        ds = self.resolve({"folder_id": folder_id})
        ds.found_via = via
        return ds

    def resolve(self, ref: Dict[str, Any]) -> Dataset:
        folder_id = ref.get("folder_id")
        if not folder_id and ref.get("doi"):
            doi = str(ref["doi"]).replace("https://doi.org/", "").strip("/")
            match = [r for r in self._dois() if r.get("doi", "").lower() == doi.lower()]
            if not match:
                raise LookupError(f"CBS: no dataset with DOI {doi}")
            folder_id = match[0]["folder"]["id"]
        if not folder_id:
            raise ValueError("CBS reference needs 'folder_id' or 'doi'")
        folder = self._folder(folder_id)
        meta, doi = {}, ""
        for rec in self._dois():
            if rec["folder"]["id"] == folder.id:
                meta, doi = rec["folder"].get("metadata", {}) or {}, rec.get("doi", "")
        return Dataset(
            source=self.name,
            ref={"folder_id": folder.id},
            title=" ".join(str(meta.get("Title") or folder.name).split()),
            doi=doi,
            laboratory=self._lab_name(folder.laboratory_id),
            size=folder.size,
            access=self._access(folder.access_level),
            metadata={**meta, "folder_name": folder.name, "path": folder.path,
                      "laboratory_id": folder.laboratory_id, "locked": folder.lock,
                      "description": folder.description},
        )

    def browse(self, dataset: Dataset) -> List[Item]:
        folder = self._folder(dataset.ref["folder_id"])
        items = [Item(s.name, "folder", s.size, s.id, self._access(s.access_level))
                 for s in folder.sub_folders]
        items += [Item(f.name, "file", f.size, f.id, dataset.access)
                  for f in self.client.find_files(folder.id)]
        return items

    def download(self, dataset: Dataset, items: List[Item], destination: str) -> None:
        folder = self._folder(dataset.ref["folder_id"])
        base = f"{REMOTE}:/{self._lab_name(folder.laboratory_id)}{folder.path}"
        os.makedirs(destination, exist_ok=True)
        if not items:
            items = self.browse(dataset)
        for item in items:
            # Each item lands under its own name: a folder as destination/<name>/,
            # a file as destination/<name>. The client prints every file path; keep
            # that out of the caller's output, but return its "Failed:" lines with
            # the error.
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    self.client.download(f"{base}{item.name}", destination,
                                         is_recursive=item.kind == "folder",
                                         is_skip_if_exists=True)
            except Exception as e:
                failed = [ln for ln in out.getvalue().splitlines() if ln.startswith("Failed:")]
                if failed and all(ln.endswith("Expecting value: line 1 column 1 (char 0)")
                                  for ln in failed):
                    # The server answers a refused file with an empty reply.
                    raise PermissionError(
                        f"CBS refused the download of {item.name!r}: this account may "
                        f"list this folder ({dataset.access}) but not download it") from e
                raise IOError(f"CBS download of {item.name!r} failed: {e}"
                              + "".join(f"\n  {ln}" for ln in failed[:20])) from e
