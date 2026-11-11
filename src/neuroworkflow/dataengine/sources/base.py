"""The four calls every source answers, whatever client sits behind it.

A source wraps a partner's own client (MDRS for CBS, aspera-client for ATR). The
transfer is never reimplemented here: a source only translates between the
engine's Dataset/Item and the client's calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..dataset import Dataset, Item


class Source:
    name = ""
    repository = ""        # store folder; sources reaching the same repository share it

    def ready(self) -> bool:
        """Usable here: credentials present where the source needs them."""
        return True

    def search(self, text: str) -> List[Dataset]:
        """Datasets whose title/metadata match the text."""
        raise NotImplementedError

    def resolve(self, ref: Dict[str, Any]) -> Dataset:
        """The dataset a reference names, with its size and access level."""
        raise NotImplementedError

    def browse(self, dataset: Dataset) -> List[Item]:
        """The sub-folders and files directly inside the dataset."""
        raise NotImplementedError

    def measure(self, dataset: Dataset, item: Item) -> Optional[int]:
        """Size in bytes of an item whose listing gave none (e.g. an Aspera folder)."""
        return item.size

    def download(self, dataset: Dataset, items: List[Item], destination: str) -> None:
        """Download the items (all of the dataset when empty) into destination.

        Must skip files already present with the right size, and raise if any file
        failed, so the engine can mark the download complete only when it is.
        """
        raise NotImplementedError
