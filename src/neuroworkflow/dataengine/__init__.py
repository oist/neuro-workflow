"""Data engine: find a dataset at a partner repository, download it into the data
store, record what arrived, and write the raw data node. Typical use::

    from neuroworkflow.dataengine import DataEngine, load_env

    load_env("../gui/.env")           # NW_* credentials only
    eng  = DataEngine(store="./data")
    hits = eng.search("marmoset connectivity")
    eng.show(hits)
    req  = eng.request(hits[0], include=["auxiliary_data"])
    print(req.summary())              # review before downloading
    res  = eng.fetch(req)
    res.write_node("my_nodes/")       # raw data node for workflows

Sources: ``cbs_public`` and ``cbs_project`` (RIKEN CBS, MDRS client), ``atr``
(ATR Aspera share, aspera-client). Each partner is reached through its own
client; the transfer is never reimplemented here.
"""

from .dataset import DataRequest, Dataset, Item, human_size
from .engine import DataEngine, FetchResult, default_sources, ensure
from .env import load_env
from .nodewriter import write_node
from .store import ENV_STORE, RECORD_NAME, read_record, store_root

__all__ = [
    "DataEngine",
    "DataRequest",
    "Dataset",
    "ENV_STORE",
    "FetchResult",
    "Item",
    "RECORD_NAME",
    "default_sources",
    "ensure",
    "human_size",
    "load_env",
    "read_record",
    "store_root",
    "write_node",
]
