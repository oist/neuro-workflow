"""Database nodes: query external neuroscience dataset catalogs.

Two families live here, and they answer different questions:

* ``DANDIQueryNode``, ``CBSQueryNode``, ``BrainMINDSQueryNode``,
  ``BMBHumanQueryNode`` — one node per upstream API, queried live through
  ``neuroworkflow.utils.remote_catalogs``. Always current; slow, one source at a
  time, and subject to upstream outages.
* ``MDBCatalogSearchNode``, ``MDBCatalogLookupNode``, ``MDBLocalCatalogNode`` —
  queries against a bm_mindsdb (mdb) service through
  ``neuroworkflow.utils.mdb_client``. Fast, searchable across every source at
  once, reproducible between runs, and the only route to the local BIDS
  catalog. Reflects the catalog as of mdb's last sync, and needs mdb running
  (``MDB_BASE_URL``).

Import nodes directly by module path, e.g.::

    from neuroworkflow.nodes.database.DANDIQueryNode import DANDIQueryNode
"""
