"""Database nodes: query external neuroscience dataset catalogs.

Each node wraps one remote catalog client from
``neuroworkflow.utils.remote_catalogs`` and outputs the fetched dataset records
so downstream nodes can consume them. Import nodes directly by module path, e.g.::

    from neuroworkflow.nodes.database.DANDIQueryNode import DANDIQueryNode
"""
