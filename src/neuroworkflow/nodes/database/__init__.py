"""
Database nodes for the NeuroWorkflow system.

This package provides nodes that fetch records and data files from external
research-data repositories. Repository addresses and credentials are configured
on the backend (see docs/OAI_PMH_HARVEST.md), never as node parameters.

Import nodes directly from their modules:
    from neuroworkflow.nodes.database.OAIPMHHarvestNode import OAIPMHHarvestNode
    from neuroworkflow.nodes.database.OAIPMHDownloadNode import OAIPMHDownloadNode
"""
