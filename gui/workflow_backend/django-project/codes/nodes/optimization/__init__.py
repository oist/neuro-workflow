"""
Optimization nodes for NeuroWorkflow.

Optimization itself is not a step inside a workflow: a workflow is a DAG that runs
once, while a search is a loop that runs it repeatedly. The loop therefore lives
outside the graph, in ``neuroworkflow.optimization``.

What belongs here is the *specification* of a search — which algorithm, how large a
budget — carried as an ordinary node so it can be placed and edited like any other:

    from neuroworkflow.nodes.optimization.NW_Optimization import NW_Optimization
"""
