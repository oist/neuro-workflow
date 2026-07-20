from django.urls import path
from .views import (
    FlowProjectViewSet,
    FlowNodeViewSet,
    FlowEdgeViewSet,
    SampleFlowView,
    BatchCodeGenerationView,
    BatchWorkflowRunView,
    WorkflowRunStreamView,
    FlowNodeInstanceNameUpdateView,
    FlowNodeParameterUpdateView,
    WorkflowRunSubmitView,
    WorkflowRunDetailView,
    WorkflowRunListView,
    WorkflowRunCancelView,
    WorkflowRunArtifactView,
    WorkflowResultsView,
    WorkflowReportView,
    WorkflowCodeView,
    ViewerChatToolView,
)

app_name = "workflow"

# Helper for using a ViewSet as an APIView
project_list = FlowProjectViewSet.as_view({"get": "list", "post": "create"})

project_detail = FlowProjectViewSet.as_view(
    {"get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"}
)

project_flow = FlowProjectViewSet.as_view({"get": "flow", "put": "flow"})

node_list_create = FlowNodeViewSet.as_view({"get": "list", "post": "create"})

node_detail = FlowNodeViewSet.as_view(
    {"get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"}
)

edge_list_create = FlowEdgeViewSet.as_view({"get": "list", "post": "create"})

edge_detail = FlowEdgeViewSet.as_view({"delete": "destroy"})



urlpatterns = [
    # project management
    path("", project_list, name="workflow-list-create"),  # GET(list), POST(create)
    path(
        "<uuid:workflow_id>/", project_detail, name="workflow-detail"
    ),  # GET(description), PUT/PATCH(update), DELETE
    # flow data management
    path(
        "<uuid:workflow_id>/flow/", project_flow, name="workflow-flow"
    ),  # GET(flow acquisition), PUT(flow save)
    # node management
    path(
        "<uuid:workflow_id>/nodes/",
        node_list_create,
        name="workflow-node-list-create",
    ),  # GET(node list), POST(node create)
    path(
        "<uuid:workflow_id>/nodes/<str:node_id>/",
        node_detail,
        name="workflow-node-detail",
    ),  # GET(description), PUT/PATCH(update), DELETE
    # edge management
    path(
        "<uuid:workflow_id>/edges/",
        edge_list_create,
        name="workflow-edge-list-create",
    ),  # GET(edge list), POST(edge create)
    path(
        "<uuid:workflow_id>/edges/<str:edge_id>/",
        edge_detail,
        name="workflow-edge-detail",
    ),  # DELETE
    # Node instance name update
    path(
        "<uuid:workflow_id>/nodes/<str:node_id>/instance_name/",
        FlowNodeInstanceNameUpdateView.as_view(),
        name="node-instance_name-update"
    ),  # PUT(node schema.instance_name update)
    # Update node parameters
    path(
        "<uuid:workflow_id>/nodes/<str:node_id>/parameters/",
        FlowNodeParameterUpdateView.as_view(),
        name="node-parameter-update"
    ),  # PUT(node schema.parameters update)
    # Batch Code Generation - New Addition
    path(
        "<uuid:workflow_id>/generate-code/",
        BatchCodeGenerationView.as_view(),
        name="batch-code-generation"
    ),  # POST (generate code in batch from React Flow JSON)
    # Run Workflow (SSE streaming via Jupyter kernel)
    path(
        "<uuid:workflow_id>/run/",
        WorkflowRunStreamView.as_view(),
        name="workflow-run-stream"
    ),  # POST (Run workflow on Jupyter, SSE streaming output)
    # Async run management
    path(
        "<uuid:workflow_id>/runs/",
        WorkflowRunListView.as_view(),
        name="workflow-run-list"
    ),  # GET (list runs)
    path(
        "<uuid:workflow_id>/runs/submit/",
        WorkflowRunSubmitView.as_view(),
        name="workflow-run-submit"
    ),  # POST (submit a new run)
    path(
        "<uuid:workflow_id>/runs/<uuid:run_id>/",
        WorkflowRunDetailView.as_view(),
        name="workflow-run-detail"
    ),  # GET (run status + logs)
    path(
        "<uuid:workflow_id>/runs/<uuid:run_id>/cancel/",
        WorkflowRunCancelView.as_view(),
        name="workflow-run-cancel"
    ),  # POST (cancel a run)
    path(
        "<uuid:workflow_id>/runs/<uuid:run_id>/artifacts/",
        WorkflowRunArtifactView.as_view(),
        name="workflow-run-artifact"
    ),  # GET(?path=... download one fetched result file)
    # Results listing
    path(
        "<uuid:workflow_id>/results/",
        WorkflowResultsView.as_view(),
        name="workflow-results"
    ),  # GET(list result files with metadata)
    # Generated code and notebook outputs
    path(
        "<uuid:workflow_id>/code/",
        WorkflowCodeView.as_view(),
        name="workflow-code"
    ),  # GET(generated .py code + notebook cell outputs)
    # Report save/retrieve
    path(
        "<uuid:workflow_id>/report/",
        WorkflowReportView.as_view(),
        name="workflow-report"
    ),  # GET(read report), POST(save report)
    # Brain-viewer chat tool dispatch (LLM Group 1-5 tools over the run's data)
    path(
        "<uuid:workflow_id>/viewer-chat/",
        ViewerChatToolView.as_view(),
        name="workflow-viewer-chat"
    ),  # POST({tool, args, data_path}) -> tool result / action dict
    # Sample Data
    path(
        "sample-flow/", SampleFlowView.as_view(), name="sample-flow"
    ),  # GET(Sample flow acquisition)
]

# API List:
"""
# project management
GET    /workflow/                              # list
POST   /workflow/                              # create
GET    /workflow/{workflow_id}/                # description
PUT    /workflow/{workflow_id}/                # update
DELETE /workflow/{workflow_id}/                # delete

# Flow data management (save and retrieve React Flow data as is)
GET    /workflow/{workflow_id}/flow/           # get data
PUT    /workflow/{workflow_id}/flow/           # save data


# node management
GET    /workflow/{workflow_id}/nodes/          # list
POST   /workflow/{workflow_id}/nodes/          # create
GET    /workflow/{workflow_id}/nodes/{node_id}/ # description
PUT    /workflow/{workflow_id}/nodes/{node_id}/ # update
DELETE /workflow/{workflow_id}/nodes/{node_id}/ # delete

# edge management
GET    /workflow/{workflow_id}/edges/          # list
POST   /workflow/{workflow_id}/edges/          # create
DELETE /workflow/{workflow_id}/edges/{edge_id}/ # delete

# Node instance name update
PUT    /workflow/{workflow_id}/nodes/{node_id}/instance_name/  # Update the node's schema.parameters

# Update node parameters
PUT    /workflow/{workflow_id}/nodes/{node_id}/parameters/  # Update the node's schema.parameters

# Batch code generation
POST   /workflow/{workflow_id}/generate-code/  # React Flow batch code generation from JSON

# Run Workflow
POST   /workflow/{workflow_id}/run/            # Run Workflow Program

# sample data
GET    /workflow/sample-flow/                  # Sample flow data acquisition

Request example:

# Update node parameters
PUT /workflow/{workflow_id}/nodes/{node_id}/parameters/
{
  "parameter_key": "record_from_population",
  "parameter_value": 100,
  "parameter_field": "value"  # 'value', 'default_value', 'constraints', 'description', 'type'
}
Response: {
  "status": "success",
  "message": "Parameter 'record_from_population.value' updated successfully",
  "node_id": "node_id",
  "parameter_key": "record_from_population",
  "parameter_field": "value",
  "parameter_value": 100
}

# Batch code generation
POST /workflow/{workflow_id}/generate-code/
{
  "nodes": [
    {
      "id": "node1",
      "position": {"x": 100, "y": 100},
      "type": "calculationNode",
      "data": {"label": "BuildSonataNetworkNode"}
    }
  ],
  "edges": [
    {
      "id": "edge1",
      "source": "node1",
      "target": "node2"
    }
  ]
}
Response: {
  "status": "success", 
  "message": "Code generated from 1 nodes and 1 edges",
  "code_status": "Code generation completed successfully"
}
"""
