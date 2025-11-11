from django.urls import path
from .views import (
    FlowProjectViewSet,
    FlowNodeViewSet,
    FlowEdgeViewSet,
    SampleFlowView,
    BatchCodeGenerationView,
    FlowNodeInstanceNameUpdateView,
    FlowNodeParameterUpdateView,
)

app_name = "workflow"

# ViewSetをAPIViewとして使用するためのヘルパー
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
    # プロジェクト管理
    path("", project_list, name="workflow-list-create"),  # GET(一覧), POST(作成)
    path(
        "<uuid:workflow_id>/", project_detail, name="workflow-detail"
    ),  # GET(詳細), PUT/PATCH(更新), DELETE(削除)
    # フローデータ管理
    path(
        "<uuid:workflow_id>/flow/", project_flow, name="workflow-flow"
    ),  # GET(フロー取得), PUT(フロー保存)
    # ノード管理
    path(
        "<uuid:workflow_id>/nodes/",
        node_list_create,
        name="workflow-node-list-create",
    ),  # GET(ノード一覧), POST(ノード作成)
    path(
        "<uuid:workflow_id>/nodes/<str:node_id>/",
        node_detail,
        name="workflow-node-detail",
    ),  # GET(詳細), PUT/PATCH(更新), DELETE(削除)
    # エッジ管理
    path(
        "<uuid:workflow_id>/edges/",
        edge_list_create,
        name="workflow-edge-list-create",
    ),  # GET(エッジ一覧), POST(エッジ作成)
    path(
        "<uuid:workflow_id>/edges/<str:edge_id>/",
        edge_detail,
        name="workflow-edge-detail",
    ),  # DELETE(削除)
    # ノードインスタンス名更新
    path(
        "<uuid:workflow_id>/nodes/<str:node_id>/instance_name/",
        FlowNodeInstanceNameUpdateView.as_view(),
        name="node-instance_name-update"
    ),  # PUT(ノードのschema.instance_nameを更新)
    # ノードパラメーター更新
    path(
        "<uuid:workflow_id>/nodes/<str:node_id>/parameters/",
        FlowNodeParameterUpdateView.as_view(),
        name="node-parameter-update"
    ),  # PUT(ノードのschema.parametersを更新)
    # バッチコード生成 - 新規追加
    path(
        "<uuid:workflow_id>/generate-code/",
        BatchCodeGenerationView.as_view(),
        name="batch-code-generation"
    ),  # POST(React Flow JSONからバッチでコード生成)
    # サンプルデータ
    path(
        "sample-flow/", SampleFlowView.as_view(), name="sample-flow"
    ),  # GET(サンプルフロー取得)
]

# 利用可能なAPI一覧:
"""
# プロジェクト管理
GET    /workflow/                              # プロジェクト一覧
POST   /workflow/                              # プロジェクト作成
GET    /workflow/{workflow_id}/                # プロジェクト詳細
PUT    /workflow/{workflow_id}/                # プロジェクト更新
DELETE /workflow/{workflow_id}/                # プロジェクト削除

# フローデータ管理（React Flowのデータをそのまま保存・取得）
GET    /workflow/{workflow_id}/flow/           # フローデータ取得
PUT    /workflow/{workflow_id}/flow/           # フローデータ保存


# ノード管理
GET    /workflow/{workflow_id}/nodes/          # ノード一覧
POST   /workflow/{workflow_id}/nodes/          # ノード作成
GET    /workflow/{workflow_id}/nodes/{node_id}/ # ノード詳細
PUT    /workflow/{workflow_id}/nodes/{node_id}/ # ノード更新
DELETE /workflow/{workflow_id}/nodes/{node_id}/ # ノード削除

# エッジ管理
GET    /workflow/{workflow_id}/edges/          # エッジ一覧
POST   /workflow/{workflow_id}/edges/          # エッジ作成
DELETE /workflow/{workflow_id}/edges/{edge_id}/ # エッジ削除

# ノードインスタンス名更新
PUT    /workflow/{workflow_id}/nodes/{node_id}/instance_name/  # ノードのschema.parametersを更新

# ノードパラメーター更新
PUT    /workflow/{workflow_id}/nodes/{node_id}/parameters/  # ノードのschema.parametersを更新

# バッチコード生成
POST   /workflow/{workflow_id}/generate-code/  # React Flow JSONからバッチでコード生成

# サンプルデータ
GET    /workflow/sample-flow/                  # サンプルフローデータ取得

リクエスト例:

# ノードパラメーター更新
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

# バッチコード生成
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
