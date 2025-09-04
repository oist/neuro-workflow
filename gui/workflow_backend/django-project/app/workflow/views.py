from rest_framework import status, viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction
from django.http import JsonResponse
from .models import FlowProject, FlowNode, FlowEdge
from .serializers import (
    FlowProjectSerializer,
    FlowNodeSerializer,
    FlowEdgeSerializer,
    FlowDataSerializer,
)
from .services import FlowService
import json
import logging
from django.contrib.auth.models import User
import os
from pathlib import Path
from django.conf import settings
from .code_generation_service import CodeGenerationService

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name="dispatch")
class FlowProjectViewSet(viewsets.ModelViewSet):
    permission_classes = [AllowAny]
    authentication_classes = []
    """フロープロジェクトのCRUD操作"""

    serializer_class = FlowProjectSerializer
    lookup_url_kwarg = "workflow_id"

    def get_queryset(self):
        return FlowProject.objects.filter(is_active=True)

    def perform_create(self, serializer):
        if self.request.user.is_authenticated:
            owner = self.request.user
        else:
            # デフォルトユーザーを取得または作成
            owner, created = User.objects.get_or_create(
                username="anonymous_user",
                defaults={
                    "email": "anonymous@example.com",
                    "first_name": "Anonymous",
                    "last_name": "User",
                    "is_active": True,
                },
            )
            if created:
                print("Created default anonymous user")

        # プロジェクトを保存
        project = serializer.save(owner=owner)

        # プロジェクト作成時の自動コード生成は削除
        # 必要に応じて /generate-code/ エンドポイントを使用

        return project

    def create_project_python_file(self, project):
        """プロジェクト作成時にPythonファイルを生成"""
        try:
            code_service = CodeGenerationService()
            code_file = code_service.get_code_file_path(project.id)

            # 基本テンプレートを作成
            python_code = code_service._create_base_template(project)

            # ファイルに書き込み
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(python_code)

            logger.info(f"Created Python file for project {project.id}: {code_file}")

        except Exception as e:
            logger.error(f"Failed to create Python file for project {project.id}: {e}")
            # エラーが発生してもプロジェクト作成は継続

    @action(detail=True, methods=["get", "put"])
    def flow(self, request, **kwargs):
        """フローデータの取得・保存（一括保存用として残す）"""
        project = self.get_object()

        if request.method == "GET":
            flow_data = FlowService.get_flow_data(str(project.id))
            return Response(flow_data)

        elif request.method == "PUT":
            serializer = FlowDataSerializer(data=request.data)
            if serializer.is_valid():
                try:
                    FlowService.save_flow_data(
                        str(project.id),
                        serializer.validated_data["nodes"],
                        serializer.validated_data["edges"],
                    )

                    response_data = {
                        "status": "success",
                        "message": "Flow data saved successfully (code generation disabled - use /generate-code/ endpoint for batch code generation)",
                    }

                    return Response(response_data)
                except Exception as e:
                    logger.error(f"Error saving flow data: {e}")
                    return Response(
                        {"error": str(e)}, status=status.HTTP_400_BAD_REQUEST
                    )
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@method_decorator(csrf_exempt, name="dispatch")
class FlowNodeViewSet(viewsets.ModelViewSet):
    """フローノードのCRUD操作（リアルタイム対応）"""

    authentication_classes = []

    serializer_class = FlowNodeSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        project_id = self.kwargs.get("workflow_id")
        if project_id:
            return FlowNode.objects.filter(project_id=project_id)
        return FlowNode.objects.none()

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        """ノード作成（リアルタイム保存 + コード生成）"""
        project_id = self.kwargs.get("workflow_id")
        logger.info(f"Creating node in project {project_id} with data: {request.data}")

        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=project_id)

            # リクエストデータの検証
            required_fields = ["id", "position"]
            for field in required_fields:
                if field not in request.data:
                    logger.warning(f"Missing required field: {field}")
                    return Response(
                        {"error": f"Missing required field: {field}"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

            # FlowServiceを使用してノード作成（既存の処理と同じ）
            node_data = {
                "id": request.data["id"],
                "position": request.data["position"],
                "type": request.data.get("type", "default"),
                "data": request.data.get("data", {}),
            }

            # 既存ノードの確認（重複作成を防ぐ）
            try:
                existing_node = FlowNode.objects.get(
                    id=node_data["id"], project=project
                )
                logger.info(f"Node {node_data['id']} already exists, updating instead")

                # 既存の場合は更新
                existing_node.position_x = node_data["position"]["x"]
                existing_node.position_y = node_data["position"]["y"]
                existing_node.node_type = node_data.get("type", existing_node.node_type)
                existing_node.data = node_data.get("data", existing_node.data)
                existing_node.save()

                serializer = FlowNodeSerializer(existing_node)
                response_data = {
                    "status": "success",
                    "message": "Node updated (already existed - code generation disabled)",
                    "data": serializer.data,
                }

                return Response(response_data, status=status.HTTP_200_OK)

            except FlowNode.DoesNotExist:
                # 新規作成
                node = FlowService.create_node(str(project.id), node_data)

                serializer = FlowNodeSerializer(node)

                response_data = {
                    "status": "success",
                    "message": "Node created successfully (code generation disabled - use batch generation endpoint)",
                    "data": serializer.data,
                }

                logger.info(
                    f"Successfully created node {node.id} in project {project.id}"
                )
                return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(
                f"Error creating node in project {project_id}: {e}", exc_info=True
            )
            return Response(
                {"error": f"Failed to create node: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @transaction.atomic
    def update(self, request, *args, **kwargs):
        """ノード更新（位置変更、データ変更など + 条件付きコード生成）"""
        project_id = self.kwargs.get("workflow_id")
        node_id = self.kwargs.get("node_id")
        logger.info(
            f"Updating node {node_id} in project {project_id} with data: {request.data}"
        )

        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=project_id)

            # ノードの存在確認（IDで直接検索）
            try:
                existing_node = FlowNode.objects.get(id=node_id, project=project)
            except FlowNode.DoesNotExist:
                logger.warning(f"Node {node_id} not found in project {project_id}")
                return Response(
                    {"error": f"Node {node_id} not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # FlowServiceを使用してノード更新
            node_data = {}
            if "position" in request.data:
                node_data["position"] = request.data["position"]
            if "type" in request.data:
                node_data["type"] = request.data["type"]
            if "data" in request.data:
                node_data["data"] = request.data["data"]

            node = FlowService.update_node(node_id, project_id, node_data)

            serializer = FlowNodeSerializer(node)

            response_data = {
                "status": "success",
                "message": "Node updated successfully (code generation disabled - use batch generation endpoint)",
                "data": serializer.data,
            }

            logger.info(f"Successfully updated node {node_id} in project {project_id}")
            return Response(response_data)

        except Exception as e:
            logger.error(
                f"Error updating node {node_id} in project {project_id}: {e}",
                exc_info=True,
            )
            return Response(
                {"error": f"Failed to update node: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @transaction.atomic
    def destroy(self, request, *args, **kwargs):
        """ノード削除 + コード削除"""
        project_id = self.kwargs.get("workflow_id")
        node_id = self.kwargs.get("node_id")
        logger.info(f"Deleting node {node_id} from project {project_id}")

        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=project_id)

            # ノードの存在確認（IDで直接検索）
            try:
                existing_node = FlowNode.objects.get(id=node_id, project=project)
            except FlowNode.DoesNotExist:
                logger.warning(
                    f"Node {node_id} not found in project {project_id}, but returning success"
                )
                # ノードが存在しない場合も成功として扱う（冪等性）
                return Response(
                    {
                        "status": "success",
                        "message": "Node already deleted or not found",
                    },
                    status=status.HTTP_200_OK,
                )

            # FlowServiceを使用してノード削除（関連エッジも自動削除）
            FlowService.delete_node(node_id, project_id)

            response_data = {
                "status": "success",
                "message": "Node and related edges deleted successfully (code generation disabled - use batch generation endpoint)",
            }

            logger.info(
                f"Successfully deleted node {node_id} from project {project_id}"
            )
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(
                f"Error deleting node {node_id} from project {project_id}: {e}",
                exc_info=True,
            )
            return Response(
                {"error": f"Failed to delete node: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )


@method_decorator(csrf_exempt, name="dispatch")
class FlowEdgeViewSet(viewsets.ModelViewSet):
    """フローエッジのCRUD操作（リアルタイム対応）"""

    serializer_class = FlowEdgeSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get_queryset(self):
        project_id = self.kwargs.get("workflow_id")
        if project_id:
            return FlowEdge.objects.filter(project_id=project_id)
        return FlowEdge.objects.none()

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        """エッジ作成（リアルタイム保存 + WorkflowBuilder更新）"""
        project_id = self.kwargs.get("workflow_id")
        logger.info(f"Creating edge in project {project_id} with data: {request.data}")

        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=project_id)

            # リクエストデータの検証
            required_fields = ["id", "source", "target"]
            for field in required_fields:
                if field not in request.data:
                    logger.warning(f"Missing required field: {field}")
                    return Response(
                        {"error": f"Missing required field: {field}"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

            edge_data = {
                "id": request.data["id"],
                "source": request.data["source"],
                "target": request.data["target"],
                "sourceHandle": request.data.get("sourceHandle"),
                "targetHandle": request.data.get("targetHandle"),
                "data": request.data.get("data", {}),
            }

            # 既存エッジの確認（重複作成を防ぐ）
            try:
                existing_edge = FlowEdge.objects.get(
                    id=edge_data["id"], project=project
                )
                logger.info(f"Edge {edge_data['id']} already exists")

                serializer = FlowEdgeSerializer(existing_edge)
                response_data = {
                    "status": "success",
                    "message": "Edge already exists (code generation disabled)",
                    "data": serializer.data,
                }

                return Response(response_data, status=status.HTTP_200_OK)

            except FlowEdge.DoesNotExist:
                # 新規作成
                edge = FlowService.create_edge(str(project.id), edge_data)

                serializer = FlowEdgeSerializer(edge)

                response_data = {
                    "status": "success",
                    "message": "Edge created successfully (code generation disabled - use batch generation endpoint)",
                    "data": serializer.data,
                }

                logger.info(
                    f"Successfully created edge {edge.id} in project {project.id}"
                )
                return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(
                f"Error creating edge in project {project_id}: {e}", exc_info=True
            )
            return Response(
                {"error": f"Failed to create edge: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    @transaction.atomic
    def destroy(self, request, *args, **kwargs):
        """エッジ削除 + WorkflowBuilder更新"""
        project_id = self.kwargs.get("workflow_id")
        edge_id = self.kwargs.get("edge_id")
        logger.info(f"Deleting edge {edge_id} from project {project_id}")

        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=project_id)

            # エッジの存在確認（IDで直接検索）
            try:
                existing_edge = FlowEdge.objects.get(id=edge_id, project=project)
            except FlowEdge.DoesNotExist:
                logger.warning(
                    f"Edge {edge_id} not found in project {project_id}, but returning success"
                )
                # エッジが存在しない場合も成功として扱う（冪等性）
                return Response(
                    {
                        "status": "success",
                        "message": "Edge already deleted or not found",
                    },
                    status=status.HTTP_200_OK,
                )

            # FlowServiceを使用してエッジ削除
            FlowService.delete_edge(edge_id, project_id)

            response_data = {
                "status": "success",
                "message": "Edge deleted successfully (code generation disabled - use batch generation endpoint)",
            }

            logger.info(
                f"Successfully deleted edge {edge_id} from project {project_id}"
            )
            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(
                f"Error deleting edge {edge_id} from project {project_id}: {e}",
                exc_info=True,
            )
            return Response(
                {"error": f"Failed to delete edge: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )


@method_decorator(csrf_exempt, name="dispatch")
class SampleFlowView(APIView):
    """サンプルフローデータの提供"""

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request):
        """サンプルフローデータを返す"""
        try:
            sample_flow = FlowService.get_sample_flow_data()
            return Response(sample_flow, content_type="application/json")
        except Exception as e:
            logger.error(f"Error getting sample flow data: {e}")
            return Response(
                {"error": "Failed to get sample flow data"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )





@method_decorator(csrf_exempt, name="dispatch")
class JupyterLabView(APIView):
    """JupyterLabとの統合用ビュー"""
    
    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request, workflow_id):
        """JupyterLabのURLを返す"""
        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=workflow_id)
            
            # JupyterLabのURL生成
            jupyter_url = f"http://localhost:8000/user/user1/lab/tree/projects/{workflow_id}"
            
            return JsonResponse({
                "status": "success",
                "jupyter_url": jupyter_url,
                "workflow_id": str(workflow_id),
                "project_name": project.name
            })
            
        except Exception as e:
            logger.error(f"Error generating JupyterLab URL for workflow {workflow_id}: {e}")
            return JsonResponse(
                {"error": f"Failed to generate JupyterLab URL: {str(e)}"},
                status=500
            )


@method_decorator(csrf_exempt, name="dispatch")
class BatchCodeGenerationView(APIView):
    """React FlowのJSONからバッチでコード生成するビュー"""
    
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request, workflow_id):
        """React Flow JSONからコードを一括生成"""
        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=workflow_id)
            
            # リクエストボディからReact FlowのJSONデータを取得
            data = json.loads(request.body)
            nodes_data = data.get("nodes", [])
            edges_data = data.get("edges", [])
            
            logger.info(f"Batch code generation for project {workflow_id}: {len(nodes_data)} nodes, {len(edges_data)} edges")
            
            # フローデータを保存（既存の処理を再利用）
            with transaction.atomic():
                FlowService.save_flow_data(str(workflow_id), nodes_data, edges_data)
                
                # コード生成サービスを使用して一括でコード生成
                code_service = CodeGenerationService()
                success = code_service.generate_code_from_flow_data(str(workflow_id), nodes_data, edges_data)
                
                response_data = {
                    "status": "success",
                    "message": f"Code generated from {len(nodes_data)} nodes and {len(edges_data)} edges",
                    "workflow_id": str(workflow_id),
                    "nodes_processed": len(nodes_data),
                    "edges_processed": len(edges_data)
                }
                
                if success:
                    response_data["code_status"] = "Code generation completed successfully"
                    
                    # 生成されたコードファイルのパスを返す
                    code_file = code_service.get_code_file_path(workflow_id)
                    notebook_file = code_service.get_notebook_file_path(workflow_id)
                    
                    response_data["files"] = {
                        "python_file": str(code_file),
                        "notebook_file": str(notebook_file),
                        "python_exists": code_file.exists(),
                        "notebook_exists": notebook_file.exists()
                    }
                else:
                    response_data["code_status"] = "Code generation failed"
                    response_data["error"] = "Code generation process encountered errors"
                
                return Response(response_data, status=status.HTTP_200_OK)
                
        except json.JSONDecodeError:
            return Response(
                {"error": "Invalid JSON format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except FlowProject.DoesNotExist:
            return Response(
                {"error": f"Project {workflow_id} not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Error in batch code generation for project {workflow_id}: {e}")
            return Response(
                {"error": f"Batch code generation failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
