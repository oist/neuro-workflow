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
            project_name = project.name.replace(" ","").capitalize()
            code_file = code_service.get_code_file_path(project_name)

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
            #jupyter_url = f"http://localhost:8000/user/user1/lab/tree/codes/projects/{workflow_id}"
            jupyter_url = f"http://localhost:8000/user/user1/lab/tree/codes/projects/"
            #jupyter_url = f"http://localhost:8000/user/user1/lab/workspaces/auto-E/tree/codes/nodes/{workflow_id}/{workflow_id}.py"
            
            
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
class FlowNodeParameterUpdateView(APIView):
    """FlowNodeのschema.parametersを更新する（ベースノードは変更しない）"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def put(self, request, workflow_id, node_id):
        """FlowNodeのschema.parameters内の特定パラメーターを更新"""
        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=workflow_id)

            # ノードの存在確認
            node = get_object_or_404(FlowNode, id=node_id, project=project)

            # デバッグ: リクエストデータを出力
            print(f"🔍 DEBUG: Request data: {request.data}", flush=True)
            print(f"🔍 DEBUG: Current node data: {node.data}", flush=True)

            # リクエストデータの検証
            parameter_key = request.data.get("parameter_key")
            parameter_value = request.data.get("parameter_value")
            parameter_field = request.data.get("parameter_field", "value")  # 'value', 'default_value', 'constraints'

            print(f"🔍 DEBUG: Parsed - parameter_key: {parameter_key}, parameter_value: {parameter_value}, parameter_field: {parameter_field}", flush=True)

            if not parameter_key:
                return Response(
                    {"error": "parameter_key is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if parameter_value is None:
                return Response(
                    {"error": "parameter_value is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            logger.info(f"Updating parameter '{parameter_key}.{parameter_field}' to {parameter_value} in node {node_id}")

            # schema.parametersが存在するかチェック
            if "schema" not in node.data:
                print("❌ DEBUG: No schema found in node data", flush=True)
                return Response(
                    {"error": "Node schema not found"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if "parameters" not in node.data["schema"]:
                print("❌ DEBUG: No parameters found in schema", flush=True)
                return Response(
                    {"error": "Node parameters not found in schema"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if parameter_key not in node.data["schema"]["parameters"]:
                available_keys = list(node.data["schema"]["parameters"].keys())
                print(f"❌ DEBUG: Parameter '{parameter_key}' not found. Available: {available_keys}", flush=True)
                return Response(
                    {"error": f"Parameter '{parameter_key}' not found. Available: {available_keys}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # 更新前の値を取得
            old_value = node.data["schema"]["parameters"][parameter_key].get(parameter_field)
            print(f"🔍 DEBUG: Updating {parameter_key}.{parameter_field} from {old_value} to {parameter_value}", flush=True)

            # 元の値を保存（変更履歴用）
            original_value = node.data["schema"]["parameters"][parameter_key].get(parameter_field)

            # parameter_fieldで指定されたフィールドを直接更新
            print(f"🔍 DEBUG: Before update - schema.parameters[{parameter_key}]: {node.data['schema']['parameters'][parameter_key]}", flush=True)
            node.data["schema"]["parameters"][parameter_key][parameter_field] = parameter_value
            print(f"🔍 DEBUG: After update - schema.parameters[{parameter_key}]: {node.data['schema']['parameters'][parameter_key]}", flush=True)

            print(f"🔍 DEBUG: Updated {parameter_field} from {original_value} to {parameter_value}", flush=True)

            # パラメーター変更状況を追跡（全フィールドを対象に変更）
            self._update_parameter_modification_status(
                node.data, parameter_key, parameter_field,
                node.data["schema"]["parameters"][parameter_key],
                parameter_value,
                original_value
            )

            # ノードを保存
            node.save()

            print(f"✅ DEBUG: Successfully saved parameter update", flush=True)
            print(f"🔍 DEBUG: After save - node.data keys: {list(node.data.keys())}", flush=True)
            print(f"🔍 DEBUG: After save - parameter_modifications: {node.data.get('parameter_modifications', 'NOT FOUND')}", flush=True)

            logger.info(f"Successfully updated parameter '{parameter_key}.{parameter_field}' in node {node_id}")

            return Response(
                {
                    "status": "success",
                    "message": f"Parameter '{parameter_key}.{parameter_field}' updated successfully",
                    "node_id": node_id,
                    "workflow_id": str(workflow_id),
                    "parameter_key": parameter_key,
                    "parameter_field": parameter_field,
                    "parameter_value": parameter_value,
                    "updated_parameter": node.data["schema"]["parameters"][parameter_key]
                }
            )

        except Exception as e:
            logger.error(f"Parameter update failed for node {node_id}: {e}", exc_info=True)
            return Response(
                {"error": f"Parameter update failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


    def _update_parameter_modification_status(self, node_data, parameter_key, parameter_field, parameter, new_value, original_value=None):
        """パラメーターの変更状況を追跡・更新（全フィールド対応）"""
        print(f"🔍 DEBUG: Tracking modification status for {parameter_key}.{parameter_field}", flush=True)

        # parameter_modificationsの構造を確保
        if "parameter_modifications" not in node_data:
            node_data["parameter_modifications"] = {}

        modifications = node_data["parameter_modifications"]

        # パラメーター単位での追跡データ構造を確保
        if parameter_key not in modifications:
            modifications[parameter_key] = {
                "is_modified": False,
                "field_modifications": {}
            }

        param_mod = modifications[parameter_key]

        # 既存データの互換性確保（古い形式から新しい形式への変換）
        if "field_modifications" not in param_mod:
            # 古い形式のデータを新しい形式に変換
            old_original = param_mod.get("original_value")
            old_current = param_mod.get("current_value")
            param_mod["field_modifications"] = {}

            # 古いデータが存在する場合、default_valueとして移行
            if old_original is not None:
                param_mod["field_modifications"]["default_value_original"] = old_original
                param_mod["field_modifications"]["default_value"] = {
                    "current_value": old_current,
                    "is_modified": param_mod.get("is_modified", False),
                    "modified_at": param_mod.get("modified_at")
                }

            # 古いキーを削除
            for old_key in ["original_value", "current_value", "modified_at"]:
                if old_key in param_mod:
                    del param_mod[old_key]

        # 各フィールドの元の値を取得（初回時のみ保存）
        field_key = f"{parameter_field}_original"
        if field_key not in param_mod["field_modifications"]:
            # 初回変更時に元の値を保存
            param_mod["field_modifications"][field_key] = original_value

        # 現在の値と元の値を比較して変更状況を判定
        original_field_value = param_mod["field_modifications"][field_key]
        is_field_modified = new_value != original_field_value

        print(f"🔍 DEBUG: {parameter_field} - original={original_field_value}, new={new_value}, modified={is_field_modified}", flush=True)

        # フィールドの変更状況を更新
        param_mod["field_modifications"][parameter_field] = {
            "current_value": new_value,
            "is_modified": is_field_modified,
            "modified_at": None  # フロントエンドで現在時刻を設定することを想定
        }

        # パラメーター全体の変更状況を更新（いずれかのフィールドが変更されていれば True）
        param_mod["is_modified"] = any(
            field_data.get("is_modified", False)
            for field_name, field_data in param_mod["field_modifications"].items()
            if isinstance(field_data, dict) and not field_name.endswith("_original")
        )

        # 全フィールドが元の値に戻った場合はパラメーター全体を削除
        if not param_mod["is_modified"]:
            del modifications[parameter_key]

        # 全体の変更状況を更新
        node_data["has_parameter_modifications"] = len(modifications) > 0

        print(f"✅ DEBUG: Parameter '{parameter_key}.{parameter_field}' modification status: {'modified' if is_field_modified else 'default'}", flush=True)
        print(f"🔍 DEBUG: Final modifications data: {modifications}", flush=True)


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

            # コード生成サービスを使用して一括でコード生成
            code_service = CodeGenerationService()
            # 補正したプロジェクト名
            project_name = project.name.replace(" ","").capitalize()
            success = code_service.generate_code_from_flow_data(str(workflow_id), project_name, nodes_data, edges_data)

            response_data = {
                "status": "success",
                "message": f"Code generated from {len(nodes_data)} nodes and {len(edges_data)} edges",
                "workflow_id": str(workflow_id),
                "nodes_processed": len(nodes_data),
                "edges_processed": len(edges_data)
            }

            if success:
                response_data["code_status"] = "Code generation completed successfully"
                # Idからプロジェクトを取得
                project = FlowProject.objects.get(id=workflow_id)
                project_name = project.name.replace(" ","").capitalize()

                # 生成されたコードファイルのパスを返す
                code_file = code_service.get_code_file_path(project_name)
                notebook_file = code_service.get_notebook_file_path(project_name)

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


@method_decorator(csrf_exempt, name="dispatch")
class FlowNodeInstanceNameUpdateView(APIView):
    """FlowNodeのinstanceNameを更新する（ベースノードは変更しない）"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def put(self, request, workflow_id, node_id):
        """FlowNodeのinstanceNameを更新"""
        try:
            # プロジェクトの存在確認
            project = get_object_or_404(FlowProject, id=workflow_id)

            # ノードの存在確認
            node = get_object_or_404(FlowNode, id=node_id, project=project)

            # デバッグ: リクエストデータを出力
            print(f"🔍 DEBUG: Request data: {request.data}", flush=True)
            print(f"🔍 DEBUG: Current node data: {node.data}", flush=True)

            # リクエストデータの検証
            instance_name = request.data.get("instance_name")

            print(f"🔍 DEBUG: Parsed - instance_name: {instance_name}", flush=True)

            if not instance_name:
                return Response(
                    {"error": "instance_name is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            logger.info(f"Updating instance_name '{instance_name}' in node {node_id}")

            # instance_nameが存在するかチェック
            if "instanceName" not in node.data:
                print("❌ DEBUG: No instanceName found in node data", flush=True)
                return Response(
                    {"error": "Node instanceName not found"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # 更新前の値を取得
            old_value = node.data["instanceName"]
            print(f"🔍 DEBUG: Updating instanceName from {old_value} to {instance_name}", flush=True)

            # 元の値を保存（変更履歴用）
            original_value = node.data["instanceName"]

            # parameter_fieldで指定されたフィールドを直接更新
            node.data["instanceName"] = instance_name

            print(f"🔍 DEBUG: Updated instance_name from {original_value} to {instance_name}", flush=True)

            # ノードを保存
            node.save()

            print(f"✅ DEBUG: Successfully saved instance_name update", flush=True)

            return Response(
                {
                    "status": "success",
                    "message": f"instance_name instance_name updated successfully",
                    "node_id": node_id,
                    "workflow_id": str(workflow_id),
                    "updated_instance_name": node.data["instanceName"]
                }
            )

        except Exception as e:
            logger.error(f"InstanceName update failed for node {node_id}: {e}", exc_info=True)
            return Response(
                {"error": f"InstanceName update failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )