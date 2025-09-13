from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny
from django.shortcuts import get_object_or_404
from django.db import models
from .models import PythonFile, NODE_CATEGORIES
from .serializers import PythonFileSerializer, PythonFileUploadSerializer
from .services.python_file_service import PythonFileService
import logging
import hashlib
import uuid
import os
from pathlib import Path
from django.core.files import File
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileUploadView(APIView):
    """Pythonファイルアップロード用のビュー"""

    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        """ファイルをアップロードして自動解析"""
        serializer = PythonFileUploadSerializer(data=request.data)

        if serializer.is_valid():
            try:
                file_service = PythonFileService()

                # ファイルを作成・解析
                python_file = file_service.create_python_file(
                    file=serializer.validated_data["file"],
                    user=request.user if request.user.is_authenticated else None,
                    name=serializer.validated_data.get("name"),
                    description=serializer.validated_data.get("description"),
                    category=serializer.validated_data.get("category", "analysis"),
                )

                # レスポンス用のシリアライザー
                response_serializer = PythonFileSerializer(
                    python_file, context={"request": request}
                )

                return Response(
                    response_serializer.data, status=status.HTTP_201_CREATED
                )

            except ValueError as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                logger.error(f"Upload failed: {e}")
                return Response(
                    {"error": "Upload failed"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileListView(APIView):
    """Pythonファイル一覧・詳細用のビュー"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request, pk=None):
        """ファイル一覧または詳細を取得"""
        if pk:
            # 詳細を取得
            python_file = get_object_or_404(PythonFile, pk=pk, is_active=True)
            serializer = PythonFileSerializer(python_file, context={"request": request})
            return Response(serializer.data)
        else:
            # 一覧を取得
            python_files = PythonFile.objects.filter(is_active=True)

            # フィルタリング
            name = request.query_params.get("name")
            if name:
                python_files = python_files.filter(name__icontains=name)

            category = request.query_params.get("category")
            if category:
                python_files = python_files.filter(category=category)

            analyzed_only = request.query_params.get("analyzed_only")
            if analyzed_only and analyzed_only.lower() == "true":
                python_files = python_files.filter(is_analyzed=True)

            serializer = PythonFileSerializer(
                python_files, many=True, context={"request": request}
            )
            return Response(serializer.data)

    def delete(self, request, pk):
        """ファイルを削除"""
        python_file = get_object_or_404(PythonFile, pk=pk, is_active=True)

        # 権限チェック
        if (
            request.user.is_authenticated
            and python_file.uploaded_by
            and python_file.uploaded_by != request.user
        ):
            return Response(
                {"error": "You don't have permission to delete this file"},
                status=status.HTTP_403_FORBIDDEN,
            )

        # ファイルシステムからファイルを削除
        if python_file.file:
            try:
                python_file.file.delete(save=False)
                logger.info(f"Deleted file from filesystem: {python_file.file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete file from filesystem: {e}")

        # 論理削除
        python_file.is_active = False
        python_file.save()

        return Response(status=status.HTTP_204_NO_CONTENT)


@method_decorator(csrf_exempt, name="dispatch")
class UploadedNodesView(APIView):
    """アップロードされたノードクラス一覧を取得するAPI"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        """アップロードされたノードクラス一覧を返す"""
        try:
            # 解析済みの有効なファイルのみ取得
            python_files = PythonFile.objects.filter(
                is_active=True, is_analyzed=True, node_classes__isnull=False
            ).exclude(node_classes={})

            all_nodes = []
            for python_file in python_files:
                frontend_nodes = python_file.get_node_classes_for_frontend()
                all_nodes.extend(frontend_nodes)

            return Response(
                {
                    "nodes": all_nodes,
                    "total_files": python_files.count(),
                    "total_nodes": len(all_nodes),
                }
            )

        except Exception as e:
            logger.error(f"Failed to get uploaded nodes: {e}")
            return Response(
                {"error": f"Failed to get uploaded nodes: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


import json
from django.views import View
from django.http import JsonResponse
from rest_framework import permissions
from django.core.files.base import ContentFile
import os
import re


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileCodeManagementView(View):
    """
    PythonFileのコード管理ビュー
    GET: コード取得
    PUT: コード保存
    """

    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    def get(self, request, filename):
        """ファイル名を指定してソースコードを取得"""
        try:
            # .pyが付いていない場合は付けても検索
            filenames_to_search = [filename]
            if not filename.endswith(".py"):
                filenames_to_search.append(f"{filename}.py")

            # ファイル名で検索
            python_file = PythonFile.objects.filter(
                name__in=filenames_to_search, is_active=True
            ).first()

            if not python_file:
                return JsonResponse(
                    {"error": f"File '{filename}' not found"}, status=404
                )

            # file_contentを優先、なければsource_codeを使用
            code = python_file.file_content or getattr(python_file, "source_code", "")

            if not code:
                return JsonResponse(
                    {"error": "Source code not available for this file"}, status=404
                )

            return JsonResponse(
                {
                    "status": "success",
                    "code": code,
                    "filename": python_file.name,
                    "file_id": str(python_file.id),
                    "description": python_file.description,
                    "uploaded_at": (
                        python_file.created_at.isoformat()
                        if hasattr(python_file, "created_at")
                        else None
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error getting code for file {filename}: {e}")
            return JsonResponse(
                {"error": "Failed to get code", "details": str(e)}, status=500
            )

    def put(self, request, filename):
        """編集したコードをDBに保存"""
        try:
            data = json.loads(request.body)
            code = data.get("code", "")

            if not code:
                return JsonResponse({"error": "Code is required"}, status=400)

            # .pyが付いていない場合は付けても検索
            filenames_to_search = [filename]
            if not filename.endswith(".py"):
                filenames_to_search.append(f"{filename}.py")

            # ファイル名で検索
            python_file = PythonFile.objects.filter(
                name__in=filenames_to_search, is_active=True
            ).first()

            if not python_file:
                return JsonResponse(
                    {"error": f"File '{filename}' not found"}, status=404
                )

            # 権限チェック（必要に応じて）
            if (
                request.user.is_authenticated
                and python_file.uploaded_by
                and python_file.uploaded_by != request.user
            ):
                return JsonResponse(
                    {"error": "You don't have permission to edit this file"}, status=403
                )

            # PythonFileServiceを使ってコード更新と再解析を実行
            from .services.python_file_service import PythonFileService

            file_service = PythonFileService()

            # ファイル内容を更新して再解析
            python_file = file_service.update_file_content(python_file, code)

            logger.info(f"Saved code for file {filename}")
            return JsonResponse(
                {
                    "status": "success",
                    "message": "Code saved successfully",
                    "filename": python_file.name,
                    "file_id": str(python_file.id),
                    "code_length": len(code),
                }
            )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            logger.error(f"Error saving code for file {filename}: {e}")
            return JsonResponse(
                {"error": "Failed to save code", "details": str(e)}, status=500
            )

    def dispatch(self, request, *args, **kwargs):
        """HTTPメソッドに応じてルーティング"""
        filename = kwargs.get("filename")

        if not filename:
            return JsonResponse({"error": "filename is required"}, status=400)

        # codeエンドポイントはGETとPUTのみ許可
        if request.method == "GET":
            return self.get(request, filename)
        elif request.method == "PUT":
            return self.put(request, filename)
        else:
            return JsonResponse({"error": "Method not allowed"}, status=405)


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileCopyView(APIView):
    """選択したPythonファイルをコピーする"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        """ファイルIDまたはファイル名を指定してコピー"""
        try:
            data = request.data
            file_ids = data.get("file_ids", [])
            source_filename = data.get("source_filename")
            target_filename = data.get("target_filename")

            # 新しい方式: ファイル名指定のコピー
            if source_filename and target_filename:
                return self._copy_by_filename(request, source_filename, target_filename)

            # 従来の方式: file_idsでのコピー
            if not file_ids:
                return Response(
                    {
                        "error": "file_ids or source_filename/target_filename is required"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if not isinstance(file_ids, list):
                return Response(
                    {"error": "file_ids must be a list"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            copied_files = []
            errors = []

            for file_id in file_ids:
                try:
                    # 元ファイルを取得
                    original_file = get_object_or_404(
                        PythonFile, pk=file_id, is_active=True
                    )

                    # コピー名を生成
                    copy_name = self._generate_copy_name(original_file.name)

                    # ユニークなfile_hashを生成
                    unique_hash = hashlib.sha256(
                        f"{copy_name}_{uuid.uuid4()}_{original_file.id}".encode()
                    ).hexdigest()

                    # 新しいファイルオブジェクトを作成
                    copied_file = PythonFile(
                        name=copy_name,
                        description=(
                            f"Copy of {original_file.description}"
                            if original_file.description
                            else f"Copy of {original_file.name}"
                        ),
                        category=original_file.category,
                        file_content=original_file.file_content,
                        uploaded_by=(
                            request.user if request.user.is_authenticated else None
                        ),
                        node_classes=(
                            original_file.node_classes.copy()
                            if original_file.node_classes
                            else {}
                        ),
                        is_analyzed=original_file.is_analyzed,
                        analysis_error=original_file.analysis_error,
                        file_size=original_file.file_size,
                        file_hash=unique_hash,
                    )

                    # ファイルフィールドをコピー
                    if original_file.file:
                        try:
                            original_file.file.open()
                            file_content = original_file.file.read()
                            original_file.file.close()

                            # コピーファイル名に.py拡張子を確保
                            file_copy_name = copy_name
                            if not file_copy_name.endswith(".py"):
                                file_copy_name = (
                                    f"{os.path.splitext(file_copy_name)[0]}.py"
                                )

                            copied_file.file.save(
                                file_copy_name, ContentFile(file_content), save=False
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not copy file content for {file_id}: {e}"
                            )

                    copied_file.save()

                    # レスポンス用のシリアライザー
                    serializer = PythonFileSerializer(
                        copied_file, context={"request": request}
                    )
                    copied_files.append(serializer.data)

                except Exception as e:
                    logger.error(f"Failed to copy file {file_id}: {e}")
                    errors.append({"file_id": file_id, "error": str(e)})

            response_data = {
                "copied_files": copied_files,
                "total_copied": len(copied_files),
                "total_requested": len(file_ids),
            }

            if errors:
                response_data["errors"] = errors

            status_code = (
                status.HTTP_201_CREATED if copied_files else status.HTTP_400_BAD_REQUEST
            )

            return Response(response_data, status=status_code)

        except Exception as e:
            logger.error(f"Copy operation failed: {e}")
            return Response(
                {"error": "Copy operation failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _generate_copy_name(self, original_name):
        """コピー用の名前を生成（.py拡張子を確保）"""
        # 既存の "copy" や "copy (n)" パターンをチェック
        base_name, ext = os.path.splitext(original_name)

        # .py拡張子を確保（拡張子がない場合や.py以外の場合）
        if not ext or ext.lower() != ".py":
            ext = ".py"

        # "copy" が既についている場合の処理
        if base_name.endswith(" - copy"):
            base_name = base_name[:-7]  # " - copy" を削除
        elif " - copy (" in base_name and base_name.endswith(")"):
            # " - copy (n)" パターンを削除
            copy_index = base_name.find(" - copy (")
            if copy_index != -1:
                base_name = base_name[:copy_index]

        # 重複しない名前を探す
        counter = 1
        while True:
            if counter == 1:
                copy_name = f"{base_name} - copy{ext}"
            else:
                copy_name = f"{base_name} - copy ({counter}){ext}"

            # 同じ名前のファイルが存在するかチェック
            if not PythonFile.objects.filter(name=copy_name, is_active=True).exists():
                return copy_name

            counter += 1

    def _copy_by_filename(self, request, source_filename, target_filename):
        """ファイル名を指定してコピー"""
        try:
            # .pyが付いていない場合は付けて検索
            source_names = [source_filename]
            if not source_filename.endswith(".py"):
                source_names.append(f"{source_filename}.py")

            # ソースファイルを取得
            original_file = PythonFile.objects.filter(
                name__in=source_names, is_active=True
            ).first()

            if not original_file:
                return Response(
                    {"error": f"Source file '{source_filename}' not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # ターゲットファイル名に.py拡張子を追加（必要な場合）
            if not target_filename.endswith(".py"):
                target_filename = f"{target_filename}.py"

            # 同名ファイルが既に存在するかチェック
            if PythonFile.objects.filter(name=target_filename, is_active=True).exists():
                return Response(
                    {"error": f"Target file '{target_filename}' already exists"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # ソースコードをコピーしてクラス名を更新
            updated_content = self._update_class_names_in_code(
                original_file.file_content or "", target_filename
            )

            # node_classesも更新（クラス名を新しいファイル名に基づいて変更）
            updated_node_classes = self._update_node_classes(
                original_file.node_classes or {}, target_filename
            )

            # ユニークなfile_hashを生成
            unique_hash = hashlib.sha256(
                f"{target_filename}_{uuid.uuid4()}_{original_file.id}".encode()
            ).hexdigest()

            # 新しいファイルオブジェクトを作成
            copied_file = PythonFile(
                name=target_filename,
                description=(
                    f"Copy of {original_file.description}"
                    if original_file.description
                    else f"Copy of {original_file.name}"
                ),
                category=original_file.category,
                file_content=updated_content,
                uploaded_by=request.user if request.user.is_authenticated else None,
                node_classes=updated_node_classes,
                is_analyzed=original_file.is_analyzed,  # 元ファイルと同じ状態を保持
                analysis_error=original_file.analysis_error,
                file_size=(
                    len(updated_content.encode("utf-8")) if updated_content else 0
                ),
                file_hash=unique_hash,
            )

            # ファイルフィールドを作成
            if updated_content:
                try:
                    copied_file.file.save(
                        target_filename,
                        ContentFile(updated_content.encode("utf-8")),
                        save=False,
                    )
                except Exception as e:
                    logger.warning(f"Could not create file for {target_filename}: {e}")

            copied_file.save()

            # レスポンス用のシリアライザー
            serializer = PythonFileSerializer(copied_file, context={"request": request})

            return Response(
                {
                    "copied_file": serializer.data,
                    "source_filename": original_file.name,
                    "target_filename": target_filename,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            logger.error(f"Copy by filename failed: {e}")
            return Response(
                {"error": "Copy operation failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _update_class_names_in_code(self, source_code, target_filename):
        """ソースコード内のクラス名をファイル名に基づいて更新"""
        if not source_code:
            return source_code

        # ターゲットファイル名から拡張子を除去してクラス名のベースを作成
        base_name = os.path.splitext(target_filename)[0]

        # ファイル名をPascalCaseのクラス名に変換
        # 例: "my_node.py" -> "MyNode", "test-node.py" -> "TestNode"
        class_name = "".join(
            word.capitalize() for word in re.split(r"[_\-]", base_name)
        )

        updated_code = source_code

        # クラス定義のパターンを探して置換
        class_pattern = r"^class\s+(\w+)(\s*\([^)]*\))?:"

        def replace_class_name(match):
            original_class_name = match.group(1)
            inheritance = match.group(2) or ""
            return f"class {class_name}{inheritance}:"

        # 複数行対応でクラス定義を置換
        updated_code = re.sub(
            class_pattern, replace_class_name, updated_code, flags=re.MULTILINE
        )

        return updated_code

    def _update_node_classes(self, original_node_classes, target_filename):
        """node_classes内のクラス名を新しいファイル名に基づいて更新"""
        if not original_node_classes:
            return original_node_classes

        # ターゲットファイル名から拡張子を除去してクラス名のベースを作成
        base_name = os.path.splitext(target_filename)[0]

        # ファイル名をPascalCaseのクラス名に変換
        new_class_name = "".join(
            word.capitalize() for word in re.split(r"[_\-]", base_name)
        )

        updated_node_classes = {}

        for class_name, class_info in original_node_classes.items():
            # クラス名を新しい名前に変更
            updated_class_info = (
                class_info.copy() if isinstance(class_info, dict) else class_info
            )

            # class_info内にクラス名が含まれている場合も更新
            if isinstance(updated_class_info, dict):
                # 'name'フィールドがある場合は更新
                if "name" in updated_class_info:
                    updated_class_info["name"] = new_class_name

                # その他のクラス名参照も更新（必要に応じて）
                for key, value in updated_class_info.items():
                    if isinstance(value, str) and value == class_name:
                        updated_class_info[key] = new_class_name

            # 新しいクラス名をキーとして追加
            updated_node_classes[new_class_name] = updated_class_info

        return updated_node_classes

    def _replace_dict_parameter_value(
        self, source_code, parameter_key, parameter_value
    ):
        """辞書内のパラメータ値を安全に置換（配列対応）"""
        logger.info(
            f"Replacing dict parameter '{parameter_key}' with value: {parameter_value} (type: {type(parameter_value)})"
        )
        formatted_value = self._format_value_for_python(parameter_value)
        logger.info(f"Formatted value: {formatted_value}")

        # より安全なパターン：キーを見つけてから値の終端を正確に検出
        patterns = [
            rf'(["\']){re.escape(parameter_key)}\1\s*:\s*',  # 'key': または "key":
        ]

        for pattern in patterns:
            match = re.search(pattern, source_code)
            if match:
                # キー部分の終了位置
                key_end = match.end()

                # 値の終端を検出（括弧のネスト、文字列も考慮）
                value_end = self._find_dict_value_end(source_code, key_end)

                if value_end > key_end:
                    # 値部分を完全に置換（元の値は削除）
                    new_source = (
                        source_code[:key_end]
                        + formatted_value
                        + source_code[value_end:]
                    )
                    return new_source

        return source_code

    def _find_dict_value_end(self, source_code, start_pos):
        """辞書の値の終端位置を検出（配列・辞書のネストに対応）"""
        bracket_count = 0
        brace_count = 0
        paren_count = 0
        in_string = False
        quote_char = None
        i = start_pos

        # 開始位置から値を読み進める
        while i < len(source_code):
            char = source_code[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                elif char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    # 配列が完全に閉じられた場合、次の文字をチェック
                    if bracket_count < 0:
                        return i
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    # 辞書が完全に閉じられた場合、またはこれが外側の辞書の終了
                    if brace_count < 0:
                        return i
                elif char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count < 0:
                        return i
                elif (
                    char == ","
                    and bracket_count == 0
                    and brace_count == 0
                    and paren_count == 0
                ):
                    # 全てのブラケット/ブレースが閉じられている場合のみ値の終端
                    return i
            else:
                if char == quote_char and (i == 0 or source_code[i - 1] != "\\"):
                    in_string = False
                    quote_char = None

            i += 1

        return len(source_code)

    def _replace_function_parameter_value(
        self, source_code, parameter_key, parameter_value
    ):
        """関数引数のパラメータ値を安全に置換（配列対応）"""
        logger.info(
            f"Replacing function parameter '{parameter_key}' with value: {parameter_value} (type: {type(parameter_value)})"
        )
        formatted_value = self._format_value_for_python(parameter_value)
        logger.info(f"Formatted value: {formatted_value}")

        # パターン: parameter_key= の後の値の終端を正確に検出
        pattern = rf"\b{re.escape(parameter_key)}\s*=\s*"
        match = re.search(pattern, source_code)

        if match:
            # パラメータキー部分の終了位置
            key_end = match.end()

            # 値の終端を検出（括弧のネスト、文字列も考慮）
            value_end = self._find_function_parameter_value_end(source_code, key_end)

            if value_end > key_end:
                # 値部分を完全に置換（元の値は削除）
                new_source = (
                    source_code[:key_end] + formatted_value + source_code[value_end:]
                )
                return new_source

        return source_code

    def _find_function_parameter_value_end(self, source_code, start_pos):
        """関数パラメータの値の終端位置を検出（配列・辞書のネストに対応）"""
        bracket_count = 0
        brace_count = 0
        paren_count = 0
        in_string = False
        quote_char = None
        i = start_pos

        while i < len(source_code):
            char = source_code[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                elif char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1
                    if bracket_count < 0:
                        return i
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count < 0:
                        return i
                elif char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count < 0:
                        # 外側の括弧の終了
                        return i
                elif (
                    char == ","
                    and bracket_count == 0
                    and brace_count == 0
                    and paren_count == 0
                ):
                    # 全てのブラケット/ブレース/括弧が閉じられている場合のみパラメータの終端
                    return i
            else:
                if char == quote_char and (i == 0 or source_code[i - 1] != "\\"):
                    in_string = False
                    quote_char = None

            i += 1

        return len(source_code)


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileParameterUpdateView(APIView):
    """ファイルのパラメーター値を更新する"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def put(self, request):
        """パラメーター値を更新"""
        try:
            data = request.data
            parameter_key = data.get("parameter_key")
            parameter_value = data.get("parameter_value")
            parameter_field = data.get(
                "parameter_field", "value"
            )  # 'value', 'default_value', 'constraints'
            file_id = data.get("file_id")
            filename = data.get("filename")

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

            # ファイルを取得（file_idまたはfilenameで検索）
            python_file = None
            if file_id:
                python_file = get_object_or_404(PythonFile, pk=file_id, is_active=True)
            elif filename:
                filenames_to_search = [filename]
                if not filename.endswith(".py"):
                    filenames_to_search.append(f"{filename}.py")

                python_file = PythonFile.objects.filter(
                    name__in=filenames_to_search, is_active=True
                ).first()

            if not python_file:
                return Response(
                    {"error": "File not found"}, status=status.HTTP_404_NOT_FOUND
                )

            # 権限チェック
            if (
                request.user.is_authenticated
                and python_file.uploaded_by
                and python_file.uploaded_by != request.user
            ):
                return Response(
                    {"error": "You don't have permission to edit this file"},
                    status=status.HTTP_403_FORBIDDEN,
                )

            # デバッグ: parameter_valueの型と値を確認
            logger.info(
                f"Received parameter_value: {parameter_value} (type: {type(parameter_value)})"
            )

            # ソースコード内のパラメーター値を更新（すべてのケースで統一）
            updated_code = self._update_parameter_in_source_code(
                python_file.file_content or "",
                parameter_key,
                parameter_field,
                parameter_value,
            )

            if updated_code == python_file.file_content:
                return Response(
                    {
                        "status": "no_change",
                        "message": f"Parameter '{parameter_key}' with field '{parameter_field}' not found or already has the same value",
                        "filename": python_file.name,
                    }
                )

            # PythonFileServiceを使ってコード更新と再解析を実行
            from .services.python_file_service import PythonFileService

            file_service = PythonFileService()

            # ファイル内容を更新して再解析
            python_file = file_service.update_file_content(python_file, updated_code)
            logger.info(
                f"Parameter '{parameter_key}.{parameter_field}' updated and re-analyzed for file {python_file.name}"
            )

            return Response(
                {
                    "status": "success",
                    "message": f"Parameter '{parameter_key}' updated successfully",
                    "filename": python_file.name,
                    "file_id": str(python_file.id),
                    "parameter_key": parameter_key,
                    "parameter_field": parameter_field,
                    "parameter_value": parameter_value,
                    "is_analyzed": python_file.is_analyzed,
                }
            )

        except Exception as e:
            logger.error(f"Parameter update failed: {e}")
            return Response(
                {"error": "Parameter update failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _update_parameter_in_source_code(
        self, source_code, parameter_key, parameter_field, parameter_value
    ):
        """ソースコード内の指定されたパラメーターのフィールドを更新"""
        if not source_code:
            return source_code

        updated_code = source_code

        if parameter_field == "value":
            # パターンを順番に試行し、一つでも変更があれば終了
            original_code = updated_code

            # パターン1: 変数代入（例: record_from_population = 100）
            variable_pattern = rf"^(\s*){re.escape(parameter_key)}\s*=\s*.*$"

            def replace_variable_assignment(match):
                indent = match.group(1)
                formatted_value = self._format_value_for_python(parameter_value)
                return f"{indent}{parameter_key} = {formatted_value}"

            updated_code = re.sub(
                variable_pattern,
                replace_variable_assignment,
                updated_code,
                flags=re.MULTILINE,
            )

            # 変数代入で変更があった場合は終了
            if updated_code != original_code:
                return updated_code

            # パターン2: 辞書内の値（例: {"record_from_population": 100} or {"time_window": [0.0, 1000.0]}）
            # より安全な方法：値の終端を正確に検出
            updated_code = self._replace_dict_parameter_value(
                updated_code, parameter_key, parameter_value
            )

            # 辞書内の値で変更があった場合は終了
            if updated_code != original_code:
                return updated_code

            # パターン3: 関数呼び出しの引数（例: func(record_from_population=100) or func(time_window=[0, 100])）
            # 配列に対応した安全な方法を使用
            updated_code = self._replace_function_parameter_value(
                updated_code, parameter_key, parameter_value
            )

        else:
            # parameter_field が 'default_value', 'constraints' などの場合
            # PortParameterのような構造を探して更新
            updated_code = self._update_parameter_metadata_in_code(
                updated_code, parameter_key, parameter_field, parameter_value
            )

        return updated_code

    def _format_value_for_python(self, value):
        """Pythonコード用に値をフォーマット"""
        logger.info(f"Formatting value: {value} (type: {type(value)})")

        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, list):
            # 配列の各要素を適切にフォーマット
            formatted_elements = []
            for item in value:
                if isinstance(item, str):
                    formatted_elements.append(f'"{item}"')
                else:
                    formatted_elements.append(str(item))
            return f"[{', '.join(formatted_elements)}]"
        elif isinstance(value, dict):
            return str(value).replace("'", '"')
        else:
            return str(value)

    def _update_parameter_metadata_in_code(
        self, source_code, parameter_key, field_name, field_value
    ):
        """ソースコード内のパラメーターメタデータ（default_value, constraintsなど）を更新"""
        updated_code = source_code

        print(
            "これフィールドのデータ全部",
            f"key:{parameter_key}, name:{field_name}, value:{field_value}",
            flush=True,
        )

        # まず、parameters={} 内に指定されたparameter_keyが存在するかチェック
        if not self._parameter_exists_in_parameters_dict(source_code, parameter_key):
            # parameters内に存在しない場合は何もしない
            logger.info(
                f"Parameter '{parameter_key}' not found in parameters dict, skipping"
            )
            return source_code

        logger.info(
            f"Starting update for parameter '{parameter_key}', field '{field_name}'"
        )

        # シンプルなアプローチ: ParameterDefinition内のfield_nameを直接更新
        return self._update_parameter_field_simple(
            updated_code, parameter_key, field_name, field_value
        )

    def _update_parameter_field_simple(
        self, source_code, parameter_key, field_name, field_value
    ):
        """シンプルな正規表現でParameterDefinition内のフィールドを更新"""

        # ParameterDefinitionパターンを探す（より柔軟なパターン）
        # 'parameter_key': ParameterDefinition( ... field_name=old_value, ... )
        param_pattern = (
            rf"(['\"]){re.escape(parameter_key)}\1\s*:\s*ParameterDefinition\s*\("
        )

        match = re.search(param_pattern, source_code)
        if not match:
            logger.warning(
                f"Parameter '{parameter_key}' not found in ParameterDefinition format"
            )
            return source_code

        # ParameterDefinition全体の範囲で検索・置換
        start_pos = match.start()

        # ParameterDefinitionの終了位置を見つける（より正確な方法）
        paren_count = 0
        end_pos = match.end()
        in_string = False
        quote_char = None
        
        for i in range(match.end(), len(source_code)):
            char = source_code[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                elif char == "(":
                    paren_count += 1
                elif char == ")":
                    if paren_count == 0:
                        end_pos = i + 1
                        break
                    paren_count -= 1
            else:
                if char == quote_char and (i == 0 or source_code[i-1] != '\\'):
                    in_string = False
                    quote_char = None

        # ParameterDefinition部分を抽出
        param_def = source_code[start_pos:end_pos]

        # field_nameのフィールドを安全に更新
        updated_param_def = self._replace_parameter_field_value(param_def, field_name, field_value)

        if updated_param_def != param_def:
            updated_code = (
                source_code[:start_pos] + updated_param_def + source_code[end_pos:]
            )
            logger.info(
                f"Successfully updated {parameter_key}.{field_name} to {field_value}"
            )
            return updated_code
        else:
            logger.info(f"No changes made to {parameter_key}.{field_name}")
            return source_code
    
    def _replace_parameter_field_value(self, param_def, field_name, field_value):
        """ParameterDefinition内の特定フィールドを安全に置換"""
        # フィールド名のパターンを探す
        field_pattern = rf'\b{re.escape(field_name)}\s*=\s*'
        match = re.search(field_pattern, param_def)
        
        if not match:
            return param_def
            
        # フィールド値の開始位置
        value_start = match.end()
        
        # フィールド値の終端を検出（配列・辞書のネストに対応）
        value_end = self._find_parameter_field_value_end(param_def, value_start)
        
        if value_end > value_start:
            formatted_value = self._format_value_for_python(field_value)
            # 値を完全に置換
            return (
                param_def[:value_start] + 
                formatted_value + 
                param_def[value_end:]
            )
        
        return param_def
    
    def _find_parameter_field_value_end(self, param_def, start_pos):
        """ParameterDefinition内のフィールド値の終端位置を検出"""
        bracket_count = 0
        brace_count = 0
        paren_count = 0
        in_string = False
        quote_char = None
        i = start_pos
        
        while i < len(param_def):
            char = param_def[i]
            
            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    quote_char = char
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '(':
                    paren_count += 1
                elif char == ')':
                    if paren_count > 0:
                        paren_count -= 1
                    else:
                        # ParameterDefinitionの終了
                        return i
                elif char == ',' and bracket_count == 0 and brace_count == 0 and paren_count == 0:
                    # フィールドの終端
                    return i
            else:
                if char == quote_char and (i == 0 or param_def[i-1] != '\\'):
                    in_string = False
                    quote_char = None
            
            i += 1
        
        return len(param_def)

    def _update_specific_parameter_field(
        self, source_code, parameter_key, field_name, field_value
    ):
        """指定されたパラメーターの指定されたフィールドのみを更新（より確実な方法）"""

        # 完全一致する '"parameter_key": ParameterDefinition(' パターンを探す
        exact_patterns = [
            rf'"{re.escape(parameter_key)}"\s*:\s*ParameterDefinition\s*\(',  # ダブルクォート
            rf"'{re.escape(parameter_key)}'\s*:\s*ParameterDefinition\s*\(",  # シングルクォート
        ]

        target_match = None
        target_pattern = None

        for pattern in exact_patterns:
            matches = list(re.finditer(pattern, source_code))
            if matches:
                # 完全一致するもののみを選択
                for match in matches:
                    # マッチした部分の前後をチェックして、完全な独立したキーかを確認
                    start_pos = match.start()

                    # 前の文字をチェック（英数字でないことを確認）
                    if start_pos > 0:
                        prev_char = source_code[start_pos - 1]
                        if prev_char.isalnum() or prev_char == "_":
                            continue  # 部分マッチなのでスキップ

                    target_match = match
                    target_pattern = pattern
                    logger.info(
                        f"Found exact match for '{parameter_key}' using pattern: {pattern}"
                    )
                    break

                if target_match:
                    break

        if not target_match:
            logger.warning(f"No exact match found for parameter '{parameter_key}'")
            return source_code

        # ParameterDefinition(...) の終端を見つける
        start_pos = target_match.end() - 1  # '(' の位置
        end_pos = self._find_matching_paren(source_code, start_pos)

        if end_pos == -1:
            logger.error(
                f"Could not find matching closing paren for parameter '{parameter_key}'"
            )
            return source_code

        # ParameterDefinition内の内容を取得
        param_content = source_code[start_pos + 1 : end_pos]

        logger.info(f"Extracted content for '{parameter_key}': {param_content}")

        # この内容内で指定されたフィールドを更新
        updated_content = self._update_field_in_parameter_content(
            param_content, field_name, field_value
        )

        logger.info(f"Updated content for '{parameter_key}': {updated_content}")

        # 元のコードを更新
        if updated_content != param_content:
            updated_code = (
                source_code[: start_pos + 1] + updated_content + source_code[end_pos:]
            )
            logger.info(
                f"Successfully updated parameter '{parameter_key}' field '{field_name}'"
            )
        else:
            logger.info(
                f"No changes needed for parameter '{parameter_key}' field '{field_name}'"
            )

        return updated_code

    def _find_matching_paren(self, source_code, start_pos):
        """指定位置の開き括弧に対応する閉じ括弧の位置を見つける"""
        paren_count = 0
        in_string = False
        escape_next = False
        quote_char = None

        for i, char in enumerate(source_code[start_pos:], start_pos):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if not in_string and char in ['"', "'"]:
                in_string = True
                quote_char = char
            elif in_string and char == quote_char:
                in_string = False
                quote_char = None
            elif not in_string:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        return i

        return -1  # 対応する閉じ括弧が見つからない

    def _update_field_in_parameter_content(
        self, param_content, field_name, field_value
    ):
        """ParameterDefinition内のコンテンツで特定フィールドを更新"""
        formatted_value = self._format_value_for_python(field_value)

        # より正確なパターン: フィールド名の境界を厳密に定義
        # ネストした辞書や複雑な値も正確に抽出
        field_pattern = (
            rf"\b{re.escape(field_name)}\s*=\s*(?:[^,\)]*(?:\([^)]*\)[^,\)]*)*)"
        )

        # 値の終端をより正確に検出するため、括弧のネストも考慮
        def find_field_value_end(content, start_pos):
            """フィールド値の終端位置を見つける"""
            paren_count = 0
            brace_count = 0
            bracket_count = 0
            in_string = False
            quote_char = None
            i = start_pos

            while i < len(content):
                char = content[i]

                if not in_string:
                    if char in ['"', "'"]:
                        in_string = True
                        quote_char = char
                    elif char == "(":
                        paren_count += 1
                    elif char == ")":
                        if paren_count > 0:
                            paren_count -= 1
                        else:
                            # 外側の括弧に到達
                            return i
                    elif char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                    elif char == "[":
                        bracket_count += 1
                    elif char == "]":
                        bracket_count -= 1
                    elif (
                        char == ","
                        and paren_count == 0
                        and brace_count == 0
                        and bracket_count == 0
                    ):
                        # フィールドの終端
                        return i
                else:
                    if char == quote_char and (i == 0 or content[i - 1] != "\\"):
                        in_string = False
                        quote_char = None

                i += 1

            return len(content)

        # フィールドを検索して更新
        field_pattern = rf"\b{re.escape(field_name)}\s*=\s*"
        field_match = re.search(field_pattern, param_content)

        logger.info(f"Searching for field '{field_name}' in content: {param_content}")
        logger.info(f"Field pattern: {field_pattern}")
        logger.info(f"Field match found: {field_match is not None}")

        if field_match:
            # 既存フィールドを更新
            field_start = field_match.start()
            value_start = field_match.end()
            value_end = find_field_value_end(param_content, value_start)

            logger.info(
                f"Field position: start={field_start}, value_start={value_start}, value_end={value_end}"
            )
            logger.info(f"Original value: {param_content[value_start:value_end]}")

            # フィールド部分を置換
            updated_content = (
                param_content[:field_start]
                + f"{field_name}={formatted_value}"
                + param_content[value_end:]
            )
            logger.info(f"Replacement result: {updated_content}")
            return updated_content
        else:
            # 新しいフィールドを追加
            param_content = param_content.strip()
            if param_content:
                # 既存のパラメーターがある場合
                return f"{param_content}, {field_name}={formatted_value}"
            else:
                # 空の場合
                return f"{field_name}={formatted_value}"

    def _parameter_exists_in_parameters_dict(self, source_code, parameter_key):
        """parameters={}辞書内に指定されたパラメーターキーが存在するかチェック"""

        # parameters = { ... } のブロックを探す
        params_pattern = r"parameters\s*=\s*\{"

        match = re.search(params_pattern, source_code)

        if not match:
            return False

        # parameters辞書の開始位置
        start_pos = match.end() - 1  # '{' の位置

        # 辞書の終わりを見つける（ネストした括弧を考慮）
        brace_count = 0
        end_pos = len(source_code) - 1  # デフォルトでファイル終端に設定
        in_string = False
        escape_next = False
        quote_char = None

        for i, char in enumerate(source_code[start_pos:], start_pos):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if not in_string and char in ['"', "'"]:
                in_string = True
                quote_char = char
            elif in_string and char == quote_char:
                in_string = False
                quote_char = None
            elif not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break

        # parameters辞書の内容を取得
        params_content = source_code[start_pos : end_pos + 1]

        # print("これパラメーターコンテンツ", params_content, flush=True)

        # 指定されたparameter_keyが含まれているかチェック
        # ダブルクォートまたはシングルクォートでマッチ
        patterns = [
            rf'"{re.escape(parameter_key)}"\s*:\s*ParameterDefinition',
            rf"'{re.escape(parameter_key)}'\s*:\s*ParameterDefinition",
        ]

        for pattern in patterns:
            if re.search(pattern, params_content):
                return True

        return False


@method_decorator(csrf_exempt, name="dispatch")
class NodeCategoryListView(APIView):
    """ノードカテゴリ一覧取得用のビュー"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def get(self, request):
        """利用可能なカテゴリ一覧を返す"""
        try:
            categories = [
                {"value": value, "label": label} for value, label in NODE_CATEGORIES
            ]

            return Response({"categories": categories, "default": "analysis"})
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return Response(
                {"error": "Failed to get categories"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@method_decorator(csrf_exempt, name="dispatch")
class BulkSyncNodesView(APIView):
    """nodesフォルダの内容を一括でDBに同期するビュー"""

    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        """nodesフォルダをスキャンしてDBに一括追加"""
        try:
            nodes_path = Path(settings.MEDIA_ROOT)

            if not nodes_path.exists():
                return Response(
                    {"error": f"Nodes folder not found: {nodes_path}"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            results = {
                "total_scanned": 0,
                "added": 0,
                "skipped": 0,
                "errors": 0,
                "files": {"added": [], "skipped": [], "errors": []},
            }

            # 各カテゴリフォルダをスキャン
            valid_categories = [category[0] for category in NODE_CATEGORIES]

            for category in valid_categories:
                category_path = nodes_path / category

                if not category_path.exists():
                    logger.info(f"Category folder not found, creating: {category_path}")
                    category_path.mkdir(parents=True, exist_ok=True)
                    continue

                # .pyファイルをスキャン
                for py_file in category_path.glob("*.py"):
                    results["total_scanned"] += 1
                    result = self._process_file(py_file, category)

                    if result["status"] == "added":
                        results["added"] += 1
                        results["files"]["added"].append(result)
                    elif result["status"] == "skipped":
                        results["skipped"] += 1
                        results["files"]["skipped"].append(result)
                    else:
                        results["errors"] += 1
                        results["files"]["errors"].append(result)

            return Response(results, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Bulk sync failed: {e}")
            return Response(
                {"error": "Bulk sync failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _process_file(self, file_path, category):
        """個別ファイルの処理"""
        try:
            filename = file_path.name

            # ファイル内容を読み取り
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # ファイルハッシュで重複チェック
            file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

            # 既存ファイルチェック（ハッシュまたはファイル名+カテゴリで、アクティブなファイルのみ）
            existing_file = PythonFile.objects.filter(
                (
                    models.Q(file_hash=file_hash)
                    | models.Q(name=filename, category=category)
                )
                & models.Q(is_active=True)
            ).first()

            if existing_file:
                # より詳細な重複理由を提供
                if existing_file.file_hash == file_hash:
                    reason = "Identical content already exists"
                else:
                    reason = "File with same name in same category already exists"

                return {
                    "status": "skipped",
                    "filename": filename,
                    "category": category,
                    "reason": reason,
                    "existing_id": str(existing_file.id),
                    "existing_name": existing_file.name,
                }

            # 新しいファイルとして作成（DB登録のみ、ファイルコピーなし）
            python_file = PythonFile.objects.create(
                name=filename,
                description=f"Synced from {category} folder",
                category=category,
                file_content=file_content,
                file_size=file_path.stat().st_size,
                file_hash=file_hash,
                # fileフィールドは空のまま（file_contentがあるので不要）
            )

            # 自動解析実行
            try:
                file_service = PythonFileService()
                file_service._analyze_file(python_file)
            except Exception as e:
                logger.warning(f"Analysis failed for {filename}: {e}")

            return {
                "status": "added",
                "filename": filename,
                "category": category,
                "file_id": str(python_file.id),
                "analyzed": python_file.is_analyzed,
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                "status": "error",
                "filename": file_path.name if file_path else "unknown",
                "category": category,
                "error": str(e),
            }
