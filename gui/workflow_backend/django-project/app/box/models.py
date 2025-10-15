from django.db import models
from django.contrib.auth.models import User
import uuid
import os


# カテゴリの選択肢
NODE_CATEGORIES = [
    ('analysis', 'Analysis'),
    ('io', 'I/O'),
    ('network', 'Network'),
    ('optimization', 'Optimization'),
    ('simulation', 'Simulation'),
    ('stimulus', 'Stimulus'),
]


def get_upload_path(instance, filename):
    """カテゴリに基づいてアップロード先を決定"""
    category = getattr(instance, 'category', 'uncategorized')
    return os.path.join(category, filename)


class PythonFile(models.Model):
    """アップロードされたPythonファイルモデル"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    category = models.CharField(
        max_length=50,
        choices=NODE_CATEGORIES,
        default='analysis',
        help_text='Node category for organizing files'
    )
    file = models.FileField(upload_to=get_upload_path)
    file_content = models.TextField(default="")
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="uploaded_files",
        null=True,
        blank=True,
    )

    # ノード解析結果
    node_classes = models.JSONField(
        default=dict, blank=True
    )  # 解析されたノードクラス情報
    is_analyzed = models.BooleanField(default=False)  # 解析済みフラグ
    analysis_error = models.TextField(blank=True, null=True)  # 解析エラー情報

    # メタデータ
    file_size = models.IntegerField(default=0)  # ファイルサイズ（バイト）
    file_hash = models.CharField(
        max_length=64, unique=True, default="default"  # 一時的なデフォルト値
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "box_pythonfile"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def get_node_classes_for_frontend(self):
        """フロントエンド用のノードクラス情報を返す"""
        if not self.node_classes:
            return []

        frontend_nodes = []
        for class_name, class_info in self.node_classes.items():
            # 元の構造を保持したままフロントエンド用に整形
            frontend_node = {
                "id": f"uploaded_{self.id}_{class_name}",
                "type": "uploadedNode",
                "label": class_name,
                "description": class_info.get("description", ""),
                "category": self.get_category_display(),
                "file_id": str(self.id),
                "class_name": class_name,
                "file_name": self.name,
                # schemaにすべての情報を含める
                "schema": self._convert_to_full_schema(class_info),
            }

            frontend_nodes.append(frontend_node)

        return frontend_nodes

    def _convert_to_full_schema(self, class_info):
        """すべての情報をschemaに含める（オブジェクト形式）"""
        schema = {
            "inputs": {},
            "outputs": {},
            "parameters": {},
            "methods": class_info.get("methods", {}),
        }

        # inputs を変換
        if "inputs" in class_info:
            for input_name, input_info in class_info["inputs"].items():
                schema["inputs"][input_name] = {
                    "name": input_name,
                    "type": self._map_port_type_to_frontend(
                        input_info.get("type", "any")
                    ),
                    "description": input_info.get("description", ""),
                    "port_direction": "input",
                    "required": input_info.get("required", False),
                    "optional": input_info.get("optional", False),
                }
                # default_valueがある場合のみ追加
                if "default_value" in input_info:
                    schema["inputs"][input_name]["default_value"] = input_info[
                        "default_value"
                    ]
                # constraintsがある場合のみ追加
                if "constraints" in input_info:
                    schema["inputs"][input_name]["constraints"] = input_info[
                        "constraints"
                    ]

        # outputs を変換
        if "outputs" in class_info:
            for output_name, output_info in class_info["outputs"].items():
                schema["outputs"][output_name] = {
                    "name": output_name,
                    "type": self._map_port_type_to_frontend(
                        output_info.get("type", "any")
                    ),
                    "description": output_info.get("description", ""),
                    "port_direction": "output",
                    "optional": output_info.get("optional", False),
                }

        # parameters を変換
        if "parameters" in class_info:
            schema["parameters"] = self._convert_parameters(
                class_info.get("parameters", {})
            )

        return schema

    def _convert_parameters(self, parameters):
        """parametersを変換（元の構造を保持しつつ、必要な情報を追加）"""
        converted_params = {}

        for param_name, param_info in parameters.items():
            param_data = {
                "name": param_name,
                "type": self._map_port_type_to_frontend(param_info.get("type", "any")),
                "description": param_info.get("description", ""),
            }

            # default_valueがある場合のみ追加
            if "default_value" in param_info:
                param_data["default_value"] = param_info["default_value"]

            # constraintsがある場合のみ追加（そのまま保持）
            if "constraints" in param_info:
                param_data["constraints"] = param_info["constraints"]

            # widget_typeなどの追加メタデータがある場合
            if "widget_type" in param_info:
                param_data["widget_type"] = param_info["widget_type"]

            converted_params[param_name] = param_data

        return converted_params

    def _map_port_type_to_frontend(self, port_type):
        """PortTypeをフロントエンド用の型に変換"""
        type_mapping = {
            "int": "int",
            "float": "float",
            "str": "str",
            "bool": "bool",
            "list": "list",
            "dict": "dict",
            "object": "object",
            "any": "any",
        }
        return type_mapping.get(str(port_type).lower(), "any")
