import ast
from typing import Dict, List, Optional, Any
from enum import Enum
import sqlite3
import json


class PortTypeMapping(Enum):
    """PortType enumeration mapping."""

    ANY = "any"
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    OBJECT = "object"


class NodeDatabase:
    """Database handler for node information."""

    def __init__(self, db_path: str = "nodes.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Nodes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    node_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Ports table (inputs and outputs)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER,
                    port_name TEXT NOT NULL,
                    port_type TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    description TEXT,
                    is_input BOOLEAN NOT NULL,
                    FOREIGN KEY (node_id) REFERENCES nodes (id),
                    UNIQUE(node_id, port_name, is_input)
                )
            """
            )

            # Parameters table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER,
                    param_name TEXT NOT NULL,
                    default_value TEXT,
                    description TEXT,
                    constraints TEXT,
                    FOREIGN KEY (node_id) REFERENCES nodes (id),
                    UNIQUE(node_id, param_name)
                )
            """
            )

            # Methods table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS methods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER,
                    method_name TEXT NOT NULL,
                    description TEXT,
                    input_ports TEXT,
                    output_ports TEXT,
                    FOREIGN KEY (node_id) REFERENCES nodes (id),
                    UNIQUE(node_id, method_name)
                )
            """
            )

            conn.commit()

    def save_node(self, node_info: Dict[str, Any]) -> int:
        """Save node information to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert or update node
            cursor.execute(
                """
                INSERT OR REPLACE INTO nodes (class_name, description, node_type)
                VALUES (?, ?, ?)
            """,
                (
                    node_info["class_name"],
                    node_info.get("description", ""),
                    node_info.get("node_type", ""),
                ),
            )

            node_id = cursor.lastrowid

            # Clear existing related data
            cursor.execute("DELETE FROM ports WHERE node_id = ?", (node_id,))
            cursor.execute("DELETE FROM parameters WHERE node_id = ?", (node_id,))
            cursor.execute("DELETE FROM methods WHERE node_id = ?", (node_id,))

            # Save inputs
            for port_name, port_info in node_info.get("inputs", {}).items():
                cursor.execute(
                    """
                    INSERT INTO ports (node_id, port_name, port_type, data_type, description, is_input)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        node_id,
                        port_name,
                        "input",
                        port_info.get("type", "any"),
                        port_info.get("description", ""),
                        True,
                    ),
                )

            # Save outputs
            for port_name, port_info in node_info.get("outputs", {}).items():
                cursor.execute(
                    """
                    INSERT INTO ports (node_id, port_name, port_type, data_type, description, is_input)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        node_id,
                        port_name,
                        "output",
                        port_info.get("type", "any"),
                        port_info.get("description", ""),
                        False,
                    ),
                )

            # Save parameters
            for param_name, param_info in node_info.get("parameters", {}).items():
                cursor.execute(
                    """
                    INSERT INTO parameters (node_id, param_name, default_value, description, constraints)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        node_id,
                        param_name,
                        str(param_info.get("default_value", "")),
                        param_info.get("description", ""),
                        json.dumps(param_info.get("constraints", {})),
                    ),
                )

            # Save methods
            for method_name, method_info in node_info.get("methods", {}).items():
                cursor.execute(
                    """
                    INSERT INTO methods (node_id, method_name, description, input_ports, output_ports)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        node_id,
                        method_name,
                        method_info.get("description", ""),
                        json.dumps(method_info.get("inputs", [])),
                        json.dumps(method_info.get("outputs", [])),
                    ),
                )

            conn.commit()
            return node_id


class PythonNodeAnalyzer:
    """Pythonファイルからノード情報を解析するサービス"""

    def __init__(self, db_path: str = "nodes.db"):
        self.db = NodeDatabase(db_path)

    def analyze_file_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Pythonファイルの内容を解析してノード情報を抽出

        Args:
            content: Pythonファイルの内容

        Returns:
            List of node information dictionaries
        """
        try:
            tree = ast.parse(content)
            nodes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    node_info = self._analyze_class_node(node, content, tree)
                    if node_info:
                        nodes.append(node_info)
                        # Save to database
                        node_id = self.db.save_node(node_info)
                        print(
                            f"Saved node '{node_info['class_name']}' with ID: {node_id}"
                        )

            return nodes
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

    def _analyze_class_node(
        self, class_node: ast.ClassDef, content: str, tree: ast.AST
    ) -> Optional[Dict[str, Any]]:
        """
        クラスノードを解析してノード情報を抽出

        Args:
            class_node: ASTクラスノード
            content: 元のファイル内容
            tree: 完全なAST

        Returns:
            Node information dictionary or None
        """
        # NODE_DEFINITION属性を探す
        node_definition = None
        for node in class_node.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "NODE_DEFINITION"
            ):
                node_definition = node.value
                break

        if not node_definition:
            return None

        # NODE_DEFINITIONから情報を抽出
        try:
            definition_info = self._extract_node_definition_ast(node_definition, tree)
            if definition_info:
                return {
                    "class_name": class_node.name,
                    "description": definition_info.get("description", ""),
                    "node_type": definition_info.get("type", ""),
                    "inputs": definition_info.get("inputs", {}),
                    "outputs": definition_info.get("outputs", {}),
                    "parameters": definition_info.get("parameters", {}),
                    "methods": definition_info.get("methods", {}),
                }
        except Exception as e:
            print(f"Error analyzing class {class_node.name}: {e}")
            return None

    def _extract_node_definition_ast(
        self, node_def: ast.AST, tree: ast.AST
    ) -> Dict[str, Any]:
        """
        ASTを使ってNODE_DEFINITIONから情報を抽出

        Args:
            node_def: NODE_DEFINITION のASTノード
            tree: 完全なAST

        Returns:
            Extracted definition information
        """
        result = {}

        if not isinstance(node_def, ast.Call):
            return result

        # NodeDefinitionSchemaの引数を解析
        for keyword in node_def.keywords:
            if keyword.arg == "description":
                result["description"] = self._extract_string_value(keyword.value)
            elif keyword.arg == "type":
                result["type"] = self._extract_string_value(keyword.value)
            elif keyword.arg == "inputs":
                result["inputs"] = self._extract_port_dict(keyword.value, tree)
            elif keyword.arg == "outputs":
                result["outputs"] = self._extract_port_dict(keyword.value, tree)
            elif keyword.arg == "parameters":
                result["parameters"] = self._extract_parameter_dict(keyword.value)
            elif keyword.arg == "methods":
                result["methods"] = self._extract_method_dict(keyword.value)

        return result

    def _extract_string_value(self, node: ast.AST) -> str:
        """文字列値を抽出"""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Str):  # Python 3.7以前の互換性
            return node.s
        return ""

    def _extract_port_dict(
        self, node: ast.AST, tree: ast.AST
    ) -> Dict[str, Dict[str, Any]]:
        """ポート定義辞書を抽出"""
        if not isinstance(node, ast.Dict):
            return {}

        ports = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, (ast.Constant, ast.Str)):
                port_name = self._extract_string_value(key)
                port_info = self._extract_port_definition(value, tree)
                if port_info:
                    ports[port_name] = port_info

        return ports

    def _extract_port_definition(self, node: ast.AST, tree: ast.AST) -> Dict[str, Any]:
        """PortDefinitionから情報を抽出"""
        if not isinstance(node, ast.Call):
            return {}

        port_info = {}
        for keyword in node.keywords:
            if keyword.arg == "type":
                port_info["type"] = self._extract_port_type(keyword.value, tree)
            elif keyword.arg == "description":
                port_info["description"] = self._extract_string_value(keyword.value)

        return port_info

    def _extract_port_type(self, node: ast.AST, tree: ast.AST) -> str:
        """PortTypeを抽出してマッピング"""
        if isinstance(node, ast.Attribute):
            # PortType.OBJECT のような形式
            if isinstance(node.value, ast.Name) and node.value.id == "PortType":
                port_type_name = node.attr.upper()
                return self._map_port_type(port_type_name)
        elif isinstance(node, ast.Name):
            # 直接の変数参照の場合
            return self._map_port_type(node.id.upper())
        elif isinstance(node, (ast.Constant, ast.Str)):
            # 直接の文字列の場合
            return self._extract_string_value(node).lower()

        return "any"

    def _map_port_type(self, port_type_name: str) -> str:
        """
        PortType列挙型をフロントエンド用の型にマッピング

        Args:
            port_type_name: PortType の名前（例: "OBJECT", "DICT"）

        Returns:
            フロントエンド用の型名
        """
        type_mapping = {
            "INT": "int",
            "FLOAT": "float",
            "STR": "str",
            "BOOL": "bool",
            "LIST": "list",
            "DICT": "dict",  # 明示的にDICTをdictにマッピング
            "OBJECT": "object",  # 明示的にOBJECTをobjectにマッピング
            "ANY": "any",
        }

        mapped_type = type_mapping.get(port_type_name.upper(), "any")
        print(f"Debug: Mapping {port_type_name} -> {mapped_type}")  # デバッグ用
        return mapped_type

    def _extract_parameter_dict(self, node: ast.AST) -> Dict[str, Dict[str, Any]]:
        """パラメータ定義辞書を抽出"""
        if not isinstance(node, ast.Dict):
            return {}

        parameters = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, (ast.Constant, ast.Str)):
                param_name = self._extract_string_value(key)
                param_info = self._extract_parameter_definition(value)
                if param_info:
                    parameters[param_name] = param_info

        return parameters

    def _extract_parameter_definition(self, node: ast.AST) -> Dict[str, Any]:
        """ParameterDefinitionから情報を抽出"""
        if not isinstance(node, ast.Call):
            return {}

        param_info = {}
        for keyword in node.keywords:
            if keyword.arg == "default_value":
                param_info["default_value"] = self._extract_value(keyword.value)
            elif keyword.arg == "description":
                param_info["description"] = self._extract_string_value(keyword.value)
            elif keyword.arg == "constraints":
                param_info["constraints"] = self._extract_constraints(keyword.value)

        return param_info

    def _extract_method_dict(self, node: ast.AST) -> Dict[str, Dict[str, Any]]:
        """メソッド定義辞書を抽出"""
        if not isinstance(node, ast.Dict):
            return {}

        methods = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, (ast.Constant, ast.Str)):
                method_name = self._extract_string_value(key)
                method_info = self._extract_method_definition(value)
                if method_info:
                    methods[method_name] = method_info

        return methods

    def _extract_method_definition(self, node: ast.AST) -> Dict[str, Any]:
        """MethodDefinitionから情報を抽出"""
        if not isinstance(node, ast.Call):
            return {}

        method_info = {}
        for keyword in node.keywords:
            if keyword.arg == "description":
                method_info["description"] = self._extract_string_value(keyword.value)
            elif keyword.arg == "inputs":
                method_info["inputs"] = self._extract_list_values(keyword.value)
            elif keyword.arg == "outputs":
                method_info["outputs"] = self._extract_list_values(keyword.value)

        return method_info

    def _extract_list_values(self, node: ast.AST) -> List[str]:
        """リストから文字列値を抽出"""
        if not isinstance(node, ast.List):
            return []

        values = []
        for item in node.elts:
            value = self._extract_string_value(item)
            if value:
                values.append(value)

        return values

    def _extract_value(self, node: ast.AST) -> Any:
        """汎用的な値を抽出"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, (ast.Str, ast.Num)):  # Python 3.7以前の互換性
            return node.s if isinstance(node, ast.Str) else node.n
        elif isinstance(node, ast.List):
            # 配列の要素を再帰的に抽出
            return [self._extract_value(elem) for elem in node.elts]
        elif isinstance(node, ast.Dict):
            # 辞書の要素を再帰的に抽出
            result = {}
            for key, value in zip(node.keys, node.values):
                key_str = self._extract_value(key)
                result[key_str] = self._extract_value(value)
            return result
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return str(node)

    def _extract_constraints(self, node: ast.AST) -> Dict[str, Any]:
        """制約辞書を抽出"""
        if not isinstance(node, ast.Dict):
            return {}

        constraints = {}
        for key, value in zip(node.keys, node.values):
            if isinstance(key, (ast.Constant, ast.Str)):
                constraint_name = self._extract_string_value(key)
                constraint_value = self._extract_value(value)
                constraints[constraint_name] = constraint_value

        return constraints
