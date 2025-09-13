import re
import os
import json
from pathlib import Path
from django.conf import settings
from .models import FlowProject, FlowNode, FlowEdge
import logging
import traceback

logger = logging.getLogger(__name__)


class CodeGenerationService:
    """ワークフローからPythonコードを生成するサービス（.ipynb変換機能付き）"""

    def __init__(self):
        self.code_dir = Path(settings.BASE_DIR) / "projects"
        self.code_dir.mkdir(exist_ok=True)

        # 正規表現パターンを事前定義
        self._compile_patterns()

    def _compile_patterns(self):
        """使用する正規表現パターンをコンパイル"""
        self.patterns = {
            # WorkflowBuilderインポートを検出
            "workflow_builder_import": re.compile(
                r"^(from\s+neuroworkflow\s+import\s+WorkflowBuilder)$", re.MULTILINE
            ),
        }

    def get_code_file_path(self, project_id):
        """プロジェクトIDからコードファイルパスを取得"""
        return self.code_dir / str(project_id) / f"{project_id}.py"

    def get_notebook_file_path(self, project_id):
        """プロジェクトIDからnotebookファイルパスを取得"""
        return self.code_dir / str(project_id) / f"{project_id}.ipynb"

    def _convert_py_to_ipynb(self, project_id):
        """Pythonファイルをjupyter notebookに変換"""
        try:
            code_file = self.get_code_file_path(project_id)
            notebook_file = self.get_notebook_file_path(project_id)

            if not code_file.exists():
                logger.error(f"Python file does not exist: {code_file}")
                return False

            with open(code_file, "r", encoding="utf-8") as f:
                py_content = f.read()

            # Pythonコードをnotebook形式に変換
            notebook_content = self._create_notebook_from_python(py_content)

            # notebookファイルに保存
            with open(notebook_file, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully converted to notebook: {notebook_file}")
            return True

        except Exception as e:
            logger.error(f"Error converting to notebook: {e}")
            logger.error(traceback.format_exc())
            return False

    def _create_notebook_from_python(self, py_content):
        """Pythonコードからnotebook構造を作成"""
        # Pythonコードを適切なセルに分割
        cells = self._split_python_into_cells(py_content)

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        return notebook

    def _split_python_into_cells(self, py_content):
        """Pythonコードを適切なセルに分割"""
        lines = py_content.split("\n")
        cells = []
        current_cell = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # コメントブロックをmarkdownセルとして扱う
            if line.strip().startswith('"""') and len(line.strip()) > 3:
                # 現在のセルを保存
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []

                docstring_content = []
                docstring_content.append(line.strip()[3:])  # 最初の"""を除去
                i += 1

                while i < len(lines) and not lines[i].strip().endswith('"""'):
                    docstring_content.append(lines[i])
                    i += 1

                if i < len(lines):
                    # 最後の"""を除去
                    last_line = lines[i].rstrip()
                    if last_line.endswith('"""'):
                        last_line = last_line[:-3]
                    if last_line:
                        docstring_content.append(last_line)

                markdown_content = "\n".join(docstring_content).strip()
                if markdown_content:
                    cell = self._create_markdown_cell(markdown_content)
                    if cell:
                        cells.append(cell)

            # インポート部分を一つのセルにまとめる
            elif line.strip().startswith(
                ("import ", "from ")
            ) or line.strip().startswith("sys.path"):
                if current_cell and not any(
                    l.strip().startswith(("import ", "from ", "sys.path"))
                    for l in current_cell
                ):
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []
                current_cell.append(line)

            # 関数定義の開始 - 関数全体を一つのセルに
            elif line.strip().startswith("def "):
                # 現在のセルを保存
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []

                # 関数全体を読み込む
                function_lines = [line]
                i += 1

                # 関数の中身を全て読み込む（インデントで判断）
                while i < len(lines):
                    next_line = lines[i]
                    # 空行は含める
                    if next_line.strip() == "":
                        function_lines.append(next_line)
                    # インデントがある行または関数内のコメント
                    elif next_line.startswith("    ") or next_line.startswith("\t"):
                        function_lines.append(next_line)
                    # 新しい関数定義、クラス定義、またはトップレベルコードが始まった
                    elif next_line.strip().startswith(
                        ("def ", "class ", "if __name__")
                    ) or (next_line.strip() and not next_line.startswith((" ", "\t"))):
                        # 関数終了、インデックスを戻す
                        i -= 1
                        break
                    else:
                        function_lines.append(next_line)
                    i += 1

                # 関数セルを作成
                cell = self._create_code_cell("\n".join(function_lines))
                if cell:
                    cells.append(cell)

            # クラス定義の開始
            elif line.strip().startswith("class "):
                # 現在のセルを保存
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []
                current_cell.append(line)

            # メイン実行部分 - if __name__ == "__main__": から最後まで
            elif line.strip() == 'if __name__ == "__main__":':
                # 現在のセルを保存
                if current_cell:
                    cell = self._create_code_cell("\n".join(current_cell))
                    if cell:
                        cells.append(cell)
                    current_cell = []

                # メイン部分の開始
                main_lines = [line]
                i += 1

                # ファイルの最後まで全て読み込む
                while i < len(lines):
                    main_lines.append(lines[i])
                    i += 1

                # メイン実行セルを作成
                cell = self._create_code_cell("\n".join(main_lines))
                if cell:
                    cells.append(cell)
                break  # ファイル終端なのでループ終了

            else:
                current_cell.append(line)

            i += 1

        # 残りのコードを追加
        if current_cell:
            cell = self._create_code_cell("\n".join(current_cell))
            if cell:
                cells.append(cell)

        return cells

    def _create_code_cell(self, source_code):
        """コードセルを作成"""
        # 空のコードは除外
        if not source_code.strip():
            return None

        # 改行を保持するため、各行を配列の要素として保持し、最後に改行文字を追加
        lines = source_code.split("\n")
        # 最後の行以外は改行文字を追加
        source_lines = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:  # 最後の行以外
                source_lines.append(line + "\n")
            else:  # 最後の行
                source_lines.append(line)

        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines,
        }

    def _create_markdown_cell(self, markdown_text):
        """マークダウンセルを作成"""
        # 改行を保持するため、各行を配列の要素として保持し、最後に改行文字を追加
        lines = markdown_text.split("\n")
        # 最後の行以外は改行文字を追加
        source_lines = []
        for i, line in enumerate(lines):
            if i < len(lines) - 1:  # 最後の行以外
                source_lines.append(line + "\n")
            else:  # 最後の行
                source_lines.append(line)

        return {"cell_type": "markdown", "metadata": {}, "source": source_lines}

    def _create_base_template(self, project):
        """基本テンプレートを作成（セクションコメント付き）"""
        return f'''#!/usr/bin/env python3
"""
{project.description if project.description else f"Generated workflow for project: {project.name}"}
"""
import sys
import os

# Add paths for JupyterLab environment
sys.path.append('../neuro/src')
sys.path.append('../upload_nodes')

from neuroworkflow import WorkflowBuilder

def main():
    """Run a simple neural simulation workflow."""
    
    # Analysis field
    
    # IO field
    
    # Network field
    
    # Optimization field
    
    # Simulation field
    
    # Stimulus field
    
    # Create workflow field
    workflow = WorkflowBuilder("neural_simulation")
    
    # Print workflow information
    print(workflow)
    
    # Execute workflow
    print("\\nExecuting workflow...")
    success = workflow.execute()
    
    if success:
        print("Workflow execution completed successfully!")
    else:
        print("Workflow execution failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

    def _generate_import_statement(self, class_name):
        """クラス名から動的にimport文を生成"""
        try:
            # クラス名のバリデーション
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", class_name):
                logger.warning(f"Invalid class name format: {class_name}")
                return None

            # neuroworkflowの既知のクラスかチェック
            known_neuroworkflow_classes = {
                "BuildSonataNetworkNode": "from neuroworkflow.nodes.network import BuildSonataNetworkNode",
                "SimulateSonataNetworkNode": "from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode",
            }

            if class_name in known_neuroworkflow_classes:
                return known_neuroworkflow_classes[class_name]

            # カスタムノードとして upload_nodes からimport
            # upload_nodes/{ClassName}.py から {ClassName} をimport
            return f"from upload_nodes.{class_name} import {class_name}"

        except Exception as e:
            logger.error(f"Error generating import statement for {class_name}: {e}")
            return None

    def _generate_node_code_block(self, node):
        """ノードのコードブロックを動的に生成（categoryベース）"""
        label = node.data.get("label", "").strip()

        category = (
            node.data.get("nodeType", "")
            or getattr(node, "node_type", "")
            or node.data.get("category", "")
        ).strip()

        node_id = node.id

        logger.info(
            f"DEBUG: Generating code block for node {node_id} - label: '{label}', category: '{category}', node_data: {node.data}"
        )

        if not label:
            var_name = self._sanitize_variable_name(node_id, "node")
            logger.info(f"DEBUG: No label provided for node {node_id}")
            return f"""    # Node with no label (ID: {node_id})
        {var_name} = None  # TODO: Add implementation"""

        # categoryが'network'または'simulation'の場合のみコード生成（case insensitive）
        category_lower = category.lower()
        if category_lower not in ["network", "simulation"]:
            logger.info(
                f"DEBUG: Skipping code generation for node {node_id} - category '{category}' (normalized: '{category_lower}') not in ['network', 'simulation']"
            )
            return ""

        var_name = self._generate_variable_name_by_category(
            label, node_id, category_lower
        )
        constructor_arg = self._generate_constructor_arg_by_category(
            label, category_lower
        )
        configure_block = self._generate_configure_block_by_category(
            label, category_lower, node.data
        )

        code_block = f"""    {var_name} = {label}("{constructor_arg}")"""

        if configure_block:
            code_block += f"""
    {var_name}.configure(
{configure_block}
    )"""

        logger.info(f"DEBUG: Generated code block for node {node_id}:\n{code_block}")
        return code_block

    def _sanitize_variable_name(self, node_id, prefix):
        """ノードIDを有効な変数名に変換"""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", str(node_id))
        if sanitized and sanitized[0].isdigit():
            sanitized = f"{prefix}_{sanitized}"
        elif not sanitized:
            sanitized = prefix
        return sanitized

    def _generate_variable_name_by_category(self, class_name, node_id, category):
        """categoryベースで変数名を生成（短い名前）"""
        # node_idから数値部分だけを抽出（最初の数値部分を使用）
        import re

        match = re.search(r"\d+", node_id)
        if match:
            # 最初の6桁まで（または全体が6桁未満ならそのまま）
            short_id = match.group()[:6]
        else:
            # 数値がない場合はnode_idの最初の8文字
            short_id = node_id.replace("calc_", "").replace("_", "")[:8]

        if category == "network":
            return f"network_{short_id}"
        elif category == "simulation":
            return f"sim_{short_id}"  # simulationも短縮
        else:
            # フォールバック
            prefix = re.sub("([A-Z]+)", r"_\1", class_name).lower().strip("_")[:4]
            return f"{prefix}_{short_id}"

    def _generate_constructor_arg_by_category(self, class_name, category):
        """categoryベースでコンストラクタ引数を生成"""
        if category == "network":
            # Networkカテゴリの場合はクラス名から"Node"を除去して"Builder"を追加
            if class_name.endswith("Node"):
                base_name = class_name[:-4]  # "Node"を除去
            else:
                base_name = class_name
            return f"{base_name}Builder"
        elif category == "simulation":
            # Simulationカテゴリの場合はクラス名から"Node"を除去して"Simulation"を追加
            if class_name.endswith("Node"):
                base_name = class_name[:-4]  # "Node"を除去
            else:
                base_name = class_name
            return f"{base_name}Simulation"
        else:
            # フォールバック
            if class_name.endswith("Node"):
                return class_name[:-4]
            return class_name

    def _generate_configure_block_by_category(self, class_name, category, node_data):
        """categoryベースでconfigureブロックを生成"""
        if category == "network":
            return self._generate_network_configure_block(class_name, node_data)
        elif category == "simulation":
            return self._generate_simulation_configure_block(class_name, node_data)
        else:
            return ""  # その他のカテゴリは設定なし

    def _generate_network_configure_block(self, class_name, node_data):
        """networkカテゴリ用のconfigureブロックを生成"""
        # デフォルト設定
        config_params = {
            "sonata_path": "../data/300_pointneurons",
            "net_config_file": "circuit_config.json",
            "sim_config_file": "simulation_config.json",
            "hdf5_hyperslab_size": 1024,
        }

        # node_dataからパラメータを上書き（もし含まれている場合）
        parameters = node_data.get("parameters", {})
        for key, value in parameters.items():
            if key in config_params:
                config_params[key] = value

        # configureブロックをフォーマット
        config_lines = []
        for key, value in config_params.items():
            if isinstance(value, str):
                config_lines.append(f'            {key}="{value}"')
            else:
                config_lines.append(f"            {key}={value}")

        return ",\n".join(config_lines)

    def _generate_simulation_configure_block(self, class_name, node_data):
        """simulationカテゴリ用のconfigureブロックを生成"""
        # デフォルト設定
        config_params = {
            "simulation_time": 1000.0,
            "record_from_population": "internal",
            "record_n_neurons": 40,
        }

        # node_dataからパラメータを上書き（もし含まれている場合）
        parameters = node_data.get("parameters", {})
        for key, value in parameters.items():
            if key in config_params:
                config_params[key] = value

        # configureブロックをフォーマット
        config_lines = []
        for key, value in config_params.items():
            if isinstance(value, str):
                config_lines.append(f'            {key}="{value}"')
            else:
                config_lines.append(f"            {key}={value}")

        return ",\n".join(config_lines)

    def _get_section_name_from_category(self, category):
        """カテゴリからセクション名を取得"""
        category_to_section = {
            "analysis": "Analysis",
            "io": "IO",
            "network": "Network",
            "optimization": "Optimization",
            "simulation": "Simulation",
            "stimulus": "Stimulus",
        }
        return category_to_section.get(category.lower(), "Analysis")

    def _get_builder_name_from_label(self, label, node_id, category=None):
        """ラベルとカテゴリからBuilderでの名前を取得"""
        if category == "network":
            # Networkカテゴリの場合は"Builder"を追加
            if label.endswith("Node"):
                base_name = label[:-4]
            else:
                base_name = label
            return f"{base_name}Builder"
        elif category == "simulation":
            # Simulationカテゴリの場合は"Simulation"を追加
            if label.endswith("Node"):
                base_name = label[:-4]
            else:
                base_name = label
            return f"{base_name}Simulation"
        else:
            return f"Node_{node_id}"

    def _build_workflow_commands_from_json(self, nodes_data, edges_data):
        """ノードとエッジ情報からワークフローコマンドを生成"""
        commands = []

        # ノードIDから変数名とBuilderNameへのマッピングを作成
        node_id_to_var = {}
        node_id_to_builder = {}

        # まず全ノードの情報を収集
        for node_data in nodes_data:
            node_id = node_data.get("id", "")
            label = node_data.get("data", {}).get("label", "")

            # categoryの取得方法を修正
            category = (
                node_data.get("data", {}).get("nodeType", "")
                or node_data.get("type", "")
                or node_data.get("data", {}).get("category", "")
            ).lower()

            # network/simulationカテゴリの処理
            if category in ["network", "simulation"]:
                # _generate_variable_name_by_categoryを使用して
                # network_xxx や simulation_xxx という変数名を生成
                var_name = self._generate_variable_name_by_category(
                    label, node_id, category
                )
                builder_name = self._get_builder_name_from_label(
                    label, node_id, category
                )
                node_id_to_var[node_id] = var_name
                node_id_to_builder[node_id] = builder_name
            else:
                # その他のカテゴリは元のnode_idをそのまま使用
                node_id_to_var[node_id] = node_id
                node_id_to_builder[node_id] = f"Node_{node_id}"

        # add_nodeコマンドを生成（全ノード）
        for node_id in node_id_to_var:
            var_name = node_id_to_var[node_id]
            commands.append(f"    workflow.add_node({var_name})")

        # connectコマンドを生成（エッジごと）
        for edge_data in edges_data:
            source_id = edge_data.get("source", "")
            target_id = edge_data.get("target", "")

            # sourceHandleとtargetHandleからポート情報を取得
            source_handle = edge_data.get("sourceHandle", "")
            target_handle = edge_data.get("targetHandle", "")

            if source_id in node_id_to_var and target_id in node_id_to_var:
                # Builder名を取得
                source_builder = node_id_to_builder.get(source_id, f"Node_{source_id}")
                target_builder = node_id_to_builder.get(target_id, f"Node_{target_id}")

                # connectコマンドを生成
                commands.append(
                    f'    workflow.connect("{source_builder}", "{source_handle}", '
                    f'"{target_builder}", "{target_handle}")'
                )

        # 最後にbuild()を追加
        commands.append("    workflow.build()")

        return commands

    def generate_code_from_flow_data(self, project_id, nodes_data, edges_data):
        """React Flow JSONデータから一括でコードを生成する新しいメソッド"""
        try:
            logger.info(
                f"=== Starting batch code generation from flow data for project {project_id} ==="
            )
            logger.info(
                f"Processing {len(nodes_data)} nodes and {len(edges_data)} edges"
            )

            # プロジェクトの基本テンプレートを作成
            project = FlowProject.objects.get(id=project_id)
            base_code = self._create_base_template(project)

            # カテゴリ別にノードを整理
            nodes_by_category = {}
            node_imports = set()

            logger.info(
                f"DEBUG: Processing {len(nodes_data)} nodes for NEW ARCHITECTURE"
            )
            for i, node_data in enumerate(nodes_data):
                logger.info(f"DEBUG: Node {i+1}: {node_data}")

                # 一時的なFlowNodeオブジェクトを作成（DBに保存しない）
                temp_node = type(
                    "TempNode",
                    (),
                    {
                        "id": node_data.get("id", ""),
                        "data": node_data.get("data", {}),
                        "position_x": node_data.get("position", {}).get("x", 0),
                        "position_y": node_data.get("position", {}).get("y", 0),
                        "node_type": node_data.get(
                            "type", "default"
                        ),  # typeフィールドを渡す
                    },
                )()

                # ノードのコードブロックを生成
                code_block = self._generate_node_code_block(temp_node)
                logger.info(f"DEBUG: Generated code block: '{code_block}'")

                if code_block and code_block.strip():
                    # categoryの取得方法を修正
                    category = (
                        temp_node.data.get("nodeType", "")
                        or temp_node.node_type
                        or temp_node.data.get("category", "")
                    ).lower()

                    if category not in nodes_by_category:
                        nodes_by_category[category] = []
                    nodes_by_category[category].append(
                        {"node": temp_node, "code_block": code_block}
                    )
                    logger.info(f"DEBUG: Added to {category} category")

                # 必要なインポートを収集（動的生成）
                label = temp_node.data.get("label", "").strip()
                if label:
                    import_statement = self._generate_import_statement(label)
                    if import_statement:
                        node_imports.add(import_statement)
                        logger.info(f"DEBUG: Added import: {import_statement}")

            # インポート文を追加
            updated_code = base_code
            logger.info(f"DEBUG: Adding {len(node_imports)} imports")
            for import_line in node_imports:
                if import_line not in updated_code:
                    # WorkflowBuilderインポートの後に追加
                    match = self.patterns["workflow_builder_import"].search(
                        updated_code
                    )
                    if match:
                        updated_code = updated_code.replace(
                            match.group(0), f"{match.group(0)}\n{import_line}"
                        )
                        logger.info(f"DEBUG: Added import: {import_line}")

            # カテゴリ別にコードブロックをセクションに挿入
            logger.info(f"DEBUG: Categories found: {list(nodes_by_category.keys())}")

            for category, node_list in nodes_by_category.items():
                section_name = self._get_section_name_from_category(category)
                logger.info(
                    f"DEBUG: Inserting {len(node_list)} nodes into '{section_name}' section"
                )

                # セクションを検出
                section_pattern = re.compile(
                    rf"^(\s*)# {re.escape(section_name)} field\s*$", re.MULTILINE
                )
                match = section_pattern.search(updated_code)

                if match:
                    insertion_point = match.end()
                    logger.info(
                        f"DEBUG: Found '{section_name}' section at position {insertion_point}"
                    )

                    # セクションのコードブロックを結合
                    section_code_blocks = [
                        node_info["code_block"] for node_info in node_list
                    ]
                    section_code = "\n".join(section_code_blocks)

                    # 挿入
                    before_section = updated_code[:insertion_point]
                    after_section = updated_code[insertion_point:]
                    updated_code = f"{before_section}\n{section_code}\n{after_section}"
                    logger.info(
                        f"DEBUG: Inserted {len(section_code_blocks)} code blocks into '{section_name}' section"
                    )
                else:
                    logger.error(f"DEBUG: Could not find '{section_name}' section")

            # Workflowコマンドを生成
            logger.info(f"DEBUG: Building workflow commands")
            workflow_commands = self._build_workflow_commands_from_json(
                nodes_data, edges_data
            )
            logger.info(f"DEBUG: Generated {len(workflow_commands)} workflow commands")
            for command in workflow_commands:
                logger.info(f"DEBUG: Command: {command}")

            # Create workflow fieldセクションにコマンドを挿入
            workflow_section_pattern = re.compile(
                r'^(\s*)workflow = WorkflowBuilder\("neural_simulation"\)\s*$',
                re.MULTILINE,
            )
            match = workflow_section_pattern.search(updated_code)

            if match:
                insertion_point = match.end()
                logger.info(
                    f"DEBUG: Found WorkflowBuilder declaration at position {insertion_point}"
                )

                if workflow_commands:
                    # コマンドを挿入
                    before_commands = updated_code[:insertion_point]
                    after_commands = updated_code[insertion_point:]
                    commands_text = "\n" + "\n".join(workflow_commands) + "\n"
                    updated_code = before_commands + commands_text + after_commands
                    logger.info(
                        f"DEBUG: Inserted {len(workflow_commands)} workflow commands"
                    )
                else:
                    logger.info(f"DEBUG: No workflow commands to insert")
            else:
                logger.error(f"DEBUG: Could not find WorkflowBuilder declaration")

            # ファイルに保存
            code_file = self.get_code_file_path(project_id)
            code_file.parent.mkdir(parents=True, exist_ok=True)

            with open(code_file, "w", encoding="utf-8") as f:
                f.write(updated_code)

            logger.info(f"DEBUG: Final generated code:\n{updated_code}")
            logger.info(f"Successfully saved generated code to: {code_file}")

            # Jupyter notebookに変換
            notebook_success = self._convert_py_to_ipynb(project_id)
            if notebook_success:
                logger.info("Successfully converted to Jupyter notebook")
            else:
                logger.warning("Failed to convert to Jupyter notebook")

            logger.info("=== Batch code generation completed successfully ===")
            return True

        except Exception as e:
            logger.error(f"=== Critical error in batch code generation: {e} ===")
            logger.error(traceback.format_exc())
            return False
