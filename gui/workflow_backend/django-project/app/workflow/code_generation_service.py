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
        self.code_dir = Path(settings.BASE_DIR) / "codes/projects"
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

    def get_code_file_path(self, project_name):
        """プロジェクトIDからコードファイルパスを取得"""
        return self.code_dir / str(project_name) / f"{project_name}.py"

    def get_notebook_file_path(self, project_name):
        """プロジェクトIDからnotebookファイルパスを取得"""
        return self.code_dir / str(project_name) / f"{project_name}.ipynb"

    def _convert_py_to_ipynb(self, project_id):
        """Pythonファイルをjupyter notebookに変換"""
        try:
            # Idからプロジェクトを取得
            project = FlowProject.objects.get(id=project_id)
            # 補正したプロジェクト名
            project_name = project.name.replace(" ","").capitalize()
            # ファイルパス取得
            code_file = self.get_code_file_path(project_name)
            notebook_file = self.get_notebook_file_path(project_name)

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
sys.path.append('../nodes')

from neuroworkflow import WorkflowBuilder

def main():
    """Run a simple neural simulation workflow."""
    
    # Create nodes
    
    # I/O field
    
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

            # 全てのノードを nodes からimport（動的生成）
            # nodes/{ClassName}.py から {ClassName} をimport
            return f"from nodes.{class_name} import {class_name}"

        except Exception as e:
            logger.error(f"Error generating import statement for {class_name}: {e}")
            return None

    def _generate_node_code_block(self, node, node_no):
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

        # 全てのカテゴリーでコード生成を行う
        category_lower = category.lower()
        logger.info(
            f"DEBUG: Generating code for node {node_id} - category '{category}' (normalized: '{category_lower}')"
        )

        var_name = self._generate_variable_name_by_category(
            label, node_id, category_lower, node_no
        )
        #constructor_arg = self._generate_constructor_arg_by_category(
        #    label, category_lower
        #)
        configure_block = self._generate_configure_block_by_category(
            label, category_lower, node.data
        )

        #code_block = f"""    {var_name} = {label}("{constructor_arg}")"""
        code_block = f"""    {var_name} = {label}("{var_name}")"""

        if configure_block:
            code_block += f"""
    {var_name}.configure(
{configure_block}
    )"""
        else:
            code_block += f"""
    {var_name}.configure()"""

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

    def _generate_variable_name_by_category(self, class_name, node_id, category, node_no):
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

        node_no_zero = str(node_no).zfill(3)

        return f"instance_{class_name}_{node_no_zero}"
        
        """
        if category == "network":
            return f"network_{short_id}"
        elif category == "simulation":
            return f"sim_{short_id}"
        elif category == "analysis":
            return f"analysis_{short_id}"
        else:
            # その他のカテゴリー
            return f"{category}_{short_id}"
        """
        
    """ コンストラクタにNodeを渡している（不使用？）"""
    def _generate_constructor_arg_by_category(self, class_name, category):
        """categoryベースでコンストラクタ引数を生成"""
        if category == "network":
            return "SonataNetworkBuilder"
        elif category == "simulation":
            return "SonataNetworkSimulation"
        elif category == "analysis":
            return "SpikeAnalyzer"
        else:
            # その他のカテゴリの場合は、カテゴリー名をそのまま使用
            return category.capitalize()

    def _generate_configure_block_by_category(self, class_name, category, node_data):
        """categoryベースでconfigureブロックを生成（パラメーター変更含む）"""
        if category == "network":
            return self._generate_network_configure_block(class_name, node_data)
        elif category == "simulation":
            return self._generate_simulation_configure_block(class_name, node_data)
        elif category == "analysis":
            return self._generate_analysis_configure_block(class_name, node_data)
        else:
            return self._generate_generic_configure_block(class_name, node_data)

    def _generate_network_configure_block(self, class_name, node_data):
        """networkカテゴリ用のconfigureブロックを生成（変更されたパラメーターのみ）"""
        # デフォルト設定は不要、変更されたパラメーターのみを取得
        modified_params = self._get_modified_parameters_for_configure(node_data, {})

        # configureブロックをフォーマット
        config_lines = []
        for key, value in modified_params.items():
            if isinstance(value, str):
                config_lines.append(f'            {key}="{value}"')
            else:
                config_lines.append(f"            {key}={value}")

        return ",\n".join(config_lines)

    def _generate_simulation_configure_block(self, class_name, node_data):
        """simulationカテゴリ用のconfigureブロックを生成（変更されたパラメーターのみ）"""
        # デフォルト設定は不要、変更されたパラメーターのみを取得
        modified_params = self._get_modified_parameters_for_configure(node_data, {})

        # configureブロックをフォーマット
        config_lines = []
        for key, value in modified_params.items():
            if isinstance(value, str):
                config_lines.append(f'            {key}="{value}"')
            else:
                config_lines.append(f"            {key}={value}")

        return ",\n".join(config_lines)

    def _generate_analysis_configure_block(self, class_name, node_data):
        """analysisカテゴリ用のconfigureブロックを生成（変更されたパラメーターのみ）"""
        # 変更されたパラメーターのみを取得
        modified_params = self._get_modified_parameters_for_configure(node_data, {})

        # configureブロックをフォーマット
        config_lines = []
        for key, value in modified_params.items():
            if isinstance(value, str):
                config_lines.append(f'            {key}="{value}"')
            else:
                config_lines.append(f"            {key}={value}")

        return ",\n".join(config_lines)

    def _generate_generic_configure_block(self, class_name, node_data):
        """その他のカテゴリー用のconfigureブロックを生成（変更されたパラメーターのみ）"""
        # 変更されたパラメーターのみを取得
        modified_params = self._get_modified_parameters_for_configure(node_data, {})

        # configureブロックをフォーマット
        config_lines = []
        for key, value in modified_params.items():
            if isinstance(value, str):
                config_lines.append(f'            {key}="{value}"')
            else:
                config_lines.append(f"            {key}={value}")

        return ",\n".join(config_lines)

    def _get_modified_parameters_for_configure(self, node_data, default_params):
        """
        ノードのパラメーター変更情報を取得し、変更されたもののみを返す
        カテゴリ別に変更されたパラメーターを適切なconfigure設定に変換
        """
        # 変更されたパラメーターのみを格納する辞書
        modified_params_only = {}

        # schema.parametersから現在の値を取得
        schema = node_data.get("schema", {})
        parameters = schema.get("parameters", {})

        logger.info(f"DEBUG: Processing parameters for configure block: {parameters}")

        # parameter_modificationsから変更情報を取得
        modifications = node_data.get("parameter_modifications", {})
        logger.info(f"DEBUG: parameter_modifications: {modifications}")

        # 変更されたパラメーターのみを処理
        for param_key, param_info in parameters.items():
            current_value = param_info.get("default_value")  # default_valueから現在の値を取得
            logger.info(f"DEBUG: Processing parameter '{param_key}' with value '{current_value}'")

            # configure設定にマッピングされるパラメーター名を取得
            config_key = self._map_parameter_to_config_key(param_key)
            logger.info(f"DEBUG: Parameter '{param_key}' mapped to config_key '{config_key}'")

            # default_valueの変更があった時のみチェック
            if param_key in modifications:
                modification_info = modifications[param_key]
                # 新しいデータ構造に対応
                original_value = modification_info.get("field_modifications", {}).get("default_value_original", "")
                is_modified = modification_info.get("is_modified", False)
                # default_valueが元の値と異なるかチェック
                has_default_value_changed = (current_value != original_value) and is_modified
                logger.info(f"DEBUG: Parameter '{param_key}' default_value changed: {original_value} -> {current_value} (changed: {has_default_value_changed}, is_modified: {is_modified})")
            else:
                has_default_value_changed = False
                logger.info(f"DEBUG: Parameter '{param_key}' not found in modifications")

            # default_valueの変更があったパラメーターのみを追加
            if (param_key in modifications and
                has_default_value_changed and
                config_key and current_value is not None):

                # 型変換を行う
                converted_value = self._convert_parameter_value(current_value, config_key)
                modified_params_only[config_key] = converted_value

                modification_info = modifications[param_key]
                original_value = modification_info.get("field_modifications", {}).get("default_value_original", "")
                logger.info(f"DEBUG: Added parameter with default_value change '{param_key}' -> '{config_key}': {original_value} -> {current_value}")

        # parameter_modificationsの全ての項目を確実に処理するための追加チェック
        # より詳細なログ出力で3つ目以降の認識問題を解決
        logger.info(f"DEBUG: Starting additional check for all modifications: {len(modifications)} items")
        for i, (param_key, modification_info) in enumerate(modifications.items()):
            logger.info(f"DEBUG: Processing modification {i+1}/{len(modifications)}: {param_key}")

            if param_key not in parameters:
                logger.warning(f"DEBUG: Parameter '{param_key}' found in modifications but not in schema.parameters")
                continue

            param_info = parameters[param_key]
            current_value = param_info.get("default_value")
            # 新しいデータ構造に対応
            original_value = modification_info.get("field_modifications", {}).get("default_value_original", "")
            is_modified = modification_info.get("is_modified", False)

            # configure設定にマッピングされるパラメーター名を取得
            config_key = self._map_parameter_to_config_key(param_key)
            logger.info(f"DEBUG: Modification {i+1} - '{param_key}' -> '{config_key}': {original_value} -> {current_value} (is_modified: {is_modified})")

            # より寛容な変更判定（型変換後に比較）
            current_converted = self._convert_parameter_value(current_value, config_key)
            original_converted = self._convert_parameter_value(original_value, config_key)

            has_change = (current_converted != original_converted) and is_modified
            already_processed = (config_key in modified_params_only)

            logger.info(f"DEBUG: Modification {i+1} - has_change: {has_change}, already_processed: {already_processed}")

            # まだ処理されていないパラメーターで変更があるものを追加
            if (config_key not in modified_params_only and
                has_change and
                config_key and current_value is not None):

                # 型変換を行う
                converted_value = self._convert_parameter_value(current_value, config_key)
                modified_params_only[config_key] = converted_value

                logger.info(f"DEBUG: Added modification {i+1} '{param_key}' -> '{config_key}': {original_value} -> {current_value}")
            else:
                logger.info(f"DEBUG: Skipped modification {i+1} '{param_key}' (already processed or no change)")

        logger.info(f"DEBUG: Found {len(modified_params_only)} modified parameters for configure block")
        return modified_params_only

    def _map_parameter_to_config_key(self, parameter_key):
        """
        パラメーター名をそのまま使用（マッピング不要）
        """
        # パラメーター名をそのまま返す
        return parameter_key

    def _convert_parameter_value(self, value, config_key):
        """
        パラメーター値を適切な型に変換（配列と数値対応）
        """
        # 配列の処理
        if isinstance(value, list):
            # リストの場合はPythonの配列表記に変換
            return value

        # JSONから文字列として来た配列の処理
        if isinstance(value, str) and value.strip().startswith('[') and value.strip().endswith(']'):
            try:
                import json
                parsed_value = json.loads(value)
                if isinstance(parsed_value, list):
                    return parsed_value
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Could not parse array string '{value}' for {config_key}, keeping as string")
                return value

        # 数値系のパラメーター
        numeric_keys = ["simulation_time", "record_n_neurons", "hdf5_hyperslab_size"]

        # 数値かどうかを自動判定
        if config_key in numeric_keys or self._is_numeric_value(value):
            try:
                # floatまたはintに変換
                if config_key == "simulation_time" or '.' in str(value):
                    return float(value)
                else:
                    return int(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert '{value}' to number for {config_key}, keeping as string")
                return value

        # 文字列系のパラメーター（そのまま返す）
        return value

    def _is_numeric_value(self, value):
        """値が数値かどうかを判定"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _extract_handle_name(self, handle_string):
        """
        ハンドル文字列から必要な部分を抽出
        例: calc_1757868335061_i6nyja6de-sonata_net-output-object -> sonata_net
        """
        if not handle_string:
            return ""

        try:
            # '-' で分割して2番目の部分を取得
            parts = handle_string.split('-')
            if len(parts) >= 2:
                return parts[1]  # sonata_net 部分
            else:
                # 分割できない場合はそのまま返す
                return handle_string
        except Exception as e:
            logger.warning(f"Could not extract handle name from '{handle_string}': {e}")
            return handle_string

    def _get_section_name_from_category(self, category):
        """カテゴリからセクション名を取得"""
        category_to_section = {
            "analysis": "Analysis",
            "io": "I/O",
            "network": "Network",
            "optimization": "Optimization",
            "simulation": "Simulation",
            "stimulus": "Stimulus",
        }
        return category_to_section.get(category.lower(), "Analysis")

    def _get_builder_name_from_label(self, label, node_id, category=None):
        """ラベルとカテゴリからBuilderでの名前を取得（コンストラクタ引数と同じルール）"""
        if category == "network":
            return "SonataNetworkBuilder"
        elif category == "simulation":
            return "SonataNetworkSimulation"
        elif category == "analysis":
            return "SpikeAnalyzer"
        else:
            # その他のカテゴリの場合は、カテゴリー名をそのまま使用
            return category.capitalize()

    def _build_workflow_commands_from_json(self, nodes_data, edges_data):
        """ノードとエッジ情報からワークフローコマンドを生成"""
        commands = []

        # ノードIDから変数名とBuilderNameへのマッピングを作成
        node_id_to_var = {}
        node_id_to_builder = {}

        # ノード番号採番
        node_no = 1

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

            # 全てのカテゴリーでvariable_nameとbuilder_nameを生成
            var_name = self._generate_variable_name_by_category(
                label, node_id, category, node_no
            )
            """
            builder_name = self._get_builder_name_from_label(
                label, node_id, category
            )
            """
            builder_name = var_name

            node_id_to_var[node_id] = var_name
            node_id_to_builder[node_id] = builder_name
            node_no += 1

        # add_nodeコマンドを生成（全ノード）
        for node_id in node_id_to_var:
            var_name = node_id_to_var[node_id]
            commands.append(f"    workflow.add_node({var_name})")

        # connectコマンドを生成（エッジごと）
        for edge_data in edges_data:
            source_id = edge_data.get("source", "")
            target_id = edge_data.get("target", "")

            # sourceHandleとtargetHandleからポート情報を取得
            source_handle_raw = edge_data.get("sourceHandle", "")
            target_handle_raw = edge_data.get("targetHandle", "")

            # ハンドル名から必要な部分を抽出 (例: calc_xxx-sonata_net-output-object -> sonata_net)
            source_handle = self._extract_handle_name(source_handle_raw)
            target_handle = self._extract_handle_name(target_handle_raw)

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

    def generate_code_from_flow_data(self, project_id, project_name, nodes_data, edges_data):
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

            # ノード番号を採番
            node_no = 1
            for i, node_data in enumerate(nodes_data):
                logger.info(f"DEBUG: Node {i+1}: {node_data}")

                # 実際のDBからノード情報を取得してパラメーター変更情報を含める
                node_id = node_data.get("id", "")
                try:
                    # DBから実際のノードを取得（パラメーター変更情報含む）
                    db_node = FlowNode.objects.get(id=node_id, project_id=project_id)
                    # DBのdataにパラメーター変更情報が含まれているのでそれを使用
                    enhanced_node_data = db_node.data.copy()
                    # 位置情報などはJSONから取得
                    enhanced_node_data.update(node_data.get("data", {}))

                    logger.info(f"DEBUG: Enhanced node data with parameter modifications: {enhanced_node_data}")
                except FlowNode.DoesNotExist:
                    logger.warning(f"Node {node_id} not found in DB, using JSON data only")
                    enhanced_node_data = node_data.get("data", {})

                # 一時的なFlowNodeオブジェクトを作成（パラメーター変更情報含む）
                temp_node = type(
                    "TempNode",
                    (),
                    {
                        "id": node_id,
                        "data": enhanced_node_data,
                        "position_x": node_data.get("position", {}).get("x", 0),
                        "position_y": node_data.get("position", {}).get("y", 0),
                        "node_type": node_data.get(
                            "type", "default"
                        ),  # typeフィールドを渡す
                    },
                )()

                # ノードのコードブロックを生成
                code_block = self._generate_node_code_block(temp_node, node_no)                
                logger.info(f"DEBUG: Generated code block: '{code_block}'")
                # ノード番号カウント
                node_no += 1

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
                    #rf"^(\s*)# {re.escape(section_name)}\s*$", re.MULTILINE
                )
                match = section_pattern.search(updated_code)

                if match:
                    insertion_point = match.end()
                    logger.info(
                        f"DEBUG: Found '{section_name}' section at position {insertion_point}"
                    )

                    # セクションの既存コードを削除して新しいコードに置換
                    # 次のセクションまたはCreate workflow fieldまでを検索
                    next_section_pattern = re.compile(
                        r'^(\s*)# (Analysis|IO|Network|Optimization|Simulation|Stimulus|Create workflow) field\s*$',
                        re.MULTILINE
                    )

                    # insertion_point以降で次のセクションを検索
                    remaining_code = updated_code[insertion_point:]
                    next_match = next_section_pattern.search(remaining_code)

                    if next_match:
                        # 次のセクションが見つかった場合、そこまでを置換
                        section_end = insertion_point + next_match.start()
                    else:
                        # 次のセクションが見つからない場合、ファイル終端まで
                        section_end = len(updated_code)

                    # セクションのコードブロックを結合
                    section_code_blocks = [
                        node_info["code_block"] for node_info in node_list
                    ]
                    section_code = "\n".join(section_code_blocks)

                    # セクション内容を置換（既存コードを削除して新しいコードを挿入）
                    before_section = updated_code[:insertion_point]
                    after_section = updated_code[section_end:]
                    updated_code = f"{before_section}\n{section_code}\n{after_section}"
                    logger.info(
                        f"DEBUG: Replaced section content with {len(section_code_blocks)} code blocks in '{section_name}' section"
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
            code_file = self.get_code_file_path(project_name)
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
