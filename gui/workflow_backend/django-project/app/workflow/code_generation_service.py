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
            # WorkflowBuilder全体を検出（インデントを考慮した閉じ括弧まで）
            "workflow_section": re.compile(
                r"(\s*)workflow\s*=\s*\(.*?\n\1\)", re.DOTALL | re.MULTILINE
            ),
            # ノード定義を検出（configure呼び出しも含む）
            "node_definition": re.compile(
                r"^(\s*)({var_name})\s*=\s*\w+Node\([^)]*\)(?:\s*\n\s*\2\.configure\([^)]*\))?",
                re.MULTILINE | re.DOTALL,
            ),
            # インポート文を検出
            "import_statement": re.compile(
                r"^from\s+neuroworkflow\.nodes\.\w+\s+import\s+(\w+)$", re.MULTILINE
            ),
            # WorkflowBuilderインポートを検出
            "workflow_builder_import": re.compile(
                r"^(from\s+neuroworkflow\s+import\s+WorkflowBuilder)$", re.MULTILINE
            ),
            # クラス使用を検出
            "class_usage": {
                "BuildSonataNetworkNode": re.compile(r"BuildSonataNetworkNode\s*\("),
                "SimulateSonataNetworkNode": re.compile(
                    r"SimulateSonataNetworkNode\s*\("
                ),
            },
        }

    def get_code_file_path(self, project_id):
        """プロジェクトIDからコードファイルパスを取得"""
        return self.code_dir / str(project_id) / f"{project_id}.py"

    def get_notebook_file_path(self, project_id):
        """プロジェクトIDからnotebookファイルパスを取得"""
        return self.code_dir / str(project_id) / f"{project_id}.ipynb"

    def add_node_code(self, project_id, node):
        """ノード追加: インポート + コードブロック + WorkflowBuilder更新 + .ipynb変換を一括処理"""
        try:
            logger.info(
                f"=== Starting add_node_code for node {node.id} in project {project_id} ==="
            )
            logger.info(f"Node label: {node.data.get('label', 'Unknown')}")

            code_file = self.get_code_file_path(project_id)
            code_file.parent.mkdir(parents=True, exist_ok=True)

            # 既存コードの読み込みまたは新規作成
            if not code_file.exists():
                project = FlowProject.objects.get(id=project_id)
                existing_code = self._create_base_template(project)
                logger.info("Created new base template")
            else:
                with open(code_file, "r", encoding="utf-8") as f:
                    existing_code = f.read()
                logger.info("Loaded existing code file")

            # 1. インポート文を追加
            logger.info("Step 1: Adding imports")
            updated_code, import_success = self._add_import_for_node(
                existing_code, node
            )
            if not import_success:
                logger.warning(f"Failed to add imports for node {node.id}")

            # 2. ノードのコードブロックを追加
            logger.info("Step 2: Adding node code block")
            new_code_block = self._generate_node_code_block(node)
            logger.info(f"Generated code block:\n{new_code_block}")

            updated_code, insert_success = self._insert_node_code_block(
                updated_code, new_code_block, node.id
            )
            if not insert_success:
                logger.error(f"Failed to insert code block for node {node.id}")
                return False

            # 3. WorkflowBuilderチェーンを更新（これが重要！）
            logger.info("Step 3: Updating WorkflowBuilder chain")
            updated_code, chain_success = self._update_workflow_chain(
                updated_code, project_id
            )
            if not chain_success:
                logger.error(f"Failed to update workflow chain for node {node.id}")
                # それでも保存はする

            # ファイルに保存
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(updated_code)

            # 4. .ipynbファイルに変換
            logger.info("Step 4: Converting to Jupyter notebook")
            notebook_success = self._convert_py_to_ipynb(project_id)
            if not notebook_success:
                logger.warning("Failed to convert to notebook, but .py file was saved")

            logger.info(
                f"=== Successfully completed add_node_code for node {node.id} ==="
            )
            return True

        except Exception as e:
            logger.error(
                f"=== Critical error in add_node_code for node {node.id}: {e} ==="
            )
            logger.error(traceback.format_exc())
            return False

    def remove_node_code(self, project_id, node_id):
        """ノード削除: コードブロック削除 + インポート整理 + WorkflowBuilder更新 + .ipynb変換を一括処理"""
        try:
            logger.info(
                f"=== Starting remove_node_code for node {node_id} in project {project_id} ==="
            )

            code_file = self.get_code_file_path(project_id)

            if not code_file.exists():
                logger.info("Code file not found, returning success")
                return True

            with open(code_file, "r", encoding="utf-8") as f:
                existing_code = f.read()

            # 1. ノードのコードブロックを削除（修正版）
            logger.info("Step 1: Removing node code block")
            updated_code, remove_success = self._remove_node_code_block(
                existing_code, node_id
            )
            if not remove_success:
                logger.warning(
                    f"Node code block not found for {node_id}, continuing..."
                )

            # 2. 不要なインポート文を削除
            logger.info("Step 2: Cleaning up unused imports")
            updated_code, cleanup_success = self._cleanup_unused_imports(
                updated_code, project_id
            )
            if not cleanup_success:
                logger.warning("Import cleanup had issues, continuing...")

            # 3. WorkflowBuilderチェーンを更新（これが重要！）
            logger.info("Step 3: Updating WorkflowBuilder chain")
            updated_code, chain_success = self._update_workflow_chain(
                updated_code, project_id
            )
            if not chain_success:
                logger.error(
                    f"Failed to update workflow chain after removing node {node_id}"
                )

            # ファイルに保存
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(updated_code)

            # 4. .ipynbファイルに変換
            logger.info("Step 4: Converting to Jupyter notebook")
            notebook_success = self._convert_py_to_ipynb(project_id)
            if not notebook_success:
                logger.warning("Failed to convert to notebook, but .py file was saved")

            logger.info(
                f"=== Successfully completed remove_node_code for node {node_id} ==="
            )
            return True

        except Exception as e:
            logger.error(
                f"=== Critical error in remove_node_code for node {node_id}: {e} ==="
            )
            logger.error(traceback.format_exc())
            return False

    def update_workflow_builder(self, project_id):
        """エッジ追加/削除時のWorkflowBuilderチェーンのみ更新 + .ipynb変換"""
        try:
            logger.info(f"=== Updating workflow builder for project {project_id} ===")

            code_file = self.get_code_file_path(project_id)

            if not code_file.exists():
                logger.warning(f"Code file does not exist for project {project_id}")
                return True

            with open(code_file, "r", encoding="utf-8") as f:
                existing_code = f.read()

            # WorkflowBuilderチェーンのみ更新
            updated_code, success = self._update_workflow_chain(
                existing_code, project_id
            )

            if not success:
                logger.error(
                    f"Failed to update WorkflowBuilder for project {project_id}"
                )
                return False

            # ファイルに保存
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(updated_code)

            # .ipynbファイルに変換
            logger.info("Converting to Jupyter notebook after workflow update")
            notebook_success = self._convert_py_to_ipynb(project_id)
            if not notebook_success:
                logger.warning("Failed to convert to notebook, but .py file was saved")

            logger.info(
                f"Successfully updated WorkflowBuilder for project {project_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Critical error updating WorkflowBuilder for project {project_id}: {e}"
            )
            logger.error(traceback.format_exc())
            return False

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

    # 以下、既存のメソッドはそのまま...
    def _create_base_template(self, project):
        """基本テンプレートを作成（JupyterLab用パス設定）"""
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

    workflow = (
        WorkflowBuilder("neural_simulation")
            .build()
    )

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

    def _add_import_for_node(self, existing_code, node):
        """ノードの種類に応じてインポート文を動的に追加"""
        try:
            label = node.data.get("label", "").strip()
            logger.info(f"Adding imports for node with label: {label}")

            if not label:
                logger.info("No label provided for node")
                return existing_code, True

            # 動的にimport文を生成
            import_line = self._generate_import_statement(label)
            if not import_line:
                logger.warning(f"Could not generate import for label: {label}")
                return existing_code, True

            # 既に存在しない場合のみ追加
            if import_line in existing_code:
                logger.info(f"Import already exists: {import_line}")
                return existing_code, True

            # WorkflowBuilderインポートの位置を検出
            match = self.patterns["workflow_builder_import"].search(existing_code)
            if not match:
                logger.error("WorkflowBuilder import not found!")
                return existing_code, False

            # WorkflowBuilderインポートの直後に追加
            existing_code = existing_code.replace(
                match.group(0), f"{match.group(0)}\n{import_line}"
            )
            logger.info(f"Added import: {import_line}")

            return existing_code, True

        except Exception as e:
            logger.error(f"Error adding imports: {e}")
            return existing_code, False

    def _generate_import_statement(self, class_name):
        """クラス名から動的にimport文を生成"""
        try:
            # クラス名のバリデーション
            if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', class_name):
                logger.warning(f"Invalid class name format: {class_name}")
                return None

            # neuroworkflowの既知のクラスかチェック
            known_neuroworkflow_classes = {
                'BuildSonataNetworkNode': 'from neuroworkflow.nodes.network import BuildSonataNetworkNode',
                'SimulateSonataNetworkNode': 'from neuroworkflow.nodes.simulation import SimulateSonataNetworkNode',
            }
            
            if class_name in known_neuroworkflow_classes:
                return known_neuroworkflow_classes[class_name]
            
            # カスタムノードとして upload_nodes からimport
            # upload_nodes/{ClassName}.py から {ClassName} をimport
            return f"from upload_nodes.{class_name} import {class_name}"
            
        except Exception as e:
            logger.error(f"Error generating import statement for {class_name}: {e}")
            return None

    def _cleanup_unused_imports(self, existing_code, project_id):
        """使用されなくなったインポート文を削除（正規表現版）"""
        try:
            # 使用されているクラスを検出
            used_classes = set()

            for class_name, pattern in self.patterns["class_usage"].items():
                if pattern.search(existing_code):
                    used_classes.add(class_name)
                    logger.info(f"Class {class_name} is still in use")

            # インポート文を処理
            lines = existing_code.split("\n")
            updated_lines = []
            removed_imports = []

            for line in lines:
                should_keep = True

                # インポート文をチェック
                if "from neuroworkflow.nodes" in line:
                    match = self.patterns["import_statement"].match(line)
                    if match:
                        class_name = match.group(1)
                        if class_name not in used_classes:
                            should_keep = False
                            removed_imports.append(class_name)
                            logger.info(f"Removing unused import: {class_name}")

                if should_keep:
                    updated_lines.append(line)

            if removed_imports:
                logger.info(f"Removed imports for: {removed_imports}")
            else:
                logger.info("No unused imports to remove")

            return "\n".join(updated_lines), True

        except Exception as e:
            logger.error(f"Error cleaning imports: {e}")
            return existing_code, False

    def _generate_node_code_block(self, node):
        """ノードのコードブロックを動的に生成"""
        label = node.data.get("label", "").strip()
        node_id = node.id

        if not label:
            var_name = self._sanitize_variable_name(node_id, "node")
            return f"""    # Node with no label (ID: {node_id})
    {var_name} = None  # TODO: Add implementation"""

        # 動的に変数名とコンストラクタ引数を生成
        var_name = self._generate_variable_name(label, node_id)
        constructor_arg = self._generate_constructor_arg(label)
        configure_block = self._generate_configure_block(label)

        # コードブロック生成
        code_block = f"""    {var_name} = {label}("{constructor_arg}")"""
        
        if configure_block:
            code_block += f"""
    {var_name}.configure({configure_block})"""
            
        return code_block

    def _generate_variable_name(self, class_name, node_id):
        """クラス名とノードIDから変数名を生成"""
        # クラス名からプレフィックスを推測
        if "BuildSonataNetworkNode" in class_name:
            prefix = "build_network"
        elif "SimulateSonataNetworkNode" in class_name:
            prefix = "simulate_network"
        elif "Network" in class_name:
            prefix = "network"
        elif "Analysis" in class_name:
            prefix = "analysis"
        elif "Simulation" in class_name:
            prefix = "simulation"
        else:
            # クラス名をsnake_caseに変換
            prefix = re.sub('([A-Z]+)', r'_\1', class_name).lower().strip('_')
            
        return self._sanitize_variable_name(node_id, prefix)

    def _generate_constructor_arg(self, class_name):
        """クラス名からコンストラクタ引数を生成"""
        # 既知のクラスのマッピング
        known_constructor_args = {
            'BuildSonataNetworkNode': 'SonataNetworkBuilder',
            'SimulateSonataNetworkNode': 'SonataNetworkSimulation',
        }
        
        if class_name in known_constructor_args:
            return known_constructor_args[class_name]
        
        # デフォルトはクラス名からNodeを除去
        if class_name.endswith('Node'):
            return class_name[:-4]
        return class_name

    def _generate_configure_block(self, class_name):
        """クラス名からconfigure引数を生成"""
        # 既知のクラスの設定
        known_configurations = {
            'BuildSonataNetworkNode': """
        sonata_path="../data/300_pointneurons",
        net_config_file="circuit_config.json",
        sim_config_file="simulation_config.json",
        hdf5_hyperslab_size=1024""",
            'SimulateSonataNetworkNode': """
        simulation_time=1000.0,
        record_from_population="internal",
        record_n_neurons=40""",
        }
        
        return known_configurations.get(class_name, "")  # カスタムノードは設定なし

    def _sanitize_variable_name(self, node_id, prefix):
        """ノードIDを有効な変数名に変換"""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", str(node_id))
        if sanitized and sanitized[0].isdigit():
            sanitized = f"{prefix}_{sanitized}"
        elif not sanitized:
            sanitized = prefix
        return sanitized

    def _insert_node_code_block(self, existing_code, new_code_block, node_id):
        """main関数内にノードのコードブロックを挿入（正規表現版）"""
        try:
            # 既存の同じノードのコードブロックを削除
            code_without_existing, _ = self._remove_node_code_block(
                existing_code, node_id
            )

            # workflow = ( の位置を検出
            workflow_pattern = re.compile(r"^(\s*)workflow\s*=\s*\($", re.MULTILINE)
            match = workflow_pattern.search(code_without_existing)

            if not match:
                logger.error("Could not find 'workflow = (' pattern")
                return code_without_existing, False

            # workflow定義の前に新しいコードブロックを挿入
            insertion_point = match.start()

            # 適切な改行を追加
            before_workflow = code_without_existing[:insertion_point].rstrip()
            after_workflow = code_without_existing[insertion_point:]

            # 新しいコードを挿入
            updated_code = f"{before_workflow}\n\n{new_code_block}\n\n{after_workflow}"

            logger.info(f"Successfully inserted code block for node {node_id}")
            return updated_code, True

        except Exception as e:
            logger.error(f"Error inserting node code block: {e}")
            return existing_code, False

    def _remove_node_code_block(self, existing_code, node_id):
        """特定のノードのコードブロックを削除（修正版：configure呼び出しも含めて削除）"""
        try:
            # 削除対象の変数名パターンを生成
            possible_var_names = []
            for prefix in ["build_network", "simulate_network", "node"]:
                var_name = self._sanitize_variable_name(node_id, prefix)
                possible_var_names.append(var_name)

            logger.info(
                f"Attempting to remove code blocks for variables: {possible_var_names}"
            )

            found_any = False
            for var_name in possible_var_names:
                # ノード定義とconfigure呼び出しを一括で削除する正規表現
                # 変数定義から始まり、.configure()の閉じ括弧まで
                pattern = re.compile(
                    rf"^\s*{re.escape(var_name)}\s*=\s*[^(]+\([^)]*\)(?:\s*\n\s*{re.escape(var_name)}\.configure\([^)]*\))?",
                    re.MULTILINE | re.DOTALL,
                )

                matches = pattern.findall(existing_code)
                if matches:
                    for match in matches:
                        logger.info(f"Found and removing code block:\n{match}")
                    existing_code = pattern.sub("", existing_code)
                    found_any = True
                    logger.info(f"Removed code block for variable: {var_name}")

            # 余分な空行を削除（3行以上の連続空行を2行に）
            existing_code = re.sub(r"\n{3,}", "\n\n", existing_code)

            if not found_any:
                logger.warning(f"No code blocks found for node {node_id}")
                return existing_code, False

            return existing_code, True

        except Exception as e:
            logger.error(f"Error removing node code block: {e}")
            return existing_code, False

    def _update_workflow_chain(self, existing_code, project_id):
        """WorkflowBuilderチェーンを更新（完全置換版）"""
        try:
            project = FlowProject.objects.get(id=project_id)
            nodes = FlowNode.objects.filter(project=project)
            edges = FlowEdge.objects.filter(project=project)

            logger.info(
                f"Updating workflow chain: {nodes.count()} nodes, {edges.count()} edges"
            )

            # 新しいWorkflowBuilderチェーンを構築
            new_workflow_lines = self._build_workflow_chain_lines(nodes, edges)

            # 既存のWorkflowBuilder部分を検出
            # インデントレベルを考慮した正規表現
            match = self.patterns["workflow_section"].search(existing_code)

            if not match:
                # 別のパターンを試す（フォールバック）
                # workflow = ( から同じインデントレベルの ) まで
                fallback_pattern = re.compile(
                    r"(\s*)workflow\s*=\s*\(.*?\n\1\)", re.DOTALL
                )
                match = fallback_pattern.search(existing_code)

                if not match:
                    logger.error("Could not find existing workflow section!")
                    logger.error(f"First 500 chars of code:\n{existing_code[:500]}")
                    return existing_code, False

            # インデントを保持
            indent = match.group(1)

            # 新しいworkflow定義を構築（インデントを保持）
            new_workflow = f"{indent}workflow = (\n"
            for line in new_workflow_lines:
                new_workflow += f"{line}\n"
            new_workflow += f"{indent})"

            # マッチした部分全体を新しいworkflow定義で置換
            start_pos = match.start()
            end_pos = match.end()
            updated_code = (
                existing_code[:start_pos] + new_workflow + existing_code[end_pos:]
            )

            # 置換が成功したか確認
            if updated_code == existing_code:
                logger.error("Workflow section was not replaced!")
                return existing_code, False

            logger.info("Successfully updated workflow chain")
            logger.info(f"New workflow section:\n{new_workflow}")

            return updated_code, True

        except Exception as e:
            logger.error(f"Error updating workflow chain: {e}")
            logger.error(traceback.format_exc())
            return existing_code, False

    def _build_workflow_chain_lines(self, nodes, edges):
        """WorkflowBuilderチェーンの行を構築（修正版）"""
        chain_lines = []

        # WorkflowBuilderの開始
        chain_lines.append('        WorkflowBuilder("neural_simulation")')

        # 実際に存在するノードを追加
        existing_node_ids = set()
        for node in nodes:
            var_name = self._get_node_variable_name(node)
            chain_lines.append(f"            .add_node({var_name})")
            existing_node_ids.add(str(node.id))
            logger.info(f"Added node to chain: {var_name} (ID: {node.id})")

        # エッジを追加（両端のノードが存在する場合のみ）
        for edge in edges:
            # エッジの両端が実際に存在するノードか確認
            if (
                str(edge.source) not in existing_node_ids
                or str(edge.target) not in existing_node_ids
            ):
                logger.warning(
                    f"Skipping edge {edge.source} -> {edge.target}: node not found"
                )
                continue

            try:
                source_node = FlowNode.objects.get(id=edge.source)
                target_node = FlowNode.objects.get(id=edge.target)

                source_name = self._get_node_builder_name(edge.source)
                target_name = self._get_node_builder_name(edge.target)

                # ノードタイプによって接続方法を決定
                source_label = source_node.data.get("label", "")
                target_label = target_node.data.get("label", "")

                if "BuildSonataNetworkNode" in str(
                    source_label
                ) and "SimulateSonataNetworkNode" in str(target_label):
                    # SonataNetwork特有の接続
                    chain_lines.append(
                        f'            .connect("{source_name}", "sonata_net", "{target_name}", "sonata_net")'
                    )
                    chain_lines.append(
                        f'            .connect("{source_name}", "node_collections", "{target_name}", "node_collections")'
                    )
                    logger.info(
                        f"Added SonataNetwork connections: {source_name} -> {target_name}"
                    )
                else:
                    # 一般的な接続
                    source_output = (
                        edge.source_handle if edge.source_handle else "default_output"
                    )
                    target_input = (
                        edge.target_handle if edge.target_handle else "default_input"
                    )
                    chain_lines.append(
                        f'            .connect("{source_name}", "{source_output}", "{target_name}", "{target_input}")'
                    )
                    logger.info(
                        f"Added general connection: {source_name} -> {target_name}"
                    )

            except FlowNode.DoesNotExist:
                logger.warning(
                    f"Node not found for edge: {edge.source} -> {edge.target}"
                )
                continue
            except Exception as e:
                logger.error(f"Error processing edge {edge.id}: {e}")
                continue

        # 最後に.build()を追加
        chain_lines.append("            .build()")

        logger.info(f"Built workflow chain with {len(chain_lines)} lines")
        return chain_lines

    def _get_node_variable_name(self, node):
        """ノードから変数名を取得"""
        label = node.data.get("label", "")
        if "BuildSonataNetworkNode" in str(label):
            return self._sanitize_variable_name(node.id, "build_network")
        elif "SimulateSonataNetworkNode" in str(label):
            return self._sanitize_variable_name(node.id, "simulate_network")
        else:
            return self._sanitize_variable_name(node.id, "node")

    def _get_node_builder_name(self, node_id):
        """ノードIDからBuilderでの名前を取得"""
        try:
            node = FlowNode.objects.get(id=node_id)
            label = node.data.get("label", "")
            if "BuildSonataNetworkNode" in str(label):
                return "SonataNetworkBuilder"
            elif "SimulateSonataNetworkNode" in str(label):
                return "SonataNetworkSimulation"
            else:
                return f"Node_{node_id}"
        except FlowNode.DoesNotExist:
            logger.error(f"Node {node_id} not found in database")
            return f"Node_{node_id}"
        except Exception as e:
            logger.error(f"Error getting builder name for node {node_id}: {e}")
            return f"Node_{node_id}"

    def generate_code_from_flow_data(self, project_id, nodes_data, edges_data):
        """React Flow JSONデータから一括でコードを生成する新しいメソッド"""
        try:
            logger.info(f"=== Starting batch code generation from flow data for project {project_id} ===")
            logger.info(f"Processing {len(nodes_data)} nodes and {len(edges_data)} edges")
            
            # プロジェクトの基本テンプレートを作成
            project = FlowProject.objects.get(id=project_id)
            base_code = self._create_base_template(project)
            
            # ノードデータからコードブロックを生成
            node_code_blocks = []
            node_imports = set()
            
            for node_data in nodes_data:
                # 一時的なFlowNodeオブジェクトを作成（DBに保存しない）
                temp_node = type('TempNode', (), {
                    'id': node_data.get('id', ''),
                    'data': node_data.get('data', {}),
                    'position_x': node_data.get('position', {}).get('x', 0),
                    'position_y': node_data.get('position', {}).get('y', 0),
                    'node_type': node_data.get('type', 'default')
                })()
                
                # ノードのコードブロックを生成
                code_block = self._generate_node_code_block(temp_node)
                if code_block:
                    node_code_blocks.append(code_block)
                    
                # 必要なインポートを収集（動的生成）
                label = temp_node.data.get("label", "").strip()
                if label:
                    import_statement = self._generate_import_statement(label)
                    if import_statement:
                        node_imports.add(import_statement)
            
            # インポート文を追加
            updated_code = base_code
            for import_line in node_imports:
                if import_line not in updated_code:
                    # WorkflowBuilderインポートの後に追加
                    match = self.patterns["workflow_builder_import"].search(updated_code)
                    if match:
                        updated_code = updated_code.replace(
                            match.group(0), f"{match.group(0)}\n{import_line}"
                        )
            
            # ノードのコードブロックを挿入
            if node_code_blocks:
                # workflow = ( の位置を検出
                workflow_pattern = re.compile(r"^(\s*)workflow\s*=\s*\($", re.MULTILINE)
                match = workflow_pattern.search(updated_code)
                
                if match:
                    insertion_point = match.start()
                    before_workflow = updated_code[:insertion_point].rstrip()
                    after_workflow = updated_code[insertion_point:]
                    
                    # 全てのノードコードブロックを結合
                    all_node_code = "\n\n".join(node_code_blocks)
                    updated_code = f"{before_workflow}\n\n{all_node_code}\n\n{after_workflow}"
            
            # WorkflowBuilderチェーンを生成（データベースの代わりにJSONデータを使用）
            workflow_chain_lines = self._build_workflow_chain_from_json(nodes_data, edges_data)
            
            # WorkflowBuilderセクションを更新
            match = self.patterns["workflow_section"].search(updated_code)
            if match:
                indent = match.group(1)
                new_workflow = f"{indent}workflow = (\n"
                for line in workflow_chain_lines:
                    new_workflow += f"{line}\n"
                new_workflow += f"{indent})"
                
                # 既存のworkflowセクションを新しいもので置換
                updated_code = (
                    updated_code[:match.start()] + new_workflow + updated_code[match.end():]
                )
            
            # ファイルに保存
            code_file = self.get_code_file_path(project_id)
            code_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(updated_code)
            
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

    def _build_workflow_chain_from_json(self, nodes_data, edges_data):
        """JSONデータからWorkflowBuilderチェーンの行を構築"""
        chain_lines = []
        
        # WorkflowBuilderの開始
        chain_lines.append('        WorkflowBuilder("neural_simulation")')
        
        # ノードを追加
        node_vars = {}  # node_id -> variable_name のマッピング
        for node_data in nodes_data:
            node_id = node_data.get('id', '')
            label = node_data.get('data', {}).get('label', '')
            
            # 変数名を生成
            if "BuildSonataNetworkNode" in str(label):
                var_name = self._sanitize_variable_name(node_id, "build_network")
            elif "SimulateSonataNetworkNode" in str(label):
                var_name = self._sanitize_variable_name(node_id, "simulate_network")
            else:
                var_name = self._sanitize_variable_name(node_id, "node")
            
            node_vars[node_id] = var_name
            chain_lines.append(f"            .add_node({var_name})")
            logger.info(f"Added node to chain: {var_name} (ID: {node_id})")
        
        # エッジを追加
        for edge_data in edges_data:
            source_id = edge_data.get('source', '')
            target_id = edge_data.get('target', '')
            
            if source_id not in node_vars or target_id not in node_vars:
                logger.warning(f"Skipping edge {source_id} -> {target_id}: node not found")
                continue
            
            # ノードの種類を判定
            source_node_data = next((n for n in nodes_data if n.get('id') == source_id), None)
            target_node_data = next((n for n in nodes_data if n.get('id') == target_id), None)
            
            if not source_node_data or not target_node_data:
                continue
                
            source_label = source_node_data.get('data', {}).get('label', '')
            target_label = target_node_data.get('data', {}).get('label', '')
            
            source_name = self._get_builder_name_from_label(source_label, source_id)
            target_name = self._get_builder_name_from_label(target_label, target_id)
            
            if "BuildSonataNetworkNode" in str(source_label) and "SimulateSonataNetworkNode" in str(target_label):
                # SonataNetwork特有の接続
                chain_lines.append(
                    f'            .connect("{source_name}", "sonata_net", "{target_name}", "sonata_net")'
                )
                chain_lines.append(
                    f'            .connect("{source_name}", "node_collections", "{target_name}", "node_collections")'
                )
                logger.info(f"Added SonataNetwork connections: {source_name} -> {target_name}")
            else:
                # 一般的な接続
                source_output = edge_data.get('sourceHandle', 'default_output')
                target_input = edge_data.get('targetHandle', 'default_input')
                chain_lines.append(
                    f'            .connect("{source_name}", "{source_output}", "{target_name}", "{target_input}")'
                )
                logger.info(f"Added general connection: {source_name} -> {target_name}")
        
        # 最後に.build()を追加
        chain_lines.append("            .build()")
        
        logger.info(f"Built workflow chain with {len(chain_lines)} lines")
        return chain_lines

    def _get_builder_name_from_label(self, label, node_id):
        """ラベルからBuilderでの名前を取得"""
        if "BuildSonataNetworkNode" in str(label):
            return "SonataNetworkBuilder"
        elif "SimulateSonataNetworkNode" in str(label):
            return "SonataNetworkSimulation"
        else:
            return f"Node_{node_id}"

    def update_project_code(self, project_id):
        """プロジェクト全体のコードを再生成（既存メソッドの改善版）"""
        try:
            logger.info(f"=== Updating entire project code for project {project_id} ===")
            
            # データベースから現在のフローデータを取得
            project = FlowProject.objects.get(id=project_id)
            nodes = FlowNode.objects.filter(project=project)
            edges = FlowEdge.objects.filter(project=project)
            
            # JSONフォーマットに変換
            nodes_data = []
            for node in nodes:
                node_data = {
                    "id": node.id,
                    "position": {"x": node.position_x, "y": node.position_y},
                    "type": node.node_type,
                    "data": node.data,
                }
                nodes_data.append(node_data)
            
            edges_data = []
            for edge in edges:
                edge_data = {
                    "id": edge.id,
                    "source": edge.source_node_id,
                    "target": edge.target_node_id,
                }
                if edge.source_handle:
                    edge_data["sourceHandle"] = edge.source_handle
                if edge.target_handle:
                    edge_data["targetHandle"] = edge.target_handle
                if edge.edge_data:
                    edge_data["data"] = edge.edge_data
                edges_data.append(edge_data)
            
            # バッチ生成メソッドを使用
            success = self.generate_code_from_flow_data(project_id, nodes_data, edges_data)
            
            if success:
                logger.info("=== Project code update completed successfully ===")
            else:
                logger.error("=== Project code update failed ===")
            
            return success
            
        except Exception as e:
            logger.error(f"=== Critical error updating project code: {e} ===")
            logger.error(traceback.format_exc())
            return False
