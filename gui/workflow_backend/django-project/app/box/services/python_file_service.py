import os
import hashlib
from django.core.files.storage import default_storage
from ..models import PythonFile
from .python_analyzer import PythonNodeAnalyzer


class PythonFileService:
    """Pythonファイル管理サービス"""

    def __init__(self):
        self.analyzer = PythonNodeAnalyzer()

    def create_python_file(self, file, user=None, name=None, description=None, category='analysis'):
        """
        Pythonファイルを作成し、自動解析を実行

        Args:
            file: アップロードされたファイル
            user: アップロードユーザー
            name: ファイル名（オプション）
            description: 説明（オプション）
            category: ファイルカテゴリ（オプション）

        Returns:
            PythonFile instance
        """
        # ファイル内容を読み取り
        file_content = file.read().decode("utf-8")

        # ファイルハッシュを計算（重複チェック用）
        file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

        # 重複チェック
        existing_file = PythonFile.objects.filter(file_hash=file_hash).first()
        if existing_file:
            raise ValueError(f"File already exists: {existing_file.name}")

        # ファイル名を決定
        if not name:
            name = file.name

        # PythonFileインスタンス作成
        python_file = PythonFile.objects.create(
            name=name,
            description=description or "",
            category=category,
            file=file,
            file_content=file_content,
            uploaded_by=user,
            file_size=file.size,
            file_hash=file_hash,
        )

        # 自動解析実行
        self._analyze_file(python_file)

        return python_file

    def _analyze_file(self, python_file):
        """
        ファイルを解析してノード情報を抽出

        Args:
            python_file: PythonFileインスタンス
        """
        try:
            # ファイル内容を解析
            node_classes = self.analyzer.analyze_file_content(python_file.file_content)

            print(f"Analyzed {len(node_classes)} node classes:")
            for node in node_classes:
                print(f"  - {node['class_name']}")

            # 解析結果をDBに保存（node_typeを追加！）
            python_file.node_classes = {
                node["class_name"]: {
                    "description": node["description"],
                    "node_type": node["node_type"],  # ← これが抜けていた！
                    "parameters": node["parameters"],
                    "inputs": node["inputs"],
                    "outputs": node["outputs"],
                    "methods": node["methods"],
                }
                for node in node_classes
            }
            python_file.is_analyzed = True
            python_file.analysis_error = None
            python_file.save()

            print(
                f"Successfully analyzed {len(node_classes)} node classes from {python_file.name}"
            )

        except Exception as e:
            # 解析失敗時もファイルは保存するが、エラー情報を記録
            python_file.is_analyzed = False
            python_file.analysis_error = str(e)
            python_file.save()

            print(f"Failed to analyze file {python_file.name}: {e}")

    def get_file_content(self, python_file):
        """ファイル内容を取得"""
        return python_file.file_content

    def update_file_content(self, python_file, content):
        """ファイル内容を更新し、物理ファイルも更新して再解析を実行"""
        # file_content フィールドを更新
        python_file.file_content = content
        python_file.file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        python_file.file_size = len(content.encode("utf-8"))
        
        # nodes/{category}/フォルダ内の物理ファイルを更新
        self._update_nodes_folder_file(python_file, content)
        
        # Djangoのfile フィールドも更新（既存の実装を保持）
        if python_file.file:
            try:
                # 既存のファイルを削除
                old_file_path = python_file.file.name
                if default_storage.exists(old_file_path):
                    default_storage.delete(old_file_path)
                
                # 新しいファイル内容で物理ファイルを作成
                from django.core.files.base import ContentFile
                new_file = ContentFile(content.encode("utf-8"))
                python_file.file.save(python_file.name, new_file, save=False)
                
            except Exception as e:
                print(f"Warning: Failed to update Django file field for {python_file.name}: {e}")
        
        # DBを保存
        python_file.save()

        # 再解析実行
        self._analyze_file(python_file)

        return python_file
    
    def _update_nodes_folder_file(self, python_file, content):
        """nodes/{category}/フォルダ内の物理ファイルを更新"""
        try:
            from django.conf import settings
            from pathlib import Path
            
            # カテゴリを小文字に変換
            category = python_file.category.lower()
            
            # nodes/{category}/フォルダのパスを構築
            nodes_folder = Path(settings.MEDIA_ROOT) / category
            
            # フォルダが存在しない場合は作成
            nodes_folder.mkdir(parents=True, exist_ok=True)
            
            # ファイル名を取得（.pyが付いていない場合は追加）
            filename = python_file.name
            if not filename.endswith('.py'):
                filename = f"{filename}.py"
            
            # ファイルパスを構築
            file_path = nodes_folder / filename
            
            # ファイルに書き込み
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Successfully updated physical file: {file_path}")
            
        except Exception as e:
            print(f"Warning: Failed to update nodes folder file for {python_file.name}: {e}")
            # 物理ファイル更新に失敗してもDBは更新されているので処理を続行

    def validate_python_syntax(self, content):
        """Python構文をチェック"""
        try:
            compile(content, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, str(e)
