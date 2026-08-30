import os
import hashlib
from django.core.files.storage import default_storage
from ..models import PythonFile
from .python_analyzer import PythonNodeAnalyzer
import logging

logger = logging.getLogger(__name__)

class PythonFileService:
    """Python file management service"""

    def __init__(self):
        self.analyzer = PythonNodeAnalyzer()

    def create_python_file(self, file, user=None, name=None, description=None, category='analysis', tenant=None):
        """
        Create a Python file and run the automated analysis

        Args:
            file: uploaded file
            user: upload user
            name: Filename (optional)
            description: Description (optional)
            category: File Category (Optional)
            tenant: App tenant (internal | hackathon)

        Returns:
            PythonFile instance
        """
        from app.tenants import TENANT_INTERNAL, get_user_tenant, normalize_tenant
        from app.box.models import PythonFile as PF
        from app.box.governance import log_node_event

        tenant = normalize_tenant(tenant or (get_user_tenant(user) if user else TENANT_INTERNAL))

        # read file contents
        file_content = file.read().decode("utf-8")
        if hasattr(file, "seek"):
            file.seek(0)

        # Calculate file hash (for duplicate check)
        file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

        # duplicate check (per tenant)
        existing_file = PythonFile.objects.filter(file_hash=file_hash, tenant=tenant).first()

        # Decide file name
        if not name:
            name = file.name

        if existing_file is not None:
            same_owner = (
                user is not None
                and existing_file.uploaded_by_id is not None
                and existing_file.uploaded_by_id == user.id
            )
            if not same_owner:
                raise ValueError(
                    "A node with identical content already exists in this tenant."
                )
            python_file = existing_file
            python_file.name = name
            python_file.description = description or ""
            python_file.category = category
            python_file.file_content = file_content
            python_file.file_size = file.size
            python_file.file_hash = file_hash
            python_file.is_active = True
            python_file.file = file
            # Keep uploaded_by and status — do not take over another user's node
            # and do not reset the review pipeline.
        else:
            python_file = PythonFile.objects.create(
                name=name,
                description=description or "",
                category=category,
                file=file,
                file_content=file_content,
                uploaded_by=user,
                file_size=file.size,
                file_hash=file_hash,
                tenant=tenant,
                status=PF.Status.PRIVATE if user else PF.Status.PUBLIC,
            )
            log_node_event(
                python_file,
                actor=user,
                action="uploaded",
                to_status=python_file.status,
            )

        self._persist_node_file(python_file, file_content)

        # Automatic analysis execution
        self._analyze_file(python_file)

        return python_file

    def _analyze_file(self, python_file):
        """
        Parse the file and extract node information

        Args:
            python_file: PythonFile instance
        """
        try:
            # parse file contents
            node_classes = self.analyzer.analyze_file_content(python_file.file_content)

            print(f"Analyzed {len(node_classes)} node classes:")
            for node in node_classes:
                print(f"  - {node['class_name']}")

            # Save analysis results to DB
            python_file.node_classes = {
                node["class_name"]: {
                    "description": node["description"],
                    "node_type": node["node_type"],
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
            # Even if analysis fails, the file will be saved, but error information will be recorded.
            python_file.is_analyzed = False
            python_file.analysis_error = str(e)
            python_file.save()

            print(f"Failed to analyze file {python_file.name}: {e}")

    def get_file_content(self, python_file):
        """Get file contents"""
        return python_file.file_content

    def update_file_content(self, python_file, content):
        """Update the file contents, update the physical file, and re-analyze"""
        # file_content update field
        python_file.file_content = content
        python_file.file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        python_file.file_size = len(content.encode("utf-8"))
        
        # nodes/{category}/Update physical files in a folder
        self._persist_node_file(python_file, content)
        
        # Djangoのfile Also update the field (preserve existing implementation)
        if python_file.file:
            try:
                # Delete existing files
                old_file_path = python_file.file.name
                if default_storage.exists(old_file_path):
                    default_storage.delete(old_file_path)
                
                # Create a physical file with the new file contents
                from django.core.files.base import ContentFile
                new_file = ContentFile(content.encode("utf-8"))
                python_file.file.save(python_file.name, new_file, save=False)
                
            except Exception as e:
                print(f"Warning: Failed to update Django file field for {python_file.name}: {e}")
        
        # Save DB
        python_file.save()

        # Execute reanalysis
        self._analyze_file(python_file)

        return python_file
    
    def _persist_node_file(self, python_file, content):
        """Write bytes under the tenant nodes root; drop hackathon copies from MEDIA_ROOT."""
        from app.tenants import TENANT_HACKATHON, normalize_tenant

        self._update_nodes_folder_file(python_file, content)
        if normalize_tenant(getattr(python_file, "tenant", None)) != TENANT_HACKATHON:
            return
        if not python_file.file:
            return
        try:
            name = python_file.file.name
            if name and default_storage.exists(name):
                default_storage.delete(name)
        except Exception as e:
            logger.warning(
                "Failed to drop internal MEDIA_ROOT copy of hackathon node %s: %s",
                python_file.name,
                e,
            )

    def _update_nodes_folder_file(self, python_file, content):
        """nodes/{category}/Update physical files in a folder"""
        try:
            from django.conf import settings
            from pathlib import Path
            from app.workflow.path_utils import nodes_root

            # Convert categories to lowercase
            category = python_file.category.lower()

            # nodes/{category}/Building a Folder Path
            nodes_folder = nodes_root(getattr(python_file, "tenant", None)) / category
            
            # If the folder does not exist, create it
            nodes_folder.mkdir(parents=True, exist_ok=True)
            
            # get file name(If there is no .py file, add it.)
            filename = python_file.name
            if not filename.endswith('.py'):
                filename = f"{filename}.py"
            
            # build file path
            file_path = nodes_folder / filename
            
            # write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Successfully updated physical file: {file_path}")
            
        except Exception as e:
            print(f"Warning: Failed to update nodes folder file for {python_file.name}: {e}")
            # Even if the physical file update fails, the DB is updated, so processing continues

    def validate_python_syntax(self, content):
        """Check Python syntax"""
        try:
            compile(content, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, str(e)
