from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.db import models
from .models import PythonFile, NODE_CATEGORIES
from .models import get_categories
from .serializers import PythonFileSerializer, PythonFileUploadSerializer
from .services.python_file_service import PythonFileService
import logging
import hashlib
import uuid
import os
import json
from pathlib import Path
from django.core.files import File
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from app.auth.authentication import KeycloakAuthentication

logger = logging.getLogger(__name__)


def _can_modify_python_file(user, python_file):
    """User-created files are owner-only; synced catalog files are shared."""
    if not user or not user.is_authenticated:
        return False
    return not python_file.uploaded_by_id or python_file.uploaded_by_id == user.id


def _visible_python_files(user):
    return PythonFile.objects.filter(is_active=True).filter(
        models.Q(uploaded_by=user) | models.Q(uploaded_by__isnull=True)
    )


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileUploadView(APIView):
    """Python view for file upload"""

    parser_classes = (MultiPartParser, FormParser)
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Upload a file and analyze it automatically"""
        serializer = PythonFileUploadSerializer(data=request.data)

        if serializer.is_valid():
            try:
                file_service = PythonFileService()

                # Create and analyze files
                python_file = file_service.create_python_file(
                    file=serializer.validated_data["file"],
                    user=request.user,
                    name=serializer.validated_data.get("name"),
                    description=serializer.validated_data.get("description"),
                    category=serializer.validated_data.get("category", "analysis"),
                )

                # Serializer for the response
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
    """Python file list and details view"""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, pk=None):
        """Get file list or details"""
        if pk:
            # Get details
            python_file = get_object_or_404(_visible_python_files(request.user), pk=pk)
            serializer = PythonFileSerializer(python_file, context={"request": request})
            return Response(serializer.data)
        else:
            # get list
            python_files = _visible_python_files(request.user)

            # filtering
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
        """delete file"""
        python_file = get_object_or_404(_visible_python_files(request.user), pk=pk)

        # Permission check
        if not _can_modify_python_file(request.user, python_file):
            return Response(
                {"error": "You don't have permission to delete this file"},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Delete a file from the file system
        if python_file.file:
            try:
                python_file.file.delete(save=False)
                logger.info(f"Deleted file from filesystem: {python_file.file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete file from filesystem: {e}")

        # logical delete
        python_file.is_active = False
        python_file.save()

        return Response(status=status.HTTP_204_NO_CONTENT)


@method_decorator(csrf_exempt, name="dispatch")
class UploadedNodesView(APIView):
    """API to get a list of uploaded node classes"""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Returns a list of uploaded node classes"""
        try:
            # Only retrieve valid parsed files
            python_files = _visible_python_files(request.user).filter(
                is_analyzed=True, node_classes__isnull=False
            ).exclude(node_classes={})

            all_nodes = []
            for python_file in python_files:
                frontend_nodes = python_file.get_node_classes_for_frontend()
                all_nodes.extend(frontend_nodes)

            # Category list
            node_categories = get_categories()
            valid_categories = [category[0] for category in node_categories]
            cat_settings = {}
            nodes_path = Path(settings.MEDIA_ROOT)

            for category in valid_categories:
                category_path = nodes_path / category

                if not category_path.exists():
                    logger.info(f"Category folder not found, creating: {category_path}")
                    category_path.mkdir(parents=True, exist_ok=True)
                    continue

                settings_path = os.path.join( category_path, ".settings")

                if os.path.exists(settings_path):
                    logger.info(f"settings_path : {settings_path}")
                    try:
                        settings_open = open(settings_path, "r")
                        cat_settings[category] = json.load(settings_open)
                        settings_open.close()
                    except Exception as e:
                        logger.error(f".settings File load failed: {e}")
                else:
                    try:
                        default_settings ={ "color" : "#6b46c1" }
                        settings_open = open(settings_path, "w")
                        json.dump(default_settings, settings_open)
                        settings_open.close()
                    except Exception as e:
                        logger.error(f".settings File create failed: {e}")

            return Response(
                {
                    "nodes": all_nodes,
                    "total_files": python_files.count(),
                    "total_nodes": len(all_nodes),
                    "categories": cat_settings
                }
            )

        except Exception as e:
            logger.error(f"Failed to get uploaded nodes: {e}")
            return Response(
                {"error": f"Failed to get uploaded nodes: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


from django.http import JsonResponse
from django.core.files.base import ContentFile
import os
import re


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileCodeManagementView(APIView):
    """
    Code management view of PythonFile
    GET: Get code
    PUT: save code
    """

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, filename):
        """Get source code by specifying the file name"""
        try:
            # If .py is not attached, search will be performed even if it is attached.
            filenames_to_search = [filename]
            if not filename.endswith(".py"):
                filenames_to_search.append(f"{filename}.py")

            # Search by file name
            python_file = _visible_python_files(request.user).filter(
                name__in=filenames_to_search
            ).first()

            if not python_file:
                return JsonResponse(
                    {"error": f"File '{filename}' not found"}, status=404
                )

            # Prioritize file_content, if not available use source_code
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
        """Save the edited code to the database"""
        try:
            data = json.loads(request.body)
            code = data.get("code", "")

            if not code:
                return JsonResponse({"error": "Code is required"}, status=400)

            # If .py is not attached, search with it
            filenames_to_search = [filename]
            if not filename.endswith(".py"):
                filenames_to_search.append(f"{filename}.py")

            # Search by file name
            python_file = _visible_python_files(request.user).filter(
                name__in=filenames_to_search
            ).first()

            if not python_file:
                return JsonResponse(
                    {"error": f"File '{filename}' not found"}, status=404
                )

            # Permission checks (if necessary)
            if not _can_modify_python_file(request.user, python_file):
                return JsonResponse(
                    {"error": "You don't have permission to edit this file"}, status=403
                )

            # Use PythonFileService to update code and reparse
            from .services.python_file_service import PythonFileService

            file_service = PythonFileService()

            # Update the file contents and reparse
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

@method_decorator(csrf_exempt, name="dispatch")
class PythonFileCopyView(APIView):
    """Copy selected Python files"""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Copy by specifying file ID or file name"""
        try:
            data = request.data
            file_ids = data.get("file_ids", [])
            source_filename = data.get("source_filename")
            target_filename = data.get("target_filename")

            # New method: Copy by filename
            if source_filename and target_filename:
                return self._copy_by_filename(request, source_filename, target_filename)

            # Traditional method: Copy by file_ids
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
                    # Get original file
                    original_file = get_object_or_404(
                        _visible_python_files(request.user), pk=file_id
                    )

                    # Generate copy name
                    copy_name = self._generate_copy_name(original_file.name)

                    # Generate a unique file_hash
                    unique_hash = hashlib.sha256(
                        f"{copy_name}_{uuid.uuid4()}_{original_file.id}".encode()
                    ).hexdigest()

                    # create new file object
                    copied_file = PythonFile(
                        name=copy_name,
                        description=(
                            f"Copy of {original_file.description}"
                            if original_file.description
                            else f"Copy of {original_file.name}"
                        ),
                        category=original_file.category,
                        file_content=original_file.file_content,
                        uploaded_by=request.user,
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

                    # Copy file field
                    if original_file.file:
                        try:
                            original_file.file.open()
                            file_content = original_file.file.read()
                            original_file.file.close()

                            # Copy the file name with the .py extension
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

                    # Serializer for the response
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
        """Generate a name for the copy (preserve the .py extension)"""
        # Check for existing "copy" and "copy (n)" patterns
        base_name, ext = os.path.splitext(original_name)

        # Ensure .py extension (if no extension or something other than .py)
        if not ext or ext.lower() != ".py":
            ext = ".py"

        # Processing when "copy" is already present
        if base_name.endswith(" - copy"):
            base_name = base_name[:-7]  # " - copy" Delete
        elif " - copy (" in base_name and base_name.endswith(")"):
            # " - copy (n)" remove pattern
            copy_index = base_name.find(" - copy (")
            if copy_index != -1:
                base_name = base_name[:copy_index]

        # Find unique names
        counter = 1
        while True:
            if counter == 1:
                copy_name = f"{base_name} - copy{ext}"
            else:
                copy_name = f"{base_name} - copy ({counter}){ext}"

            # Check if a file with the same name exists
            if not PythonFile.objects.filter(name=copy_name, is_active=True).exists():
                return copy_name

            counter += 1

    def _copy_by_filename(self, request, source_filename, target_filename):
        """Copy by file name"""
        try:
            # If .py is not attached, add it and search.
            source_names = [source_filename]
            if not source_filename.endswith(".py"):
                source_names.append(f"{source_filename}.py")

            # Get source file
            original_file = _visible_python_files(request.user).filter(
                name__in=source_names
            ).first()

            if not original_file:
                return Response(
                    {"error": f"Source file '{source_filename}' not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Add the .py extension to the target filename (if needed)
            if not target_filename.endswith(".py"):
                target_filename = f"{target_filename}.py"

            # Check if a file with the same name already exists
            if PythonFile.objects.filter(name=target_filename, is_active=True).exists():
                return Response(
                    {"error": f"Target file '{target_filename}' already exists"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Copy the source code and update the class name
            updated_content = self._update_class_names_in_code(
                original_file.file_content or "", target_filename
            )

            # Also updated node_classes (class names changed based on the new filenames)
            updated_node_classes = self._update_node_classes(
                original_file.node_classes or {}, target_filename
            )

            # Generate a unique file_hash
            unique_hash = hashlib.sha256(
                f"{target_filename}_{uuid.uuid4()}_{original_file.id}".encode()
            ).hexdigest()

            # create new file object
            copied_file = PythonFile(
                name=target_filename,
                description=(
                    f"Copy of {original_file.description}"
                    if original_file.description
                    else f"Copy of {original_file.name}"
                ),
                category=original_file.category,
                file_content=updated_content,
                uploaded_by=request.user,
                node_classes=updated_node_classes,
                is_analyzed=original_file.is_analyzed,  # Keep the original file
                analysis_error=original_file.analysis_error,
                file_size=(
                    len(updated_content.encode("utf-8")) if updated_content else 0
                ),
                file_hash=unique_hash,
            )

            # Create file field
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

            # Serializer for the response
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
        """Update class names in source code based on filenames"""
        if not source_code:
            return source_code

        # Create the base class name by removing the extension from the target file name
        base_name = os.path.splitext(target_filename)[0]

        # Convert filenames to PascalCase class names
        # Example: "my_node.py" -> "MyNode", "test-node.py" -> "TestNode"
        class_name = "".join(
            word.capitalize() for word in re.split(r"[_\-]", base_name)
        )

        updated_code = source_code

        # Find and replace patterns in class definitions
        class_pattern = r"^class\s+(\w+)(\s*\([^)]*\))?:"

        def replace_class_name(match):
            original_class_name = match.group(1)
            inheritance = match.group(2) or ""
            return f"class {class_name}{inheritance}:"

        # Multi-line class definition replacement
        updated_code = re.sub(
            class_pattern, replace_class_name, updated_code, flags=re.MULTILINE
        )

        return updated_code

    def _update_node_classes(self, original_node_classes, target_filename):
        """Update the class name in node_classes based on the new file name"""
        if not original_node_classes:
            return original_node_classes

        # Create the base class name by removing the extension from the target file name
        base_name = os.path.splitext(target_filename)[0]

        # Convert filenames to PascalCase class names
        new_class_name = "".join(
            word.capitalize() for word in re.split(r"[_\-]", base_name)
        )

        updated_node_classes = {}

        for class_name, class_info in original_node_classes.items():
            # Rename the class to the new name
            updated_class_info = (
                class_info.copy() if isinstance(class_info, dict) else class_info
            )

            # Updates even if class_info contains a class name
            if isinstance(updated_class_info, dict):
                # 'name' Update if field exists
                if "name" in updated_class_info:
                    updated_class_info["name"] = new_class_name

                # Update any other class name references (if necessary)
                for key, value in updated_class_info.items():
                    if isinstance(value, str) and value == class_name:
                        updated_class_info[key] = new_class_name

            # Add the new class name as a key
            updated_node_classes[new_class_name] = updated_class_info

        return updated_node_classes

    def _replace_dict_parameter_value(
        self, source_code, parameter_key, parameter_value
    ):
        """Safely replace parameter values ​​in dictionaries (array compatible)"""
        logger.info(
            f"Replacing dict parameter '{parameter_key}' with value: {parameter_value} (type: {type(parameter_value)})"
        )
        formatted_value = self._format_value_for_python(parameter_value)
        logger.info(f"Formatted value: {formatted_value}")

        # A safer pattern: find the key and then accurately detect the end of the value
        patterns = [
            rf'(["\']){re.escape(parameter_key)}\1\s*:\s*',  # 'key': or "key":
        ]

        for pattern in patterns:
            match = re.search(pattern, source_code)
            if match:
                # End position of the key part
                key_end = match.end()

                # Detects the end of a value (considers nested parentheses and strings)
                value_end = self._find_dict_value_end(source_code, key_end)

                if value_end > key_end:
                    # Replace the value part completely (delete the original value)
                    new_source = (
                        source_code[:key_end]
                        + formatted_value
                        + source_code[value_end:]
                    )
                    return new_source

        return source_code

    def _find_dict_value_end(self, source_code, start_pos):
        """Find the end of a dictionary value (supports nested arrays and dictionaries)"""
        bracket_count = 0
        brace_count = 0
        paren_count = 0
        in_string = False
        quote_char = None
        i = start_pos

        # Read values ​​from the starting position
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
                    # If the array is completely closed, check the next character.
                    if bracket_count < 0:
                        return i
                elif char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    # When the dictionary is completely closed, or this is the end of the outer dictionary
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
                    # A value terminates only if all brackets/braces are closed
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
        """Safely replace parameter values ​​in function arguments (array compatible)"""
        logger.info(
            f"Replacing function parameter '{parameter_key}' with value: {parameter_value} (type: {type(parameter_value)})"
        )
        formatted_value = self._format_value_for_python(parameter_value)
        logger.info(f"Formatted value: {formatted_value}")

        # Pattern: Exactly find the end of the value after parameter_key=
        pattern = rf"\b{re.escape(parameter_key)}\s*=\s*"
        match = re.search(pattern, source_code)

        if match:
            # End position of the parameter key part
            key_end = match.end()

            # Detects the end of a value (considers nested parentheses and strings)
            value_end = self._find_function_parameter_value_end(source_code, key_end)

            if value_end > key_end:
                # Replace the value part completely (delete the original value)
                new_source = (
                    source_code[:key_end] + formatted_value + source_code[value_end:]
                )
                return new_source

        return source_code

    def _find_function_parameter_value_end(self, source_code, start_pos):
        """Detect the end position of a function parameter value (supports nested arrays and dictionaries)"""
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
                        # End of outer bracket
                        return i
                elif (
                    char == ","
                    and bracket_count == 0
                    and brace_count == 0
                    and paren_count == 0
                ):
                    # A parameter terminates only if all brackets/braces/parentheses are closed
                    return i
            else:
                if char == quote_char and (i == 0 or source_code[i - 1] != "\\"):
                    in_string = False
                    quote_char = None

            i += 1

        return len(source_code)


@method_decorator(csrf_exempt, name="dispatch")
class PythonFileParameterUpdateView(APIView):
    """Update parameter values ​​in a file"""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def put(self, request):
        """Update parameter value"""
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

            # Get file (search by file_id or filename)
            python_file = None
            if file_id:
                python_file = get_object_or_404(_visible_python_files(request.user), pk=file_id)
            elif filename:
                filenames_to_search = [filename]
                if not filename.endswith(".py"):
                    filenames_to_search.append(f"{filename}.py")

                python_file = _visible_python_files(request.user).filter(
                    name__in=filenames_to_search
                ).first()

            if not python_file:
                return Response(
                    {"error": "File not found"}, status=status.HTTP_404_NOT_FOUND
                )

            # Permission check
            if not _can_modify_python_file(request.user, python_file):
                return Response(
                    {"error": "You don't have permission to edit this file"},
                    status=status.HTTP_403_FORBIDDEN,
                )

            for class_info in (python_file.node_classes or {}).values():
                param_info = (class_info.get("parameters") or {}).get(parameter_key) or {}
                if param_info.get("secret") and parameter_field in ("value", "default_value"):
                    return Response(
                        {
                            "error": "Secret parameters cannot be written into node source. Use Settings → Secrets."
                        },
                        status=status.HTTP_400_BAD_REQUEST,
                    )

            # Update parameter values ​​in source code (uniform for all cases)
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

            # Use PythonFileService to update code and reparse
            from .services.python_file_service import PythonFileService

            file_service = PythonFileService()

            # Update the file contents and reparse
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
        """Updates the specified parameter field in the source code."""
        if not source_code:
            return source_code

        updated_code = source_code

        if parameter_field == "value":
            # Try patterns in order and stop if any one of them changes
            original_code = updated_code

            # Pattern 1: Variable assignment(Example: record_from_population = 100)
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

            # Exit if there is a change in variable assignment
            if updated_code != original_code:
                return updated_code

            # Pattern 2: Values ​​in a dictionary (Example: {"record_from_population": 100} or {"time_window": [0.0, 1000.0]})
            # A safer approach: accurately detecting the end of a value
            updated_code = self._replace_dict_parameter_value(
                updated_code, parameter_key, parameter_value
            )

            # Exit if there is a change in the value in the dictionary
            if updated_code != original_code:
                return updated_code

            # Pattern 3: Function call arguments (Example: func(record_from_population=100) or func(time_window=[0, 100]))
            # Use an array-safe method
            updated_code = self._replace_function_parameter_value(
                updated_code, parameter_key, parameter_value
            )

        else:
            # If parameter_field is 'default_value', 'constraints', etc.
            # Look for and update PortParameter-like structures
            updated_code = self._update_parameter_metadata_in_code(
                updated_code, parameter_key, parameter_field, parameter_value
            )

        return updated_code

    def _format_value_for_python(self, value):
        """Format values for Python code as valid Python literals."""
        logger.info(f"Formatting value: {value} (type: {type(value)})")
        # repr() always produces a valid, correctly-escaped Python literal for all
        # built-in types (str, int, float, bool, None, list, dict, tuple).
        # This avoids the previous str().replace("'", '"') approach for dicts, which
        # failed for string values containing apostrophes, booleans, nested dicts, etc.
        return repr(value)

    def _update_parameter_metadata_in_code(
        self, source_code, parameter_key, field_name, field_value
    ):
        """Update parameter metadata (default_value, constraints, etc.) in source code"""
        updated_code = source_code

        print(
            "All data in this field",
            f"key:{parameter_key}, name:{field_name}, value:{field_value}",
            flush=True,
        )

        # First, check whether the specified parameter_key exists in parameters={}
        if not self._parameter_exists_in_parameters_dict(source_code, parameter_key):
            # If not present in parameters, do nothing
            logger.info(
                f"Parameter '{parameter_key}' not found in parameters dict, skipping"
            )
            return source_code

        logger.info(
            f"Starting update for parameter '{parameter_key}', field '{field_name}'"
        )

        # Simple approach: Update the field_name directly in the ParameterDefinition
        return self._update_parameter_field_simple(
            updated_code, parameter_key, field_name, field_value
        )

    def _update_parameter_field_simple(
        self, source_code, parameter_key, field_name, field_value
    ):
        """Update fields in a ParameterDefinition with a simple regular expression"""

        # Look for the ParameterDefinition pattern (more flexible pattern)
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

        # Search and replace within the entire ParameterDefinition
        start_pos = match.start()

        # Finding the end of a ParameterDefinition (more accurate method)
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

        # Extract the ParameterDefinition part
        param_def = source_code[start_pos:end_pos]

        # Safely update the field in field_name
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
        """Safely replace specific fields in a ParameterDefinition"""
        # Look for patterns in field names
        field_pattern = rf'\b{re.escape(field_name)}\s*=\s*'
        match = re.search(field_pattern, param_def)
        
        if not match:
            return param_def
            
        # Starting position of field value
        value_start = match.end()
        
        # Detect the end of a field value (supports nested arrays and dictionaries)
        value_end = self._find_parameter_field_value_end(param_def, value_start)
        
        if value_end > value_start:
            formatted_value = self._format_value_for_python(field_value)
            # Replace value completely
            return (
                param_def[:value_start] + 
                formatted_value + 
                param_def[value_end:]
            )
        
        return param_def
    
    def _find_parameter_field_value_end(self, param_def, start_pos):
        """Find the end of a field value in a ParameterDefinition"""
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
                        # End of ParameterDefinition
                        return i
                elif char == ',' and bracket_count == 0 and brace_count == 0 and paren_count == 0:
                    # end of field
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
        """Update only the specified fields of the specified parameters (more reliable method)"""

        # Look for an exact match of the pattern '"parameter_key": ParameterDefinition('
        exact_patterns = [
            rf'"{re.escape(parameter_key)}"\s*:\s*ParameterDefinition\s*\(', 
            rf"'{re.escape(parameter_key)}'\s*:\s*ParameterDefinition\s*\(", 
        ]

        target_match = None
        target_pattern = None

        for pattern in exact_patterns:
            matches = list(re.finditer(pattern, source_code))
            if matches:
                # Select only exact matches
                for match in matches:
                    # Check the parts before and after the match to see if they are completely independent keys
                    start_pos = match.start()

                    # Check the previous character (make sure it's not an alphanumeric character)
                    if start_pos > 0:
                        prev_char = source_code[start_pos - 1]
                        if prev_char.isalnum() or prev_char == "_":
                            continue  # Partial match, skipped

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

        # ParFind the end of ParameterDefinition(...)
        start_pos = target_match.end() - 1  # '('
        end_pos = self._find_matching_paren(source_code, start_pos)

        if end_pos == -1:
            logger.error(
                f"Could not find matching closing paren for parameter '{parameter_key}'"
            )
            return source_code

        # Get the contents of ParameterDefinition
        param_content = source_code[start_pos + 1 : end_pos]

        logger.info(f"Extracted content for '{parameter_key}': {param_content}")

        # Update the specified field within this content
        updated_content = self._update_field_in_parameter_content(
            param_content, field_name, field_value
        )

        logger.info(
            "Updated parameter '%s' field '%s'",
            parameter_key,
            field_name,
        )

        # Updated original code
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
        """Find the closing parenthesis that matches the opening parenthesis at the specified position"""
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

        return -1  # Matching closing parenthesis not found

    def _update_field_in_parameter_content(
        self, param_content, field_name, field_value
    ):
        """Updates a specific field with the content in the ParameterDefinition"""
        formatted_value = self._format_value_for_python(field_value)

        # More precise patterns: strictly define field name boundaries
        # Accurate extraction of nested dictionaries and complex values
        field_pattern = (
            rf"\b{re.escape(field_name)}\s*=\s*(?:[^,\)]*(?:\([^)]*\)[^,\)]*)*)"
        )

        # Takes parentheses into account to more accurately detect the end of a value
        def find_field_value_end(content, start_pos):
            """Finding the end position of a field value"""
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
                            # Reaching the outer parenthesis
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
                        # end of field
                        return i
                else:
                    if char == quote_char and (i == 0 or content[i - 1] != "\\"):
                        in_string = False
                        quote_char = None

                i += 1

            return len(content)

        # Find and update fields
        field_pattern = rf"\b{re.escape(field_name)}\s*=\s*"
        field_match = re.search(field_pattern, param_content)

        logger.info(f"Searching for field '{field_name}' in content: {param_content}")
        logger.info(f"Field pattern: {field_pattern}")
        logger.info(f"Field match found: {field_match is not None}")

        if field_match:
            # Update existing field
            field_start = field_match.start()
            value_start = field_match.end()
            value_end = find_field_value_end(param_content, value_start)

            logger.info(
                f"Field position: start={field_start}, value_start={value_start}, value_end={value_end}"
            )
            logger.info(f"Original value: {param_content[value_start:value_end]}")

            # Replace field part
            updated_content = (
                param_content[:field_start]
                + f"{field_name}={formatted_value}"
                + param_content[value_end:]
            )
            logger.info(f"Replacement result: {updated_content}")
            return updated_content
        else:
            # Add new field
            param_content = param_content.strip()
            if param_content:
                # If there are existing parameters
                return f"{param_content}, {field_name}={formatted_value}"
            else:
                # If empty
                return f"{field_name}={formatted_value}"

    def _parameter_exists_in_parameters_dict(self, source_code, parameter_key):
        """parameters={} Checks whether the specified parameter key exists in the dictionary."""

        # parameters = { ... } Find the block of
        params_pattern = r"parameters\s*=\s*\{"

        match = re.search(params_pattern, source_code)

        if not match:
            return False

        # starting position of parameters dictionary
        start_pos = match.end() - 1  # '{' 

        # Find the end of a dictionary (taking nested parentheses into account)
        brace_count = 0
        end_pos = len(source_code) - 1  # Default to end of file
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

        # Get the contents of the parameters dictionary
        params_content = source_code[start_pos : end_pos + 1]

        # print("This parameter content", params_content, flush=True)

        # Checks whether the specified parameter_key is included
        # Matches double or single quotes
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
    """View for getting a list of node categories"""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        """Returns a list of available categories"""
        try:
            node_categories = get_categories()

            #logger.info(f"categories/ : {node_categories}")

            # Scan each category folder
            valid_categories = [category[0] for category in node_categories]

            category_settings = {}
            nodes_path = Path(settings.MEDIA_ROOT)

            for category in valid_categories:
                category_path = nodes_path / category

                if not category_path.exists():
                    logger.info(f"Category folder not found, creating: {category_path}")
                    category_path.mkdir(parents=True, exist_ok=True)
                    continue

                settings_path = os.path.join( category_path, ".settings")

                if os.path.exists(settings_path):
                    logger.info(f"settings_path : {settings_path}")
                    try:
                        settings_open = open(settings_path, "r")
                        category_settings[category] = json.load(settings_open)
                        settings_open.close()
                    except Exception as e:
                        logger.error(f".settings File load failed: {e}")
                else:
                    try:
                        default_settings ={ "color" : "#6b46c1" }
                        settings_open = open(settings_path, "w")
                        json.dump(default_settings, settings_open)
                        settings_open.close()
                    except Exception as e:
                        logger.error(f".settings File create failed: {e}")


            categories = [
                {"value": value, "label": label, "settings": category_settings[value]} for value, label in node_categories
            ]

            return Response({"categories": categories, "default": "analysis"})
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return Response(
                {"error": "Failed to get categories"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def post(self, request):
        """Save category colors => .settings"""
        try:
            data = request.data
            category_key = data.get("category")
            category_value = data.get("color")

            if category_key is None:
                return Response(
                    {"error": "category_key is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if category_value is None:
                return Response(
                    {"error": "category_value is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            category_settings = {"color": category_value}
            nodes_path = Path(settings.MEDIA_ROOT)
            category_path = nodes_path / category_key
            settings_path = os.path.join( category_path, ".settings")
            settings_open = open(settings_path, "w")
            json.dump(category_settings, settings_open)
            settings_open.close()

            response_data = {
                "status": "success",
                "message": "Category color updated successfully",
                "data": "",
            }

            logger.info(
                f"Successfully updated category color {category_key}:{category_value}"
            )
            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error updating category color: {e}")
            return Response(
                {"error": "Failed to update category color"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@method_decorator(csrf_exempt, name="dispatch")
class BulkSyncNodesView(APIView):
    """A view that synchronizes the contents of the nodes folder to the database all at once."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        """Scan the nodes folder and add them to the DB all at once"""
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

            node_categories = get_categories()

            # Scan each category folder
            valid_categories = [category[0] for category in node_categories]

            cat_settings = {}

            for category in valid_categories:
                category_path = nodes_path / category

                if not category_path.exists():
                    logger.info(f"Category folder not found, creating: {category_path}")
                    category_path.mkdir(parents=True, exist_ok=True)
                    continue

                settings_path = os.path.join( category_path, ".settings")

                if os.path.exists(settings_path):
                    logger.info(f"settings_path : {settings_path}")
                    try:
                        settings_open = open(settings_path, "r")
                        cat_settings[category] = json.load(settings_open)
                        settings_open.close()
                    except Exception as e:
                        logger.error(f".settings File load failed: {e}")
                else:
                    try:
                        default_settings ={ "color" : "#6b46c1" }
                        settings_open = open(settings_path, "w")
                        json.dump(default_settings, settings_open)
                        settings_open.close()
                    except Exception as e:
                        logger.error(f".settings File create failed: {e}")

                # Scan .py files
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

            results["settings"] = cat_settings

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

            # read file contents
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Check for duplicates using file hashes
            file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()

            # Existing file check (active files only, by hash or filename + category)
            existing_file = PythonFile.objects.filter(
                (
                    models.Q(file_hash=file_hash)
                    | models.Q(name=filename, category=category)
                )
                & models.Q(is_active=True)
            ).first()

            if existing_file:
                if existing_file.file_hash == file_hash:
                    # Identical content — nothing to do
                    return {
                        "status": "skipped",
                        "filename": filename,
                        "category": category,
                        "reason": "Identical content already exists",
                        "existing_id": str(existing_file.id),
                        "existing_name": existing_file.name,
                    }
                else:
                    # File content changed — update and re-analyze
                    try:
                        file_service = PythonFileService()
                        file_service.update_file_content(existing_file, file_content)
                        return {
                            "status": "added",
                            "filename": filename,
                            "category": category,
                            "file_id": str(existing_file.id),
                            "analyzed": existing_file.is_analyzed,
                            "reason": "Updated existing file with new content",
                        }
                    except Exception as e:
                        logger.warning(f"Update failed for {filename}: {e}")
                        return {
                            "status": "error",
                            "filename": filename,
                            "category": category,
                            "error": str(e),
                        }

            # Create a new file (DB registration only, no file copying)
            python_file = PythonFile.objects.create(
                name=filename,
                description=f"Synced from {category} folder",
                category=category,
                file_content=file_content,
                file_size=file_path.stat().st_size,
                file_hash=file_hash,
                # Leave the file field empty (not necessary since file_content is used)
            )

            # Automatic analysis execution
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
