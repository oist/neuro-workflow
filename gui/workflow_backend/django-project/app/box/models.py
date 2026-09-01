import logging
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models

logger = logging.getLogger(__name__)


# Category options => Dynamically change
NODE_CATEGORIES = [
    ["analysis", "Analysis"],
    ["io", "I/O"],
    ["network", "Network"],
    ["optimization", "Optimization"],
    ["simulation", "Simulation"],
    ["stimulus", "Stimulus"],
]


def get_categories():
    """Get the category directory as a list (only real directories, skip __init__.py etc.)."""
    sub_directories = []
    nodes_path = Path(settings.MEDIA_ROOT)
    if not nodes_path.is_dir():
        return NODE_CATEGORIES
    for item in os.listdir(nodes_path):
        if item.startswith("__") or not (nodes_path / item).is_dir():
            continue
        itemlarge = item.capitalize()
        if item == "io":
            itemlarge = "I/O"
        sub_directories.append([item, itemlarge])
    return sub_directories if sub_directories else NODE_CATEGORIES


def get_upload_path(instance, filename):
    """Upload destination determined by category"""
    category = getattr(instance, "category", "uncategorized")
    return os.path.join(category, filename)


class PythonFile(models.Model):
    """Uploaded Python File Model"""

    node_categories = get_categories()
    logger.info(f"Dynamic categories:{node_categories}")
    logger.info(f"MEDIA_ROOT：{settings.MEDIA_ROOT}")

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    category = models.CharField(
        max_length=50,
        # choices=NODE_CATEGORIES,
        choices=node_categories,
        default="analysis",
        help_text="Node category for organizing files",
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

    # Node analysis results
    node_classes = models.JSONField(
        default=dict, blank=True
    )  # Parsed node class information
    is_analyzed = models.BooleanField(default=False)  # parsed flag
    analysis_error = models.TextField(
        blank=True, null=True
    )  # Parsing error information

    # metadata
    file_size = models.IntegerField(default=0)  # File size (bytes)
    file_hash = models.CharField(
        max_length=64, unique=True, default="default"  # Temporary default value
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "box_pythonfile"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def _palette_common_fields(self, user=None):
        """Fields shared by parsed palette nodes and owner parse-failure stubs."""
        is_own = bool(
            user
            and self.uploaded_by_id
            and self.uploaded_by_id == getattr(user, "id", None)
        )
        try:
            category_display = self.get_category_display()
        except Exception:
            category_display = self.category
        fields = {
            "is_own": is_own,
            "category_key": self.category,
            "category": category_display,
            "file_id": str(self.id),
            "file_name": self.name,
        }
        status_value = getattr(self, "status", None)
        if status_value is not None:
            fields["status"] = status_value
        return fields

    def get_node_classes_for_frontend(self, user=None):
        """Returns node class information for the frontend.

        Parsed classes are draggable. Owner files with no NODE_DEFINITION
        yield a single non-draggable stub so the upload is still findable.
        """
        common = self._palette_common_fields(user)
        empty_schema = {
            "inputs": {},
            "outputs": {},
            "parameters": {},
            "methods": {},
        }

        if not self.node_classes:
            stem = Path(self.name).stem or self.name
            description = (self.analysis_error or "").strip() or (
                "No NODE_DEFINITION found — this file is not a palette node."
            )
            stub = {
                "id": f"uploaded_{self.id}_unparsed",
                "type": "uploadedNode",
                "label": stem,
                "description": description,
                "class_name": "",
                "schema": empty_schema,
                "parse_ok": False,
                "draggable": False,
            }
            stub.update(common)
            return [stub]

        frontend_nodes = []
        for class_name, class_info in self.node_classes.items():
            if not isinstance(class_info, dict):
                class_info = {}
            frontend_node = {
                "id": f"uploaded_{self.id}_{class_name}",
                "type": "uploadedNode",
                "label": class_name,
                "description": class_info.get("description", ""),
                "class_name": class_name,
                "schema": self._convert_to_full_schema(class_info),
                "parse_ok": True,
                "draggable": True,
            }
            frontend_node.update(common)
            frontend_nodes.append(frontend_node)

        return frontend_nodes

    def _convert_to_full_schema(self, class_info):
        """Include all information in the schema (object format)"""
        schema = {
            "inputs": {},
            "outputs": {},
            "parameters": {},
            "methods": class_info.get("methods", {}),
        }

        # Convert inputs
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
                # Add only if default_value exists
                if "default_value" in input_info:
                    schema["inputs"][input_name]["default_value"] = input_info[
                        "default_value"
                    ]
                # Add only if there are constraints
                if "constraints" in input_info:
                    schema["inputs"][input_name]["constraints"] = input_info[
                        "constraints"
                    ]
                if "fan_in" in input_info:
                    schema["inputs"][input_name]["fan_in"] = input_info["fan_in"]

        # Convert outputs
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
                if "fan_in" in output_info:
                    schema["outputs"][output_name]["fan_in"] = output_info["fan_in"]

        # Convert parameters
        if "parameters" in class_info:
            schema["parameters"] = self._convert_parameters(
                class_info.get("parameters", {})
            )

        return schema

    def _convert_parameters(self, parameters):
        """Convert parameters (preserve the original structure and add necessary information)"""
        converted_params = {}

        for param_name, param_info in parameters.items():
            logger.debug(f"Python_file param_name: {param_name}")
            param_data = {
                "name": param_name,
                "type": self._map_port_type_to_frontend(param_info.get("type", "any")),
                "description": param_info.get("description", ""),
            }

            # Add only if default_value exists
            if "default_value" in param_info:
                param_data["default_value"] = param_info["default_value"]

            # Add only if constraints exist (keep as is)
            if "constraints" in param_info:
                param_data["constraints"] = param_info["constraints"]

            # If there is additional metadata such as widget_type
            if "widget_type" in param_info:
                param_data["widget_type"] = param_info["widget_type"]

            # Optimization metadata
            if "optimizable" in param_info:
                param_data["optimizable"] = param_info["optimizable"]
            if "optimization_range" in param_info:
                param_data["optimization_range"] = param_info["optimization_range"]
            if "is_objective" in param_info:
                param_data["is_objective"] = param_info["is_objective"]
            if "objective_range" in param_info:
                param_data["objective_range"] = param_info["objective_range"]

            converted_params[param_name] = param_data

        return converted_params

    def _map_port_type_to_frontend(self, port_type):
        """Converting PortType to a type for the front end"""
        type_mapping = {
            "int": "int",
            "float": "float",
            "str": "str",
            "bool": "bool",
            "list": "list",
            "dict": "dict",
            "object": "object",
            "any": "any",
            "file_path": "file_path",
            "csv_file": "csv_file",
            "json_file": "json_file",
            "pickle_file": "pickle_file",
            "numpy_file": "numpy_file",
            "hdf5_file": "hdf5_file",
        }
        return type_mapping.get(str(port_type).lower(), "any")
