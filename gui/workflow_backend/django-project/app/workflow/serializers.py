from rest_framework import serializers
from .models import FlowProject, FlowNode, FlowEdge, WorkflowRun
from django.contrib.auth.models import User

from app.box.models import get_categories
from app.secrets.redaction import (
    WorkflowContextBlockedError,
    assert_context_has_no_secrets,
    param_is_secret,
    redact_node_data,
    redact_param_value,
)


def _valid_category_values() -> list[str]:
    """Return lowercase category directory names (e.g. ['analysis', 'io', ...])."""
    return [cat[0] for cat in get_categories()]


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "first_name", "last_name"]


class FlowProjectSerializer(serializers.ModelSerializer):
    owner = UserSerializer(read_only=True)
    nodes_count = serializers.SerializerMethodField()
    edges_count = serializers.SerializerMethodField()
    is_owned_by_me = serializers.SerializerMethodField()
    can_edit = serializers.SerializerMethodField()
    can_delete = serializers.SerializerMethodField()
    can_change_visibility = serializers.SerializerMethodField()

    class Meta:
        model = FlowProject
        fields = [
            "id",
            "name",
            "description",
            "workflow_context",
            "owner",
            "visibility",
            "reference",
            "hpc_target",
            "doi",
            "data_source",
            "license",
            "funding",
            "contact_email",
            "links",
            "contributors",
            "created_at",
            "updated_at",
            "is_active",
            "nodes_count",
            "edges_count",
            "is_owned_by_me",
            "can_edit",
            "can_delete",
            "can_change_visibility",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "owner",
            "is_active",
            "is_owned_by_me",
            "can_edit",
            "can_delete",
            "can_change_visibility",
        ]

    def get_nodes_count(self, obj):
        if not obj.pk:
            return 0
        return getattr(obj, "nodes", []).count() if hasattr(obj, "nodes") else 0

    def get_edges_count(self, obj):
        if not obj.pk:
            return 0
        return getattr(obj, "edges", []).count() if hasattr(obj, "edges") else 0

    def _request_user(self):
        request = self.context.get("request") if self.context else None
        return getattr(request, "user", None) if request else None

    def get_is_owned_by_me(self, obj):
        user = self._request_user()
        return bool(user and user.is_authenticated and obj.owner_id == user.id)

    def get_can_edit(self, obj):
        if self.get_is_owned_by_me(obj):
            return True
        return obj.visibility == FlowProject.Visibility.PUBLIC

    def get_can_delete(self, obj):
        return self.get_is_owned_by_me(obj)

    def get_can_change_visibility(self, obj):
        return self.get_is_owned_by_me(obj)

    def validate_workflow_context(self, value):
        try:
            assert_context_has_no_secrets(value or {})
        except WorkflowContextBlockedError as exc:
            raise serializers.ValidationError(str(exc))
        return value or {}

    def validate_name(self, value):
        name = (value or "").strip()
        if not name:
            raise serializers.ValidationError("Project name cannot be empty.")
        # Soft uniqueness: an owner should not have two active projects with
        # the same display name (helps the project picker stay unambiguous).
        owner = None
        if self.instance is not None:
            owner = self.instance.owner
        else:
            request = self.context.get("request") if self.context else None
            owner = getattr(request, "user", None) if request else None
        if owner and getattr(owner, "is_authenticated", False):
            qs = FlowProject.objects.filter(
                owner=owner, name=name, is_active=True
            )
            if self.instance is not None:
                qs = qs.exclude(pk=self.instance.pk)
            if qs.exists():
                raise serializers.ValidationError(
                    "You already have an active project with this name."
                )
        return name

    # -- Attribution JSON-list validation ------------------------------------
    # Both fields are free-form JSON lists in the DB; normalise them to a
    # predictable shape (list of dicts with only the known string keys) and
    # drop entirely-empty rows so the UI's add/remove behaviour stays clean.
    _CONTRIBUTOR_KEYS = ("name", "affiliation", "orcid", "researchmap", "role")
    _LINK_KEYS = ("label", "url")

    @staticmethod
    def _clean_rows(value, allowed_keys, field_name):
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise serializers.ValidationError(f"{field_name} must be a list.")
        cleaned = []
        for row in value:
            if not isinstance(row, dict):
                raise serializers.ValidationError(
                    f"Each {field_name} entry must be an object."
                )
            item = {k: str(row.get(k, "")).strip() for k in allowed_keys}
            if any(item.values()):  # skip rows the user left completely blank
                cleaned.append(item)
        return cleaned

    def validate_contributors(self, value):
        return self._clean_rows(value, self._CONTRIBUTOR_KEYS, "contributors")

    def validate_links(self, value):
        return self._clean_rows(value, self._LINK_KEYS, "links")


class FlowNodeSerializer(serializers.ModelSerializer):
    has_parameter_modifications = serializers.SerializerMethodField()
    modified_parameters = serializers.SerializerMethodField()
    parameter_modification_count = serializers.SerializerMethodField()

    class Meta:
        model = FlowNode
        fields = [
            "id",
            "project",
            "position_x",
            "position_y",
            "node_type",
            "data",
            "created_at",
            "updated_at",
            "has_parameter_modifications",
            "modified_parameters",
            "parameter_modification_count",
        ]
        read_only_fields = ["created_at", "updated_at", "has_parameter_modifications", "modified_parameters", "parameter_modification_count"]

    def get_has_parameter_modifications(self, obj):
        """Are there any parameter changes?"""
        return obj.has_parameter_modifications()

    def get_modified_parameters(self, obj):
        """Details of changed parameters"""
        raw = obj.get_modified_parameters()
        params = ((obj.data or {}).get("schema") or {}).get("parameters") or {}
        redacted = {}
        for key, info in raw.items():
            param = params.get(key) if isinstance(params, dict) else {}
            redacted[key] = {
                "original_value": redact_param_value(param, info.get("original_value")),
                "current_value": redact_param_value(param, info.get("current_value")),
                "modified_at": info.get("modified_at"),
            }
        return redacted

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        ret["data"] = redact_node_data(instance.data)
        return ret

    def get_parameter_modification_count(self, obj):
        """The number of parameters that were changed"""
        return obj.get_parameter_modification_count()

    def validate(self, data):
        # React Flow-style validation
        if "data" in data:
            node_data = data["data"]
            required_keys = ["label"]
            for key in required_keys:
                if key not in node_data:
                    raise serializers.ValidationError(
                        f"Node data must contain '{key}' field"
                    )

            # nodeType is required for the frontend to render correctly
            if "nodeType" not in node_data or not node_data["nodeType"]:
                raise serializers.ValidationError(
                    "Node data must contain a non-empty 'nodeType' field "
                    "(e.g. 'analysis', 'simulation'). "
                    f"Valid categories: {_valid_category_values()}"
                )

            # nodeType must match an existing category
            valid = _valid_category_values()
            if node_data["nodeType"].lower() not in valid:
                raise serializers.ValidationError(
                    f"Invalid nodeType '{node_data['nodeType']}'. "
                    f"Must be one of: {valid}"
                )

            # Validate schema structure if present
            if "schema" in node_data:
                schema = node_data["schema"]
                for section in ["inputs", "outputs", "parameters", "methods"]:
                    if section in schema and not isinstance(schema[section], dict):
                        raise serializers.ValidationError(
                            f"Node data schema.{section} must be a dictionary"
                        )
        return data


class FlowEdgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = FlowEdge
        fields = [
            "id",
            "project",
            "source_node",
            "target_node",
            "source_handle",
            "target_handle",
            "edge_data",
            "created_at",
        ]
        read_only_fields = ["created_at"]

    def validate(self, data):
        # Connections can only be made between nodes within the same project
        if data["source_node"].project != data["target_node"].project:
            raise serializers.ValidationError("Nodes must be in the same project")

        # Preventing self-reference
        if data["source_node"] == data["target_node"]:
            raise serializers.ValidationError("Cannot connect a node to itself")

        return data


# Flow-preserving serializer for the entire React Flow
class FlowDataSerializer(serializers.Serializer):
    nodes = serializers.ListField(child=serializers.DictField())
    edges = serializers.ListField(child=serializers.DictField())

    def validate_nodes(self, value):
        valid = _valid_category_values()
        for node in value:
            if "id" not in node:
                raise serializers.ValidationError("Each node must have an 'id' field")
            if (
                "position" not in node
                or "x" not in node["position"]
                or "y" not in node["position"]
            ):
                raise serializers.ValidationError(
                    "Each node must have position with x and y coordinates"
                )
            if "data" not in node:
                raise serializers.ValidationError("Each node must have a 'data' field")
            node_data = node.get("data", {})
            if "label" not in node_data:
                raise serializers.ValidationError(
                    "Each node's data must contain a 'label' field"
                )
            # nodeType is required for the frontend to render correctly
            node_type = node_data.get("nodeType")
            if not node_type:
                raise serializers.ValidationError(
                    f"Node '{node['id']}': data must contain a non-empty 'nodeType' field "
                    f"(e.g. 'analysis', 'simulation'). Valid categories: {valid}"
                )
            if node_type.lower() not in valid:
                raise serializers.ValidationError(
                    f"Node '{node['id']}': invalid nodeType '{node_type}'. "
                    f"Must be one of: {valid}"
                )
        return value

    def validate_edges(self, value):
        for edge in value:
            required_fields = ["id", "source", "target"]
            for field in required_fields:
                if field not in edge:
                    raise serializers.ValidationError(
                        f"Each edge must have a '{field}' field"
                    )
        return value


class WorkflowRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowRun
        fields = [
            "id",
            "workflow",
            "user",
            "backend",
            "status",
            "slurm_job_id",
            "exit_code",
            "stdout",
            "stderr",
            "error_message",
            "resource_requests",
            "artifacts",
            "submitted_at",
            "started_at",
            "finished_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "status",
            "slurm_job_id",
            "exit_code",
            "stdout",
            "stderr",
            "error_message",
            "artifacts",
            "submitted_at",
            "started_at",
            "finished_at",
        ]


class WorkflowRunSubmitSerializer(serializers.Serializer):
    backend = serializers.ChoiceField(
        choices=WorkflowRun.Backend.choices,
        default=WorkflowRun.Backend.JUPYTER,
    )
    resource_requests = serializers.DictField(required=False, default=dict)
