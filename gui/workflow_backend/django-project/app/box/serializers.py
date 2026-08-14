from rest_framework import serializers
from .models import PythonFile, NODE_CATEGORIES
from .models import get_categories

class PythonFileUploadSerializer(serializers.Serializer):
    """File upload serializer"""
    node_categories = get_categories()

    file = serializers.FileField()
    name = serializers.CharField(max_length=255, required=False)
    description = serializers.CharField(required=False, allow_blank=True)
    category = serializers.ChoiceField(choices=node_categories, default='analysis')

    def validate_file(self, value):
        """File validation"""
        # File extension check
        if not value.name.endswith(".py"):
            raise serializers.ValidationError("Only Python files (.py) are allowed.")

        # File size check (10MB limit)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("File size must be less than 10MB.")

        return value


class PythonFileSerializer(serializers.ModelSerializer):
    """PythonFile serializer for detailed display"""

    uploaded_by_name = serializers.CharField(
        source="uploaded_by.username", read_only=True
    )
    node_classes_count = serializers.SerializerMethodField()

    class Meta:
        model = PythonFile
        fields = [
            "id",
            "name",
            "description",
            "category",
            "file",
            "uploaded_by",
            "uploaded_by_name",
            "file_size",
            "is_analyzed",
            "analysis_error",
            "node_classes_count",
            "created_at",
            "updated_at",
            "tenant",
            "status",
            "review_comment",
        ]
        read_only_fields = [
            "id",
            "uploaded_by",
            "file_size",
            "created_at",
            "updated_at",
            "tenant",
            "status",
        ]

    def get_node_classes_count(self, obj):
        """Returns the number of node classes"""
        return len(obj.node_classes) if obj.node_classes else 0
