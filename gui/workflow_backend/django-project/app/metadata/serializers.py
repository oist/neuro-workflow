from rest_framework import serializers
from typing import Any, Optional, Dict, List
from .models import CustomDatabase
from app.secrets.redaction import make_secret_ref
from app.secrets.services import owner_secrets_qs


def _apply_api_key_secret_name(instance, name, request):
    if not name:
        return instance
    user = getattr(request, "user", None) if request else None
    owned = owner_secrets_qs(user).filter(name=name).first() if user and user.is_authenticated else None
    if owned is None:
        raise serializers.ValidationError({"api_key_secret_name": "Secret not found."})
    cfg = dict(instance.config or {})
    cfg["api_key_secret"] = make_secret_ref(owned.id, owned.name)
    instance.config = cfg
    instance.save(update_fields=["config"])
    return instance


class ParameterSuggestionSerializer(serializers.Serializer):
    """Serializer for parameter suggestion response."""
    value = serializers.JSONField(help_text="Suggested parameter value")
    source = serializers.CharField(help_text="Source of the suggestion (e.g., 'allen_brain', 'neuromorpho')")
    confidence = serializers.FloatField(help_text="Confidence score from 0.0 to 1.0")
    description = serializers.CharField(help_text="Explanation of the suggestion")
    species = serializers.CharField(required=False, allow_null=True, help_text="Species this value applies to")
    citation = serializers.CharField(required=False, allow_null=True, help_text="Paper or source citation")
    metadata = serializers.DictField(required=False, allow_null=True, help_text="Additional metadata")


class ParameterSuggestionRequestSerializer(serializers.Serializer):
    """Serializer for parameter suggestion request."""
    parameter_name = serializers.CharField(required=True, help_text="Name of the parameter")
    parameter_description = serializers.CharField(required=True, help_text="Description of the parameter")
    node_type = serializers.CharField(required=False, allow_null=True, help_text="Type of node this parameter belongs to")
    species = serializers.CharField(required=False, allow_null=True, help_text="Species to query for (mouse, monkey, human, etc.)")
    context = serializers.DictField(required=False, allow_null=True, help_text="Additional context (e.g., brain region, cell type)")


class ParameterSuggestionResponseSerializer(serializers.Serializer):
    """Serializer for the full parameter suggestion API response."""
    suggestions = ParameterSuggestionSerializer(many=True, help_text="List of parameter suggestions")
    parameter_name = serializers.CharField(help_text="Name of the parameter queried")
    parameter_description = serializers.CharField(help_text="Description of the parameter queried")
    species = serializers.CharField(required=False, allow_null=True, help_text="Species filter applied")


class CustomDatabaseSerializer(serializers.ModelSerializer):
    """Serializer for CustomDatabase model."""

    api_key = serializers.CharField(
        write_only=True,
        required=False,
        allow_blank=True,
        allow_null=True,
    )
    api_key_is_set = serializers.BooleanField(read_only=True)
    api_key_secret_name = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta:
        model = CustomDatabase
        fields = [
            'id', 'name', 'description', 'base_url', 'api_key',
            'api_key_is_set', 'api_key_secret_name',
            'config', 'adapter_type', 'is_active', 'is_verified',
            'last_tested', 'test_result', 'test_error',
            'created_by', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'created_by', 'is_verified', 'last_tested', 'test_result', 'test_error', 'api_key_is_set']

    def validate_base_url(self, value):
        """Validate base URL format."""
        if not value.startswith(('http://', 'https://')):
            raise serializers.ValidationError("Base URL must start with http:// or https://")
        return value

    def create(self, validated_data):
        raw_key = validated_data.pop("api_key", None)
        secret_name = validated_data.pop("api_key_secret_name", None)
        instance = super().create(validated_data)
        if raw_key:
            instance.set_api_key(raw_key)
            instance.save(
                update_fields=[
                    "api_key_wrapped_dek",
                    "api_key_ciphertext",
                    "api_key_nonce",
                    "api_key_key_version",
                ]
            )
        request = self.context.get("request")
        _apply_api_key_secret_name(instance, secret_name, request)
        return instance

    def update(self, instance, validated_data):
        raw_key = validated_data.pop("api_key", None)
        secret_name = validated_data.pop("api_key_secret_name", None)
        instance = super().update(instance, validated_data)
        if raw_key is not None:
            instance.set_api_key(raw_key)
            instance.save(
                update_fields=[
                    "api_key_wrapped_dek",
                    "api_key_ciphertext",
                    "api_key_nonce",
                    "api_key_key_version",
                ]
            )
        request = self.context.get("request")
        _apply_api_key_secret_name(instance, secret_name, request)
        return instance


class CustomDatabaseCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new CustomDatabase."""

    api_key = serializers.CharField(
        write_only=True,
        required=False,
        allow_blank=True,
        allow_null=True,
    )
    api_key_secret_name = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta:
        model = CustomDatabase
        fields = [
            'name', 'description', 'base_url', 'api_key', 'api_key_secret_name',
            'config', 'adapter_type', 'is_active'
        ]

    def validate_base_url(self, value):
        """Validate base URL format."""
        if not value.startswith(('http://', 'https://')):
            raise serializers.ValidationError("Base URL must start with http:// or https://")
        return value

    def create(self, validated_data):
        raw_key = validated_data.pop("api_key", None)
        secret_name = validated_data.pop("api_key_secret_name", None)
        instance = super().create(validated_data)
        if raw_key:
            instance.set_api_key(raw_key)
            instance.save(
                update_fields=[
                    "api_key_wrapped_dek",
                    "api_key_ciphertext",
                    "api_key_nonce",
                    "api_key_key_version",
                ]
            )
        request = self.context.get("request")
        _apply_api_key_secret_name(instance, secret_name, request)
        return instance


class DatabaseConnectionTestSerializer(serializers.Serializer):
    """Serializer for testing database connection."""
    base_url = serializers.URLField(required=True, help_text="Base URL of the database")
    api_key = serializers.CharField(required=False, allow_blank=True, help_text="API key if required")
    config = serializers.DictField(required=False, default=dict, help_text="Additional configuration")
