from rest_framework import serializers
from .models import ChatProfile, Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = [
            "id",
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "tool_name",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "project",
            "system_prompt",
            "metadata",
            "is_active",
            "created_at",
            "updated_at",
            "messages",
            "message_count",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_message_count(self, obj):
        return obj.messages.count()


class ConversationListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing conversations (without messages)."""
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "project",
            "is_active",
            "created_at",
            "updated_at",
            "message_count",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_message_count(self, obj):
        return obj.messages.count()


class SendMessageSerializer(serializers.Serializer):
    message = serializers.CharField()
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    project_id = serializers.UUIDField(required=False, allow_null=True)
    # Snapshot of the brain viewer the user is currently looking at, injected as
    # an ephemeral system message (not stored in conversation history).
    viewer_context = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
    # Chat profile (per-user MCP tool allowlist + system prompt override).
    # Omitted / null means the default behaviour: all tools, default prompt.
    profile_id = serializers.UUIDField(required=False, allow_null=True)


class ChatProfileSerializer(serializers.ModelSerializer):
    allowed_tools = serializers.ListField(
        child=serializers.CharField(max_length=255), allow_empty=True
    )

    class Meta:
        model = ChatProfile
        fields = [
            "id",
            "name",
            "allowed_tools",
            "system_prompt",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def validate_allowed_tools(self, value):
        # Drop duplicates while keeping the submitted order.
        return list(dict.fromkeys(value))

    def validate_name(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError("Name is required.")
        qs = ChatProfile.objects.filter(
            user=self.context["request"].user, name=value
        )
        if self.instance is not None:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError(
                "You already have a profile with this name."
            )
        return value
