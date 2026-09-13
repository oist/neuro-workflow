from rest_framework import serializers
from .models import ChatProfile, Conversation, Message

# Keep in sync with ChatProfileModal textarea maxLength.
SYSTEM_PROMPT_MAX_LENGTH = 16000


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
    # Declared explicitly so the model's unique=True does not add DRF's
    # UniqueValidator; validate_name gives the friendlier message and the DB
    # constraint (IntegrityError -> 400 in the view) covers races.
    name = serializers.CharField(max_length=100)
    allowed_tools = serializers.ListField(
        child=serializers.CharField(max_length=255, allow_blank=False),
        allow_empty=True,
    )
    system_prompt = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=SYSTEM_PROMPT_MAX_LENGTH,
    )
    is_default = serializers.BooleanField(required=False)

    class Meta:
        model = ChatProfile
        fields = [
            "id",
            "name",
            "allowed_tools",
            "system_prompt",
            "is_default",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def validate_allowed_tools(self, value):
        cleaned = []
        for name in value:
            stripped = name.strip()
            if not stripped:
                raise serializers.ValidationError("Tool names cannot be empty.")
            cleaned.append(stripped)
        # Drop duplicates while keeping the submitted order.
        return list(dict.fromkeys(cleaned))

    def validate_name(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError("Name is required.")
        qs = ChatProfile.objects.filter(name=value)
        if self.instance is not None:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError(
                "A profile with this name already exists."
            )
        return value

    def _clear_other_defaults(self, validated_data, instance=None):
        # Only one profile can be the default; the view wraps save() in a
        # transaction so this and the insert/update commit together.
        if validated_data.get("is_default"):
            qs = ChatProfile.objects.filter(is_default=True)
            if instance is not None:
                qs = qs.exclude(pk=instance.pk)
            qs.update(is_default=False)

    def create(self, validated_data):
        self._clear_other_defaults(validated_data)
        return super().create(validated_data)

    def update(self, instance, validated_data):
        self._clear_other_defaults(validated_data, instance)
        return super().update(instance, validated_data)
