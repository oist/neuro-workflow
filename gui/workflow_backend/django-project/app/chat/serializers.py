from rest_framework import serializers
from .models import ChatProfile, Conversation, Message, NotebookAgentPolicy
from .services.notebook_policy import (
    CATEGORIES,
    WORKFLOW_READ_TOOLS,
    validate_ceilings,
)

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
    # Chat profile (shared, staff-managed MCP tool allowlist + system prompt
    # override). An explicit id always wins. Omitted / null: staff get all
    # tools + the default prompt; non-staff get the admin default profile when
    # one is set (see ChatStreamView), otherwise all tools.
    profile_id = serializers.UUIDField(required=False, allow_null=True)
    # Model id from GET /api/chat/models/. Omitted / null: the first (default)
    # model of that list.
    model = serializers.CharField(required=False, allow_blank=True, allow_null=True)


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


class NotebookAgentPolicySerializer(serializers.ModelSerializer):
    """Admin ceilings of the notebook agent, plus what the kernel needs to
    apply them (the category names and the read-only workflow tools)."""

    categories = serializers.SerializerMethodField()
    workflow_read_tools = serializers.SerializerMethodField()

    class Meta:
        model = NotebookAgentPolicy
        fields = [
            "category_ceilings",
            "tool_ceilings",
            "categories",
            "workflow_read_tools",
            "updated_at",
        ]
        read_only_fields = ["updated_at"]

    def get_categories(self, obj):
        return list(CATEGORIES)

    def get_workflow_read_tools(self, obj):
        return list(WORKFLOW_READ_TOOLS)

    def validate(self, attrs):
        instance = self.instance
        errors = validate_ceilings(
            attrs.get("category_ceilings", instance.category_ceilings),
            attrs.get("tool_ceilings", instance.tool_ceilings),
        )
        if errors:
            raise serializers.ValidationError(errors)
        return attrs


class NotebookTokenSerializer(serializers.Serializer):
    """``project_id`` of the notebook token relay (POST body / MCP proxy input)."""

    project_id = serializers.UUIDField()


class NotebookTurnMessageSerializer(serializers.Serializer):
    """One message of a notebook agent turn, in the OpenAI shape of ``Message``."""

    role = serializers.ChoiceField(choices=["user", "assistant", "tool"])
    content = serializers.CharField(
        allow_blank=True, default="", trim_whitespace=False
    )
    tool_calls = serializers.ListField(
        child=serializers.DictField(), required=False, allow_null=True
    )
    tool_call_id = serializers.CharField(
        max_length=255, required=False, allow_blank=True, default=""
    )
    tool_name = serializers.CharField(
        max_length=255, required=False, allow_blank=True, default=""
    )
    is_error = serializers.BooleanField(required=False, default=False)


class NotebookTurnSerializer(serializers.Serializer):
    """One user turn recorded by the notebook agent (POST body)."""

    project_id = serializers.UUIDField()
    # Omitted / null / unknown: the turn starts a new conversation.
    conversation_id = serializers.UUIDField(required=False, allow_null=True)
    model = serializers.CharField(
        max_length=255, required=False, allow_blank=True, allow_null=True
    )
    # Claude Agent SDK session id, kept for resuming the conversation later.
    session_id = serializers.CharField(
        max_length=255, required=False, allow_blank=True, allow_null=True
    )
    messages = NotebookTurnMessageSerializer(many=True, allow_empty=False)


class NotebookMessageSerializer(MessageSerializer):
    is_error = serializers.SerializerMethodField()

    class Meta(MessageSerializer.Meta):
        fields = MessageSerializer.Meta.fields + ["is_error"]

    def get_is_error(self, obj):
        return bool((obj.raw_response or {}).get("is_error"))


class NotebookConversationListSerializer(serializers.ModelSerializer):
    """A saved notebook agent conversation, without its messages.

    Expects ``message_count`` annotated on the queryset and the request in the
    context (for ``can_delete``).
    """

    user_label = serializers.SerializerMethodField()
    model = serializers.SerializerMethodField()
    message_count = serializers.IntegerField(read_only=True)
    can_delete = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "project",
            "user_label",
            "model",
            "message_count",
            "created_at",
            "updated_at",
            "can_delete",
        ]

    def get_user_label(self, obj):
        # username is the Keycloak sub (a UUID), so prefer a readable name.
        user = obj.user
        if user is None:
            return ""
        return user.get_full_name() or user.email or user.username

    def get_model(self, obj):
        return (obj.metadata or {}).get("model", "")

    def get_can_delete(self, obj):
        user_id = self.context["request"].user.id
        return user_id in (obj.project.owner_id, obj.user_id)


class NotebookConversationSerializer(NotebookConversationListSerializer):
    messages = NotebookMessageSerializer(many=True, read_only=True)

    class Meta(NotebookConversationListSerializer.Meta):
        fields = NotebookConversationListSerializer.Meta.fields + ["messages"]
