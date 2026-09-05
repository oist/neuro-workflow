from django.db import models
from django.contrib.auth.models import User
from app.workflow.models import FlowProject
import uuid


class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, default="New Conversation")
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="chat_conversations",
        null=True, blank=True,
    )
    project = models.ForeignKey(
        FlowProject, on_delete=models.SET_NULL, related_name="chat_conversations",
        null=True, blank=True,
    )
    system_prompt = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "chat_conversations"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} ({self.id})"


class Message(models.Model):
    ROLE_CHOICES = [
        ("system", "System"),
        ("user", "User"),
        ("assistant", "Assistant"),
        ("tool", "Tool"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages",
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField(blank=True, default="")
    tool_calls = models.JSONField(null=True, blank=True)
    tool_call_id = models.CharField(max_length=255, blank=True, default="")
    tool_name = models.CharField(max_length=255, blank=True, default="")
    raw_response = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chat_messages"
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.content[:50]}"

    def to_openai_format(self):
        """Convert to OpenAI API message format."""
        msg = {"role": self.role, "content": self.content}
        if self.role == "assistant" and self.tool_calls:
            msg["tool_calls"] = self.tool_calls
            if not self.content:
                msg["content"] = None
        if self.role == "tool":
            msg["tool_call_id"] = self.tool_call_id
            msg["name"] = self.tool_name
        return msg


class ChatProfile(models.Model):
    """Per-user preset for the browser chat: which MCP tools the assistant may
    use and an optional system prompt override."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="chat_profiles",
    )
    name = models.CharField(max_length=100)
    # Explicit allowlist of MCP tool names. An empty list disables tools.
    allowed_tools = models.JSONField(default=list, blank=True)
    # Empty means "use the default assistant prompt".
    system_prompt = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "chat_profiles"
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "name"], name="chat_profile_unique_user_name",
            )
        ]

    def __str__(self):
        return f"{self.name} ({self.user_id})"
