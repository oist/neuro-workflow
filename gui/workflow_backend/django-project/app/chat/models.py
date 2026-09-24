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


class NotebookToken(models.Model):
    """Browser Keycloak access token relayed for a project's notebook kernel.

    Written by the frontend while the project's Jupyter tab is open and used
    server-side by the notebook MCP proxies on the kernel's behalf; the kernel
    never receives it. ``hub_user`` is the JupyterHub user of the relaying
    user's Jupyter space, so only kernels in that space can use the token.
    Access tokens only — never refresh tokens.
    """

    project = models.OneToOneField(
        FlowProject,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="notebook_token",
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="notebook_tokens"
    )
    hub_user = models.CharField(max_length=64)
    access_token = models.TextField()
    expires_at = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "chat_notebook_tokens"
