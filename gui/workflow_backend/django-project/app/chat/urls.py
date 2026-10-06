from django.urls import path, re_path
from .views import (
    ConversationListCreateView,
    ConversationDetailView,
    ChatStreamView,
    ChatModelsView,
    NotebookMCPToolsView,
    NotebookMCPCallView,
    NotebookTokenView,
    AnthropicProxyView,
    ChatProfileListCreateView,
    ChatProfileDetailView,
)

urlpatterns = [
    path("conversations/", ConversationListCreateView.as_view(), name="chat-conversations"),
    path("conversations/<uuid:conversation_id>/", ConversationDetailView.as_view(), name="chat-conversation-detail"),
    path("stream/", ChatStreamView.as_view(), name="chat-stream"),
    path("models/", ChatModelsView.as_view(), name="chat-models"),
    path("mcp-tools/", NotebookMCPToolsView.as_view(), name="chat-notebook-mcp-tools"),
    path("mcp-call/", NotebookMCPCallView.as_view(), name="chat-notebook-mcp-call"),
    # Browser -> backend relay of the Keycloak token used by the kernel MCP proxies.
    path("notebook-token/", NotebookTokenView.as_view(), name="chat-notebook-token"),
    path("profiles/", ChatProfileListCreateView.as_view(), name="chat-profiles"),
    path("profiles/<uuid:profile_id>/", ChatProfileDetailView.as_view(), name="chat-profile-detail"),
    # Anthropic API passthrough for the in-kernel Claude agent (key stays here).
    re_path(r"^anthropic/(?P<subpath>.*)$", AnthropicProxyView.as_view(), name="chat-anthropic-proxy"),
]
