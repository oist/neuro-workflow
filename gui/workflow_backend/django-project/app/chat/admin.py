from django.contrib import admin
from .models import ChatProfile, Conversation, Message, NotebookAgentPolicy


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ["id", "role", "content", "tool_calls", "tool_call_id", "tool_name", "created_at"]


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ["id", "title", "source", "user", "project", "is_active", "created_at", "updated_at"]
    list_filter = ["source", "is_active", "created_at"]
    search_fields = ["title"]
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["id", "conversation", "role", "content_preview", "created_at"]
    list_filter = ["role", "created_at"]

    def content_preview(self, obj):
        return obj.content[:100] if obj.content else ""
    content_preview.short_description = "Content"


@admin.register(ChatProfile)
class ChatProfileAdmin(admin.ModelAdmin):
    list_display = ["name", "is_default", "created_by", "updated_at"]
    search_fields = ["name"]


@admin.register(NotebookAgentPolicy)
class NotebookAgentPolicyAdmin(admin.ModelAdmin):
    list_display = ["__str__", "updated_by", "updated_at"]
    readonly_fields = ["updated_by", "updated_at"]

    def has_add_permission(self, request):
        return not NotebookAgentPolicy.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False

    def save_model(self, request, obj, form, change):
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
