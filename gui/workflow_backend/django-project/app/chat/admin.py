from django.contrib import admin
from .models import ChatProfile, Conversation, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    readonly_fields = ["id", "role", "content", "tool_calls", "tool_call_id", "tool_name", "created_at"]


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ["id", "title", "user", "project", "is_active", "created_at", "updated_at"]
    list_filter = ["is_active", "created_at"]
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
    list_display = ["name", "user", "updated_at"]
    search_fields = ["name"]
