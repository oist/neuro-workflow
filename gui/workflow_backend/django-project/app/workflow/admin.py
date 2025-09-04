from django.contrib import admin
from .models import FlowProject, FlowNode, FlowEdge


@admin.register(FlowProject)
class FlowProjectAdmin(admin.ModelAdmin):
    list_display = ["name", "owner", "is_active", "created_at", "updated_at"]
    list_filter = ["is_active", "created_at"]
    search_fields = ["name", "description", "owner__username"]
    ordering = ["-created_at"]


@admin.register(FlowNode)
class FlowNodeAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "project",
        "node_type",
        "position_x",
        "position_y",
        "created_at",
    ]
    list_filter = ["node_type", "created_at"]
    search_fields = ["id", "node_type", "project__name"]
    ordering = ["created_at"]


@admin.register(FlowEdge)
class FlowEdgeAdmin(admin.ModelAdmin):
    list_display = ["id", "project", "source_node", "target_node", "created_at"]
    list_filter = ["created_at"]
    search_fields = ["id", "source_node__id", "target_node__id", "project__name"]
    ordering = ["created_at"]
