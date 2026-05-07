from django.db import models
from django.contrib.auth.models import User
import uuid


class FlowProject(models.Model):
    class Visibility(models.TextChoices):
        PRIVATE = "private", "Private"
        PUBLIC = "public", "Public"

    class HpcTarget(models.TextChoices):
        RIKEN = "riken", "Riken"
        FUGAKU = "fugaku", "Fugaku"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    workflow_context = models.JSONField(default=dict, blank=True)
    owner = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="flow_projects"
    )
    visibility = models.CharField(
        max_length=16,
        choices=Visibility.choices,
        default=Visibility.PRIVATE,
        db_index=True,
    )
    reference = models.TextField(blank=True, default="")
    hpc_target = models.CharField(
        max_length=32,
        choices=HpcTarget.choices,
        blank=True,
        default="",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "flow_projects"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class FlowNode(models.Model):
    id = models.CharField(max_length=255, primary_key=True)  # React Flow node ID
    project = models.ForeignKey(
        FlowProject, on_delete=models.CASCADE, related_name="nodes"
    )
    position_x = models.FloatField()
    position_y = models.FloatField()
    node_type = models.CharField(max_length=100)
    data = models.JSONField()  # React Flow, The entire data object
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "flow_nodes"
        ordering = ["created_at"]

    def __str__(self):
        return f"Node {self.id} in {self.project.name}"

    def has_parameter_modifications(self):
        """Check for parameter changes"""
        return self.data.get("has_parameter_modifications", False)

    def get_modified_parameters(self):
        """Get the list of changed parameters"""
        modifications = self.data.get("parameter_modifications", {})
        return {
            param_key: {
                "original_value": param_info.get("original_value"),
                "current_value": param_info.get("current_value"),
                "modified_at": param_info.get("modified_at")
            }
            for param_key, param_info in modifications.items()
            if param_info.get("is_modified", False)
        }

    def get_parameter_modification_count(self):
        """Get the number of changed parameters"""
        return len(self.get_modified_parameters())


class FlowEdge(models.Model):
    id = models.CharField(max_length=255, primary_key=True)  # React Flow edge ID
    project = models.ForeignKey(
        FlowProject, on_delete=models.CASCADE, related_name="edges"
    )
    source_node = models.ForeignKey(
        FlowNode, on_delete=models.CASCADE, related_name="outgoing_edges"
    )
    target_node = models.ForeignKey(
        FlowNode, on_delete=models.CASCADE, related_name="incoming_edges"
    )
    source_handle = models.CharField(max_length=255, null=True, blank=True)
    target_handle = models.CharField(max_length=255, null=True, blank=True)
    edge_data = models.JSONField(default=dict)  # Additional edge settings
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "flow_edges"
        ordering = ["created_at"]

    def __str__(self):
        return f"Edge {self.id}: {self.source_node.id} -> {self.target_node.id}"


class WorkflowRun(models.Model):
    """Tracks a single execution of a workflow (local or remote)."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    class Backend(models.TextChoices):
        LOCAL = "local", "Local"
        SLURM = "slurm", "Slurm"
        JUPYTER = "jupyter", "Jupyter"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(
        FlowProject, on_delete=models.CASCADE, related_name="runs"
    )
    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name="workflow_runs"
    )
    backend = models.CharField(
        max_length=20, choices=Backend.choices, default=Backend.JUPYTER
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    slurm_job_id = models.CharField(max_length=64, blank=True, default="")
    exit_code = models.IntegerField(null=True, blank=True)

    stdout = models.TextField(blank=True, default="")
    stderr = models.TextField(blank=True, default="")
    error_message = models.TextField(blank=True, default="")

    resource_requests = models.JSONField(default=dict, blank=True)
    remote_run_dir = models.CharField(max_length=512, blank=True, default="")
    artifacts = models.JSONField(default=dict, blank=True)

    submitted_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "workflow_runs"
        ordering = ["-submitted_at"]

    def __str__(self):
        return f"Run {self.id} [{self.status}] of {self.workflow.name}"
