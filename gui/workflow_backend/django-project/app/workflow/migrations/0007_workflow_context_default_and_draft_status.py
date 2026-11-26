import app.workflow.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0006_tenant_project_community"),
    ]

    operations = [
        migrations.AlterField(
            model_name="flowproject",
            name="workflow_context",
            field=models.JSONField(
                blank=True, default=app.workflow.models._default_workflow_context
            ),
        ),
        migrations.AlterField(
            model_name="workflowrun",
            name="status",
            field=models.CharField(
                choices=[
                    ("draft", "Draft"),
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("cancelled", "Cancelled"),
                ],
                default="pending",
                max_length=20,
            ),
        ),
    ]
