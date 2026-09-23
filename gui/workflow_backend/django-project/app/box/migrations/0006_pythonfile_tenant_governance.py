import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


def backfill_node_governance(apps, schema_editor):
    PythonFile = apps.get_model("box", "PythonFile")
    PythonFile.objects.filter(uploaded_by__isnull=True).update(
        status="public", tenant="internal"
    )
    PythonFile.objects.filter(uploaded_by__isnull=False).update(status="private")


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("box", "0005_alter_pythonfile_category"),
    ]

    operations = [
        migrations.AddField(
            model_name="pythonfile",
            name="tenant",
            field=models.CharField(
                choices=[("internal", "Internal"), ("hackathon", "Hackathon")],
                db_index=True,
                default="internal",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="pythonfile",
            name="status",
            field=models.CharField(
                choices=[
                    ("private", "Private"),
                    ("submitted", "Submitted"),
                    ("approved", "Approved"),
                    ("public", "Public"),
                ],
                db_index=True,
                default="private",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="pythonfile",
            name="submitted_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="pythonfile",
            name="reviewed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="pythonfile",
            name="review_comment",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="pythonfile",
            name="reviewed_by",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="reviewed_nodes",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="pythonfile",
            name="file_hash",
            field=models.CharField(default="default", max_length=64),
        ),
        migrations.AddConstraint(
            model_name="pythonfile",
            constraint=models.UniqueConstraint(
                fields=("file_hash", "tenant"),
                name="box_pythonfile_hash_tenant_uniq",
            ),
        ),
        migrations.CreateModel(
            name="NodeAuditLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("action", models.CharField(max_length=32)),
                ("from_status", models.CharField(blank=True, default="", max_length=16)),
                ("to_status", models.CharField(blank=True, default="", max_length=16)),
                ("comment", models.TextField(blank=True, default="")),
                (
                    "tenant",
                    models.CharField(
                        choices=[
                            ("internal", "Internal"),
                            ("hackathon", "Hackathon"),
                        ],
                        db_index=True,
                        default="internal",
                        max_length=16,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "actor",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="node_audit_events",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "python_file",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="audit_logs",
                        to="box.pythonfile",
                    ),
                ),
            ],
            options={
                "db_table": "box_nodeauditlog",
                "ordering": ["-created_at"],
            },
        ),
        migrations.RunPython(backfill_node_governance, migrations.RunPython.noop),
    ]
