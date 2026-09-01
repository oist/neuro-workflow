import uuid

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="HarvestedRecord",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("oai_identifier", models.CharField(max_length=255, unique=True)),
                ("datestamp", models.CharField(blank=True, default="", max_length=64)),
                ("set_specs", models.JSONField(blank=True, default=list)),
                ("deleted", models.BooleanField(default=False)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("files", models.JSONField(blank=True, default=list)),
                ("search_text", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "db_table": "harvested_records",
                "ordering": ["-datestamp", "oai_identifier"],
            },
        ),
        migrations.CreateModel(
            name="HarvestRun",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[("success", "Success"), ("error", "Error")],
                        max_length=16,
                    ),
                ),
                (
                    "mode",
                    models.CharField(
                        choices=[("incremental", "Incremental"), ("full", "Full")],
                        max_length=16,
                    ),
                ),
                (
                    "from_datestamp",
                    models.CharField(blank=True, default="", max_length=64),
                ),
                ("watermark", models.CharField(blank=True, default="", max_length=64)),
                ("records_seen", models.IntegerField(default=0)),
                ("records_deleted", models.IntegerField(default=0)),
                ("error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField()),
                ("finished_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "harvest_runs",
                "ordering": ["-finished_at"],
            },
        ),
    ]
