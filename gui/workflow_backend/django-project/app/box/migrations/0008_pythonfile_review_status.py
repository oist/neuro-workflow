from django.db import migrations, models


def backfill_review_status(apps, schema_editor):
    PythonFile = apps.get_model("box", "PythonFile")
    PythonFile.objects.filter(status="submitted").update(review_status="in_review")
    PythonFile.objects.filter(status="approved").update(review_status="reviewed")
    PythonFile.objects.filter(uploaded_by__isnull=True).update(
        status="public", review_status="reviewed"
    )
    PythonFile.objects.filter(review_status="").update(review_status="unreviewed")


class Migration(migrations.Migration):

    dependencies = [
        ("box", "0007_tenant_project_community"),
    ]

    operations = [
        migrations.AddField(
            model_name="pythonfile",
            name="review_status",
            field=models.CharField(
                choices=[
                    ("unreviewed", "Unreviewed"),
                    ("in_review", "In review"),
                    ("reviewed", "Reviewed"),
                ],
                db_index=True,
                default="unreviewed",
                max_length=16,
            ),
        ),
        migrations.RunPython(backfill_review_status, migrations.RunPython.noop),
    ]
