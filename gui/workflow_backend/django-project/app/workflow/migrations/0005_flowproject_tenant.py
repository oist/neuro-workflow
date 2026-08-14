from django.db import migrations, models


def ensure_groups(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    for name in ("nw-internal", "nw-hackathon", "node-reviewers"):
        Group.objects.get_or_create(name=name)


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0004_flowproject_attribution"),
    ]

    operations = [
        migrations.AddField(
            model_name="flowproject",
            name="tenant",
            field=models.CharField(
                choices=[("internal", "Internal"), ("hackathon", "Hackathon")],
                db_index=True,
                default="internal",
                max_length=16,
            ),
        ),
        migrations.RunPython(ensure_groups, migrations.RunPython.noop),
    ]
