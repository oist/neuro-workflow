from django.db import migrations, models


def remap_tenants_and_groups(apps, schema_editor):
    FlowProject = apps.get_model("workflow", "FlowProject")
    FlowProject.objects.filter(tenant="internal").update(tenant="project")
    FlowProject.objects.filter(tenant="hackathon").update(tenant="community")

    Group = apps.get_model("auth", "Group")
    User = apps.get_model("auth", "User")
    for old_name, new_name in (
        ("nw-internal", "nw-project"),
        ("nw-hackathon", "nw-community"),
    ):
        old_group, _ = Group.objects.get_or_create(name=old_name)
        new_group, _ = Group.objects.get_or_create(name=new_name)
        user_ids = list(old_group.user_set.values_list("pk", flat=True))
        if user_ids:
            new_group.user_set.add(*User.objects.filter(pk__in=user_ids))
    Group.objects.get_or_create(name="node-reviewers")


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0005_flowproject_tenant"),
    ]

    operations = [
        migrations.AlterField(
            model_name="flowproject",
            name="tenant",
            field=models.CharField(
                choices=[("project", "Project"), ("community", "Community")],
                db_index=True,
                default="project",
                max_length=16,
            ),
        ),
        migrations.RunPython(remap_tenants_and_groups, migrations.RunPython.noop),
    ]
