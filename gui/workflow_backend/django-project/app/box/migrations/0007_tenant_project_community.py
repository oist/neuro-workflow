from django.db import migrations, models


def remap_node_tenants(apps, schema_editor):
    PythonFile = apps.get_model("box", "PythonFile")
    NodeAuditLog = apps.get_model("box", "NodeAuditLog")
    PythonFile.objects.filter(tenant="internal").update(tenant="project")
    PythonFile.objects.filter(tenant="hackathon").update(tenant="community")
    NodeAuditLog.objects.filter(tenant="internal").update(tenant="project")
    NodeAuditLog.objects.filter(tenant="hackathon").update(tenant="community")


class Migration(migrations.Migration):

    dependencies = [
        ("box", "0006_pythonfile_tenant_governance"),
    ]

    operations = [
        migrations.AlterField(
            model_name="pythonfile",
            name="tenant",
            field=models.CharField(
                choices=[("project", "Project"), ("community", "Community")],
                db_index=True,
                default="project",
                max_length=16,
            ),
        ),
        migrations.AlterField(
            model_name="nodeauditlog",
            name="tenant",
            field=models.CharField(
                choices=[("project", "Project"), ("community", "Community")],
                db_index=True,
                default="project",
                max_length=16,
            ),
        ),
        migrations.RunPython(remap_node_tenants, migrations.RunPython.noop),
    ]
