from django.db import migrations, models


def _add_column(name, ddl_type, default_sql):
    """Idempotent ADD COLUMN + drop the DB-level default (kept only for the
    backfill of existing rows), mirroring migration 0003's approach so this is
    safe to run even if a column was already created out-of-band."""
    return migrations.RunSQL(
        sql=f"""
        ALTER TABLE flow_projects
        ADD COLUMN IF NOT EXISTS {name} {ddl_type} NOT NULL DEFAULT {default_sql};
        ALTER TABLE flow_projects
        ALTER COLUMN {name} DROP DEFAULT;
        """,
        reverse_sql=f"""
        ALTER TABLE flow_projects
        DROP COLUMN IF EXISTS {name};
        """,
    )


class Migration(migrations.Migration):

    dependencies = [
        ("workflow", "0003_flowproject_reference_hpc_target"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                _add_column("doi", "varchar(255)", "''"),
                _add_column("data_source", "text", "''"),
                _add_column("license", "varchar(255)", "''"),
                _add_column("funding", "text", "''"),
                _add_column("contact_email", "varchar(255)", "''"),
                _add_column("links", "jsonb", "'[]'::jsonb"),
                _add_column("contributors", "jsonb", "'[]'::jsonb"),
            ],
            state_operations=[
                migrations.AddField(
                    model_name="flowproject",
                    name="doi",
                    field=models.CharField(blank=True, default="", max_length=255),
                ),
                migrations.AddField(
                    model_name="flowproject",
                    name="data_source",
                    field=models.TextField(blank=True, default=""),
                ),
                migrations.AddField(
                    model_name="flowproject",
                    name="license",
                    field=models.CharField(blank=True, default="", max_length=255),
                ),
                migrations.AddField(
                    model_name="flowproject",
                    name="funding",
                    field=models.TextField(blank=True, default=""),
                ),
                migrations.AddField(
                    model_name="flowproject",
                    name="contact_email",
                    field=models.CharField(blank=True, default="", max_length=255),
                ),
                migrations.AddField(
                    model_name="flowproject",
                    name="links",
                    field=models.JSONField(blank=True, default=list),
                ),
                migrations.AddField(
                    model_name="flowproject",
                    name="contributors",
                    field=models.JSONField(blank=True, default=list),
                ),
            ],
        ),
    ]
