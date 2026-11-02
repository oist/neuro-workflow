from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0003_notebooktoken"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversation",
            name="source",
            field=models.CharField(
                choices=[("browser", "Browser chat"), ("notebook", "Notebook agent")],
                db_index=True,
                default="browser",
                max_length=16,
            ),
        ),
    ]
