from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("nw_secrets", "0001_initial"),
    ]

    operations = [
        migrations.RemoveConstraint(
            model_name="usersecret",
            name="uniq_user_secret_owner_name",
        ),
        migrations.AddConstraint(
            model_name="usersecret",
            constraint=models.UniqueConstraint(
                condition=models.Q(revoked_at__isnull=True),
                fields=("owner", "name"),
                name="uniq_user_secret_owner_name_active",
            ),
        ),
    ]
