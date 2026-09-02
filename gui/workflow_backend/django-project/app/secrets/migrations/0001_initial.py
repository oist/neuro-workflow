# Generated for Task 9 owner secret store

import django.db.models.deletion
import uuid
from django.conf import settings
from django.db import migrations, models

import app.secrets.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="UserSecret",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=64, validators=[app.secrets.models.validate_secret_name])),
                ("description", models.TextField(blank=True, default="")),
                ("wrapped_dek", models.BinaryField()),
                ("ciphertext", models.BinaryField()),
                ("nonce", models.BinaryField()),
                ("key_version", models.PositiveSmallIntegerField(default=1)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("last_used_at", models.DateTimeField(blank=True, null=True)),
                ("revoked_at", models.DateTimeField(blank=True, null=True)),
                (
                    "owner",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="user_secrets",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={"ordering": ["name"]},
        ),
        migrations.CreateModel(
            name="SecretAuditEvent",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("secret_id_snapshot", models.UUIDField(blank=True, null=True)),
                ("secret_name", models.CharField(max_length=64)),
                (
                    "action",
                    models.CharField(
                        choices=[
                            ("create", "Create"),
                            ("rotate", "Rotate"),
                            ("delete", "Delete"),
                            ("inject", "Inject"),
                            ("denied", "Denied"),
                        ],
                        max_length=16,
                    ),
                ),
                ("ip_address", models.GenericIPAddressField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "actor",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="secret_audit_actions",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "owner",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="secret_audit_events",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "secret",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="audit_events",
                        to="nw_secrets.usersecret",
                    ),
                ),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.AddIndex(
            model_name="usersecret",
            index=models.Index(fields=["owner", "revoked_at"], name="nw_secrets__owner_i_revoked_idx"),
        ),
        migrations.AddConstraint(
            model_name="usersecret",
            constraint=models.UniqueConstraint(fields=("owner", "name"), name="uniq_user_secret_owner_name"),
        ),
        migrations.AddIndex(
            model_name="secretauditevent",
            index=models.Index(fields=["owner", "created_at"], name="nw_secrets__owner_created_idx"),
        ),
    ]
