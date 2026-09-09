# Encrypt CustomDatabase.api_key at rest with the owner secret KEK.

from django.db import migrations, models


def encrypt_existing_api_keys(apps, schema_editor):
    CustomDatabase = apps.get_model("metadata", "CustomDatabase")
    from app.secrets.crypto import aad_for_custom_db, envelope_encrypt
    from app.secrets.keys import get_kek

    kek = get_kek()
    for row in CustomDatabase.objects.all():
        plaintext = row.api_key
        if not plaintext:
            continue
        blob = envelope_encrypt(
            plaintext.encode("utf-8"),
            kek,
            aad_for_custom_db(row.created_by_id, str(row.id)),
        )
        row.api_key_wrapped_dek = blob.wrapped_dek
        row.api_key_ciphertext = blob.ciphertext
        row.api_key_nonce = blob.nonce
        row.api_key_key_version = blob.key_version
        row.api_key = None
        row.save(
            update_fields=[
                "api_key_wrapped_dek",
                "api_key_ciphertext",
                "api_key_nonce",
                "api_key_key_version",
                "api_key",
            ]
        )


def noop_reverse(apps, schema_editor):
    # Do not write plaintext back out of the vault.
    return


class Migration(migrations.Migration):

    dependencies = [
        ("metadata", "0001_initial"),
        ("nw_secrets", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="customdatabase",
            name="api_key_ciphertext",
            field=models.BinaryField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="customdatabase",
            name="api_key_key_version",
            field=models.PositiveSmallIntegerField(default=1),
        ),
        migrations.AddField(
            model_name="customdatabase",
            name="api_key_nonce",
            field=models.BinaryField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="customdatabase",
            name="api_key_wrapped_dek",
            field=models.BinaryField(blank=True, null=True),
        ),
        migrations.RunPython(encrypt_existing_api_keys, noop_reverse),
        migrations.RemoveField(
            model_name="customdatabase",
            name="api_key",
        ),
    ]
