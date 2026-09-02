import sys

from django.apps import AppConfig


class SecretsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app.secrets"
    verbose_name = "Owner secret store"
    label = "nw_secrets"

    def ready(self) -> None:
        if "pytest" in sys.modules:
            return
        from .keys import require_production_master_key

        require_production_master_key()
