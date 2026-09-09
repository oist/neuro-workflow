import os

from django.apps import AppConfig


class SecretsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app.secrets"
    verbose_name = "Owner secret store"
    label = "nw_secrets"

    def ready(self) -> None:
        if os.environ.get("NW_TESTING") or os.environ.get("PYTEST_CURRENT_TEST"):
            return
        from .keys import require_production_master_key

        require_production_master_key()
