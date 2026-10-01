"""SQLite Django settings are kept for import-only experiments.

This repository's migrations include Postgres-specific SQL, so isolated pytest
in CI/dev should use an ephemeral Postgres (see the cluster-sbatch progress
log) rather than this module.
"""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-cluster-sbatch-logs")
os.environ.setdefault("DB_PASSWORD", "unused")
os.environ.setdefault("DB_USER", "unused")
os.environ.setdefault("DB_NAME", "unused")
os.environ.setdefault("DB_PORT", "5432")

from .settings import *  # noqa: E402,F403

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
