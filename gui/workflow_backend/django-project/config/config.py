from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "db")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")

def _clean(value: str | None) -> str | None:
    """Strip whitespace, BOM, and surrounding quotes from a .env value.

    Some editors / paste flows leave a trailing newline, BOM, or wrap the
    value in quotes that the dotenv parser does not strip. Any of those make
    the JWT secret no longer match the bytes Supabase signs with.
    """
    if value is None:
        return None
    cleaned = value.strip().lstrip("﻿")
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1]
    return cleaned


SUPABASE_URL = (_clean(os.getenv("SUPABASE_URL")) or "").rstrip("/") or None
SUPABASE_ANON_KEY = _clean(os.getenv("SUPABASE_ANON_KEY"))
SUPABASE_JWT_SECRET = _clean(os.getenv("SUPABASE_JWT_SECRET"))

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080/auth")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "neuroworkflow")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "neuroworkflow-app")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
