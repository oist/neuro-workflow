import os
from pathlib import Path
from .config import (
    DB_HOST,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    DB_PORT,
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    SUPABASE_JWT_SECRET,
    KEYCLOAK_URL,
    KEYCLOAK_REALM,
    KEYCLOAK_CLIENT_ID,
    SECRET_KEY,
)

# ==============================================================================
# CORE SETTINGS
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = SECRET_KEY

DEBUG = True

_extra_hosts = [h.strip() for h in os.getenv("ALLOWED_HOSTS_EXTRA", "").split(",") if h.strip()]
if os.getenv("ALLOWED_HOSTS_ALL", "").lower() in ("0", "false", "no"):
    ALLOWED_HOSTS = ["localhost", "127.0.0.1"] + _extra_hosts
else:
    ALLOWED_HOSTS = ["*"]

ROOT_URLCONF = "config.urls"

WSGI_APPLICATION = "config.wsgi.application"

# ==============================================================================
# APPLICATION CONFIGURATION
# ==============================================================================

DJANGO_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]

THIRD_PARTY_APPS = [
    "rest_framework",
    "corsheaders",
]

LOCAL_APPS = [
    "app.box.apps.BoxConfig",
    "app.workflow.apps.WorkflowConfig",
    "app.metadata.apps.MetadataConfig",
    "app.chat.apps.ChatConfig",
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# ==============================================================================
# MIDDLEWARE CONFIGURATION
# ==============================================================================

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    # "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": DB_NAME,
        "USER": DB_USER,
        "PASSWORD": DB_PASSWORD,
        "HOST": DB_HOST,
        "PORT": DB_PORT,
    }
}

# ==============================================================================
# AUTHENTICATION & AUTHORIZATION
# ==============================================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# ==============================================================================
# DJANGO REST FRAMEWORK
# ==============================================================================

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "app.auth.authentication.CombinedJWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
    ],
}

# ==============================================================================
# AUTHENTICATION PROVIDER CONFIGURATION
# ==============================================================================

SUPABASE_URL = SUPABASE_URL
SUPABASE_ANON_KEY = SUPABASE_ANON_KEY
SUPABASE_JWT_SECRET = SUPABASE_JWT_SECRET

KEYCLOAK_URL = KEYCLOAK_URL
KEYCLOAK_REALM = KEYCLOAK_REALM
KEYCLOAK_CLIENT_ID = KEYCLOAK_CLIENT_ID

# ==============================================================================
# CORS CONFIGURATION
# ==============================================================================

_cors_extra = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS_EXTRA", "").split(",") if o.strip()]
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
] + _cors_extra
if os.getenv("CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes"):
    CORS_ALLOW_ALL_ORIGINS = True

CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
    "x-internal-secret",
]

CORS_ALLOW_METHODS = [
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
]

CORS_EXPOSE_HEADERS = [
    "X-Conversation-Id",
]

CORS_ALLOW_CREDENTIALS = True

_csrf_extra = [o.strip() for o in os.getenv("CSRF_TRUSTED_ORIGINS_EXTRA", "").split(",") if o.strip()]
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
] + _csrf_extra

# ==============================================================================
# SECURITY SETTINGS
# ==============================================================================

# security header
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"

# HTTPS settings (for production environments)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
# Disable SSL redirects in development environments
# SECURE_SSL_REDIRECT = True

# ==============================================================================
# INTERNATIONALIZATION
# ==============================================================================

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Tokyo"
USE_I18N = True
USE_TZ = True

# ==============================================================================
# STATIC & MEDIA FILES
# ==============================================================================

STATIC_URL = "static/"
# STATIC_ROOT = BASE_DIR / "staticfiles"  # For production environment

MEDIA_URL = "nodes/"

# 変更必須
MEDIA_ROOT = os.path.join(BASE_DIR, "codes/nodes")

PROJECTS_ROOT = os.path.join(BASE_DIR, "codes/projects")

# ==============================================================================
# FILE UPLOAD SETTINGS
# ==============================================================================

FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024  # 5MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024  # 5MB

# ==============================================================================
# TEMPLATES
# ==============================================================================

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ==============================================================================
# DJANGO SPECIFIC
# ==============================================================================

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ==============================================================================
# LOGGING (開発用)
# ==============================================================================

#LOGGING = {
#    "version": 1,
#    "disable_existing_loggers": False,
#    "handlers": {
#        "console": {
#            "class": "logging.StreamHandler",
#        },
#    },
#    "loggers": {
#        "app.auth.authentication": {
#            "handlers": ["console"],
#            "level": "DEBUG",
#            "propagate": True,
#        },
#    },
#}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "app.auth.authentication": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

