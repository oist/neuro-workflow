import os
import sys

from dockerspawner import DockerSpawner

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_handlers import CORSHandler, AuthStatusHandler

# JupyterHub configuration
c = get_config()

# Network configuration
c.JupyterHub.hub_ip = "0.0.0.0"
c.JupyterHub.port = 8000
c.JupyterHub.base_url = os.environ.get("JUPYTERHUB_BASE_URL", "/")

# Use Docker spawner
c.JupyterHub.spawner_class = DockerSpawner

# Docker spawner configuration - NEST simulator enabled image
c.DockerSpawner.image = "nest-jupyterlab:latest"  # Built from Dockerfile.nest
c.DockerSpawner.network_name = "jupyterhub-network"  # Use the Docker Compose network (must match docker-compose.yml)

# Remove containers when they stop
c.DockerSpawner.remove = True

# Volume mounts - Get host path from .env file
host_project_path = os.environ.get("HOST_PROJECT_PATH")

if not host_project_path:
    raise ValueError("HOST_PROJECT_PATH environment variable is required")

# Repository .claude directory (skills tracked in git) — mounted read-only so
# the notebook chat agent can read .claude/skills/*.md. Defaults to the repo
# root relative to host_project_path (.../gui/workflow_backend/django-project).
host_claude_path = os.environ.get("HOST_CLAUDE_PATH") or os.path.normpath(
    os.path.join(host_project_path, "..", "..", "..", ".claude")
)

c.DockerSpawner.volumes = {
    f"{host_project_path}/codes/nodes": {
        "bind": "/home/jovyan/codes/nodes",
        "mode": "rw",
    },
    f"{host_project_path}/codes/projects": {
        "bind": "/home/jovyan/codes/projects",
        "mode": "rw",
    },
    f"{host_project_path}/codes/neuroworkflow": {
        "bind": "/home/jovyan/codes/neuroworkflow",
        "mode": "rw",
    },
    host_claude_path: {
        "bind": "/home/jovyan/.claude",
        "mode": "ro",
    },
    # "jupyterhub-user-{username}": {"bind": "/home/jovyan/work", "mode": "rw"},
}

# Environment variables for spawned containers
c.DockerSpawner.environment = {
    "GRANT_SUDO": os.environ.get("JUPYTER_GRANT_SUDO", "yes"),
    "CHOWN_HOME": "yes",
    "JUPYTER_CONFIG_DIR": "/home/jovyan/.jupyter",
    # Make `import neuroworkflow` (and neuroworkflow.agent) resolve from the
    # mounted codes/ tree without per-notebook sys.path hacks.
    "PYTHONPATH": "/home/jovyan/codes",
    # Wiring for the in-notebook chat agent (Issue #52).
    # NOTE: do NOT set JUPYTERHUB_API_TOKEN here — JupyterHub injects a
    # per-server token under that name for the single-user server's own OAuth
    # with the hub; overriding it causes "Client secret mismatch" / 401 on the
    # token exchange and breaks spawning. We pass the shared backend service
    # token under a distinct name instead.
    "NEUROWORKFLOW_BACKEND_URL": os.environ.get(
        "NEUROWORKFLOW_BACKEND_URL", "http://backend:3000"
    ),
    "NEUROWORKFLOW_SERVICE_TOKEN": os.environ.get("JUPYTERHUB_API_TOKEN", ""),
    # The in-kernel Claude agent reaches Anthropic through the backend proxy, so
    # the real key never enters the kernel. The agent sends the service token as
    # its ANTHROPIC_API_KEY; the backend swaps in the real key.
    "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL")
    or (
        os.environ.get("NEUROWORKFLOW_BACKEND_URL", "http://backend:3000").rstrip("/")
        + "/api/chat/anthropic"
    ),
    "ANTHROPIC_MODEL": os.environ.get("ANTHROPIC_MODEL", ""),
    # bm_mindsdb (mdb) catalog service. Database nodes run inside the kernel and
    # talk to it directly over the jupyterhub-network, so the URL must resolve
    # from the single-user container, not from the backend.
    "MDB_BASE_URL": os.environ.get("MDB_BASE_URL", "http://mdb:8004"),
}

# Notebook configuration
c.DockerSpawner.notebook_dir = "/home/jovyan"
c.DockerSpawner.default_url = "/lab"

# JupyterLab CSP settings for iframe embedding. JUPYTERHUB_FRAME_ORIGIN may list
# several space/comma-separated origins; CSP frame-ancestors accepts a list, so
# the Jupyter iframe can be embedded from more than one public hostname.
_frame_origins = [
    o.strip()
    for o in os.environ.get("JUPYTERHUB_FRAME_ORIGIN", "http://localhost:5173")
    .replace(",", " ")
    .split()
    if o.strip()
]
_frame_ancestors = " ".join(_frame_origins) or "http://localhost:5173"
# Access-Control-Allow-Origin accepts a single value. Jupyter is reached through
# the same-origin nginx /jupyter proxy, so the primary origin is sufficient.
_frame_origin = _frame_origins[0] if _frame_origins else "http://localhost:5173"
c.DockerSpawner.args = [
    f"--ServerApp.tornado_settings={{'headers':{{'Content-Security-Policy':\"frame-ancestors {_frame_ancestors}\"}}}}",
    f"--ServerApp.allow_origin={_frame_origin}",
]
if os.environ.get("JUPYTERHUB_DISABLE_XSRF", "false").lower() == "true":
    c.DockerSpawner.args.append("--ServerApp.disable_check_xsrf=True")

_allowed_users = {
    user.strip()
    for user in os.environ.get("JUPYTERHUB_ALLOWED_USERS", "").split(",")
    if user.strip()
}
if _allowed_users:
    c.Authenticator.allowed_users = _allowed_users

if os.environ.get("JUPYTERHUB_AUTHENTICATOR", "dummy").lower() == "firstuse":
    # First-use authentication stores per-user passwords for production.
    c.JupyterHub.authenticator_class = "firstuseauthenticator.FirstUseAuthenticator"
    c.FirstUseAuthenticator.create_users = False
else:
    # Plain docker compose remains a local/dev stack.
    c.JupyterHub.authenticator_class = "jupyterhub.auth.DummyAuthenticator"
    c.DummyAuthenticator.password = os.environ.get("JUPYTERHUB_DUMMY_PASSWORD", "password")

# Hub configuration
c.JupyterHub.hub_connect_ip = "jupyterhub"

# Data persistence
c.JupyterHub.db_url = "sqlite:///jupyterhub.sqlite"

# Log level
c.JupyterHub.log_level = "INFO"

# Timeout settings
c.DockerSpawner.start_timeout = 300
c.DockerSpawner.http_timeout = 120

# =============== IFRAME EMBEDDING SUPPORT ===============
# Allow embedding in iframes by removing X-Frame-Options restrictions
c.JupyterHub.tornado_settings = {
    "headers": {
        "Content-Security-Policy": f"frame-ancestors {_frame_ancestors}",
        "Access-Control-Allow-Origin": _frame_origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PUT, DELETE",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, X-CSRFToken",
    }
}

# CORS settings for cross-origin requests
c.JupyterHub.extra_handlers = [
    (r"/api/auth-status", AuthStatusHandler),
    (r"/api/(.*)", CORSHandler),
]

# Cookie settings for iframe embedding
_cookie_secure = os.environ.get("JUPYTERHUB_COOKIE_SECURE", "false").lower() == "true"
_cookie_samesite = os.environ.get(
    "JUPYTERHUB_COOKIE_SAMESITE",
    "None" if _cookie_secure else "Lax",
)
c.JupyterHub.cookie_options = {
    "SameSite": _cookie_samesite,
    "Secure": _cookie_secure,
}

# =============== SERVICE TOKEN FOR BACKEND ===============
# Allow the Django backend to use the JupyterHub API and
# access single-user servers (for kernel execution).
_api_token = os.environ.get("JUPYTERHUB_API_TOKEN", "")
if not _api_token:
    import warnings

    warnings.warn(
        "JUPYTERHUB_API_TOKEN is not set. Backend API access will not work. "
        "Set this variable in .env or docker-compose.yml.",
        stacklevel=1,
    )
    _api_token = "unset-token-will-fail"
elif _api_token == "dev-token-change-in-production":
    import warnings

    warnings.warn(
        "JUPYTERHUB_API_TOKEN is using the default development token. "
        "Change it for production deployments.",
        stacklevel=1,
    )
c.JupyterHub.services = [
    {
        "name": "backend",
        "api_token": _api_token,
    }
]
# Grant the service token permission to start/stop servers
# and access user server APIs (kernels, etc.)
c.JupyterHub.load_roles = [
    {
        "name": "backend-role",
        "scopes": [
            "admin:servers",   # start / stop user servers
            "access:servers",  # proxy through to single-user server APIs
            "admin:users",     # read user model (needed for server status)
        ],
        "services": ["backend"],
    }
]

# ----Regular cleanup
c.JupyterHub.shutdown_on_logout = True
c.JupyterHub.cleanup_servers = True
