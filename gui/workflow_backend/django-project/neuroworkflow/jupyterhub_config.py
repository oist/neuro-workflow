import os
from dockerspawner import DockerSpawner
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_handlers import CORSHandler, AuthStatusHandler

# JupyterHub configuration
c = get_config()

# Network configuration
c.JupyterHub.hub_ip = "0.0.0.0"
c.JupyterHub.port = 8000

# Use Docker spawner
c.JupyterHub.spawner_class = DockerSpawner

# Docker spawner configuration - NEST simulator enabled image
c.DockerSpawner.image = "nest-jupyterlab:latest"  # Built from Dockerfile.nest
c.DockerSpawner.network_name = "jupyterhub-network"

# Remove containers when they stop
c.DockerSpawner.remove = True

# Volume mounts - .envファイルからホストパスを取得
host_project_path = os.environ.get("HOST_PROJECT_PATH")

if not host_project_path:
    raise ValueError("HOST_PROJECT_PATH environment variable is required")

c.DockerSpawner.volumes = {
    f"{host_project_path}/neuroworkflow/neuro": {
        "bind": "/home/jovyan/neuro",
        "mode": "ro",
    },
    f"{host_project_path}/codes/nodes": {
        "bind": "/home/jovyan/codes/nodes",
        "mode": "rw",
    },
    f"{host_project_path}/codes/projects": {
        "bind": "/home/jovyan/codes/projects", 
        "mode": "rw"
    },
    # "jupyterhub-user-{username}": {"bind": "/home/jovyan/work", "mode": "rw"},
}

# Environment variables for spawned containers
c.DockerSpawner.environment = {
    "GRANT_SUDO": "yes", 
    "CHOWN_HOME": "yes",
    "JUPYTER_CONFIG_DIR": "/home/jovyan/.jupyter",
}

# Notebook configuration
c.DockerSpawner.notebook_dir = "/home/jovyan"
c.DockerSpawner.default_url = "/lab"

# JupyterLab CSP settings for iframe embedding
c.DockerSpawner.args = [
    "--ServerApp.tornado_settings={'headers':{'Content-Security-Policy':\"frame-ancestors 'self' http://localhost:5173 http://127.0.0.1:5173 *\"}}",
    "--ServerApp.allow_origin='*'",
    "--ServerApp.disable_check_xsrf=True"
]

# User management
c.Authenticator.allowed_users = {"user1", "user2"}

# Use dummy authenticator for development (change for production!)
c.JupyterHub.authenticator_class = "jupyterhub.auth.DummyAuthenticator"
c.DummyAuthenticator.password = "password"

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
    'headers': {
        'X-Frame-Options': 'SAMEORIGIN',  # または 'ALLOWALL' for any domain
        'Content-Security-Policy': "frame-ancestors 'self' http://localhost:5173 http://127.0.0.1:5173"
    }
}

# CORS settings for cross-origin requests
c.JupyterHub.extra_handlers = [
    (r'/api/auth-status', AuthStatusHandler),
    (r'/api/(.*)', CORSHandler),
]

# Cookie settings for iframe embedding
c.JupyterHub.cookie_options = {
    'SameSite': 'None',
    'Secure': False,  # Set to True in production with HTTPS
}

# ----定期的なクリーンアップ
c.JupyterHub.shutdown_on_logout = True
c.JupyterHub.cleanup_servers = True
