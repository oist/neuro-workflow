import asyncio
import json
import logging
import os
import uuid

import httpx
from websockets.asyncio.client import connect

logger = logging.getLogger(__name__)

# JupyterHub / Jupyter Server connection settings.
#
# ``JUPYTERHUB_BASE_URL`` is the single source of truth for the JupyterHub
# subpath: the same value is consumed by the JupyterHub container itself
# (``jupyterhub_config.py``) and by this backend. Combining it with
# ``JUPYTERHUB_INTERNAL_HOST`` yields the fully-qualified API base URL.
JUPYTERHUB_INTERNAL_HOST = os.environ.get(
    "JUPYTERHUB_INTERNAL_HOST", "http://jupyterhub:8000"
).rstrip("/")
_base_url = os.environ.get("JUPYTERHUB_BASE_URL", "/").strip("/")
JUPYTERHUB_API_URL = (
    f"{JUPYTERHUB_INTERNAL_HOST}/{_base_url}"
    if _base_url
    else JUPYTERHUB_INTERNAL_HOST
)
JUPYTERHUB_API_TOKEN = os.environ.get("JUPYTERHUB_API_TOKEN") or None
JUPYTER_USER = os.environ.get("JUPYTER_EXECUTION_USER", "user1")

# Timeouts
SERVER_START_TIMEOUT = 120  # seconds to wait for server to start
KERNEL_START_TIMEOUT = 30
EXECUTE_IDLE_TIMEOUT = 600  # max seconds to wait for execution to finish


def _image_event(mime_bundle: dict) -> dict:
    """Build an image event from a Jupyter display mime bundle.

    Whitespace is stripped from the base64 payload so the frontend can
    embed it directly in a ``data:`` URI.
    """
    return {
        "type": "image",
        "data": {
            "content": "".join(mime_bundle["image/png"].split()),
            "mime": "image/png",
        },
    }


class JupyterExecutionService:
    """Execute code on a Jupyter kernel running inside the JupyterLab container.

    Flow:
      1. Ensure the user's single-user server is running (via JupyterHub API)
      2. Create a new kernel on that server (via Jupyter Server API)
      3. Connect to the kernel via WebSocket and send an execute_request
      4. Yield SSE-formatted events as output arrives
      5. Tear down the kernel when done
    """

    def __init__(self, user: str = JUPYTER_USER):
        if not JUPYTERHUB_API_TOKEN:
            raise ValueError(
                "JUPYTERHUB_API_TOKEN environment variable is not set. "
                "Configure it in docker-compose.yml or .env."
            )
        self.user = user
        self.hub_url = JUPYTERHUB_API_URL.rstrip("/")
        self.token = JUPYTERHUB_API_TOKEN
        self._headers = {
            "Authorization": f"token {self.token}",
        }

    # ------------------------------------------------------------------
    # JupyterHub: ensure the single-user server is running
    # ------------------------------------------------------------------

    async def _ensure_server_running(self) -> None:
        """Start the user's server if it is not already running."""
        async with httpx.AsyncClient(timeout=30) as client:
            # Check current status
            resp = await client.get(
                f"{self.hub_url}/hub/api/users/{self.user}",
                headers=self._headers,
            )
            resp.raise_for_status()
            user_model = resp.json()

            # The default server is keyed by empty string
            server = user_model.get("servers", {}).get("", {})
            if server.get("ready"):
                logger.info("Server for %s is already running", self.user)
                return

            # Server not running – start it
            logger.info("Starting server for %s …", self.user)
            resp = await client.post(
                f"{self.hub_url}/hub/api/users/{self.user}/server",
                headers=self._headers,
            )
            # 201 = started, 202 = already starting/pending
            if resp.status_code == 400:
                raise RuntimeError(
                    f"Failed to start server for {self.user}: "
                    f"HTTP {resp.status_code} - {resp.text}"
                )
            if resp.status_code not in (201, 202):
                resp.raise_for_status()

            # Poll until ready
            for _ in range(SERVER_START_TIMEOUT // 2):
                await asyncio.sleep(2)
                resp = await client.get(
                    f"{self.hub_url}/hub/api/users/{self.user}",
                    headers=self._headers,
                )
                resp.raise_for_status()
                server = resp.json().get("servers", {}).get("", {})
                if server.get("ready"):
                    logger.info("Server for %s is now ready", self.user)
                    return

            raise TimeoutError(
                f"Server for {self.user} did not start within {SERVER_START_TIMEOUT}s"
            )

    # ------------------------------------------------------------------
    # Jupyter Server: kernel lifecycle
    # ------------------------------------------------------------------

    def _server_api_base(self) -> str:
        """Return the base URL for the single-user server API via the Hub proxy."""
        return f"{self.hub_url}/user/{self.user}"

    async def _create_kernel(self) -> str:
        """Create a new kernel and return its ID."""
        url = f"{self._server_api_base()}/api/kernels"
        async with httpx.AsyncClient(timeout=KERNEL_START_TIMEOUT) as client:
            resp = await client.post(
                url,
                headers=self._headers,
                json={"name": "python3"},
            )
            resp.raise_for_status()
            kernel_id = resp.json()["id"]
            logger.info("Created kernel %s for user %s", kernel_id, self.user)
            return kernel_id

    async def _delete_kernel(self, kernel_id: str) -> None:
        """Delete (shut down) a kernel."""
        url = f"{self._server_api_base()}/api/kernels/{kernel_id}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.delete(url, headers=self._headers)
                if resp.status_code < 400:
                    logger.info("Deleted kernel %s", kernel_id)
                else:
                    logger.warning(
                        "Failed to delete kernel %s: %s %s",
                        kernel_id,
                        resp.status_code,
                        resp.text,
                    )
        except Exception:
            logger.exception("Error deleting kernel %s", kernel_id)

    # ------------------------------------------------------------------
    # WebSocket: execute code and stream output
    # ------------------------------------------------------------------

    async def execute_code(self, code: str, *, runtime_secrets: dict | None = None):
        """Async generator that yields SSE event dicts.

        Each yielded dict has the shape ``{"type": str, "data": dict}``.
        """
        from app.secrets.inject import redact_event, wrap_jupyter_code
        from app.secrets.logging_filter import clear_secret_values, register_secret_values

        values = tuple((runtime_secrets or {}).values())
        if runtime_secrets:
            register_secret_values(*values)
            code = wrap_jupyter_code(code, runtime_secrets)

        await self._ensure_server_running()
        kernel_id = await self._create_kernel()

        try:
            async for event in self._stream_execution(kernel_id, code):
                yield redact_event(event, values)
        finally:
            await self._delete_kernel(kernel_id)
            clear_secret_values()

    async def _stream_execution(self, kernel_id: str, code: str):
        """Connect to the kernel WebSocket and execute *code*."""
        # Build WebSocket URL – derive scheme from hub_url
        ws_scheme = "wss" if self.hub_url.startswith("https://") else "ws"
        hub_host = self.hub_url.split("://", 1)[1] if "://" in self.hub_url else self.hub_url
        ws_url = (
            f"{ws_scheme}://{hub_host}/user/{self.user}"
            f"/api/kernels/{kernel_id}/channels"
        )

        session_id = uuid.uuid4().hex
        msg_id = uuid.uuid4().hex

        execute_request = {
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": self.user,
                "session": session_id,
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "buffers": [],
            "channel": "shell",
        }

        logger.info("Connecting to kernel WS: %s", ws_url)

        async with connect(
            ws_url,
            additional_headers=self._headers,
            max_size=2**27,  # 128 MB; inline figures arrive as one base64 frame
            open_timeout=30,
        ) as ws:
            await ws.send(json.dumps(execute_request))
            logger.info("Sent execute_request (msg_id=%s)", msg_id)

            # Execution is complete only when both the shell reply and the
            # iopub "idle" status have arrived. The channels are independent:
            # execute_reply can beat large iopub messages (e.g. inline
            # figures) still in flight, so stopping on the reply alone would
            # drop them.
            reply_status = None
            iopub_idle = False
            while reply_status is None or not iopub_idle:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=EXECUTE_IDLE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    yield {
                        "type": "error",
                        "data": {
                            "ename": "TimeoutError",
                            "evalue": f"No output for {EXECUTE_IDLE_TIMEOUT}s",
                            "traceback": [],
                        },
                    }
                    yield {
                        "type": "done",
                        "data": {"status": "error"},
                    }
                    return

                msg = json.loads(raw)
                msg_type = msg.get("msg_type") or msg.get("header", {}).get(
                    "msg_type", ""
                )
                parent_msg_id = msg.get("parent_header", {}).get("msg_id", "")

                # Only handle messages that are replies to our request
                if parent_msg_id != msg_id:
                    continue

                content = msg.get("content", {})

                if msg_type == "stream":
                    stream_name = content.get("name", "stdout")
                    yield {
                        "type": stream_name,  # "stdout" or "stderr"
                        "data": {"content": content.get("text", "")},
                    }

                elif msg_type == "execute_result":
                    data = content.get("data", {})
                    if "image/png" in data:
                        yield _image_event(data)
                    else:
                        yield {
                            "type": "execute_result",
                            "data": {"content": data.get("text/plain", "")},
                        }

                elif msg_type == "display_data":
                    data = content.get("data", {})
                    if "image/png" in data:
                        yield _image_event(data)
                    elif "text/plain" in data:
                        yield {
                            "type": "execute_result",
                            "data": {"content": data.get("text/plain", "")},
                        }

                elif msg_type == "error":
                    yield {
                        "type": "error",
                        "data": {
                            "ename": content.get("ename", "Error"),
                            "evalue": content.get("evalue", ""),
                            "traceback": content.get("traceback", []),
                        },
                    }

                elif msg_type == "execute_reply":
                    reply_status = content.get("status", "ok")

                elif msg_type == "status":
                    if content.get("execution_state") == "idle":
                        iopub_idle = True

            yield {
                "type": "done",
                "data": {"status": reply_status},
            }
