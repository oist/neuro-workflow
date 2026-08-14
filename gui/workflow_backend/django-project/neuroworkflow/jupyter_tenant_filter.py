"""Jupyter ContentsManager filter: hide project dirs the opener may not see.

This is a *visual* filter for the Lab file browser. The kernel and terminal
still see every path mounted in this container. Isolation between the
internal and hackathon Labs is done with separate bind-mounts, not here.

The opener's identity comes from a short-lived NeuroWorkflow viewer token
(query ``nw_viewer`` or cookie ``nw_viewer``).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from contextvars import ContextVar
from typing import Iterable
from uuid import UUID

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_viewer_token: ContextVar[str | None] = ContextVar("nw_viewer_token", default=None)
_allowlist_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 30.0

PROJECTS_PREFIXES = (
    "codes/projects",
    "/codes/projects",
    "codes/projects/",
    "/codes/projects/",
)


def _normalize_path(path: str) -> str:
    path = (path or "").replace("\\", "/").strip("/")
    while path.startswith("./"):
        path = path[2:]
    return path


def is_projects_root(path: str) -> bool:
    return _normalize_path(path) in ("codes/projects", "codes/projects/")


def project_id_from_path(path: str) -> str | None:
    normalized = _normalize_path(path)
    parts = normalized.split("/")
    if len(parts) >= 3 and parts[0] == "codes" and parts[1] == "projects":
        return parts[2]
    return None


def is_uuid_name(name: str) -> bool:
    if not _UUID_RE.match(name or ""):
        return False
    try:
        UUID(name)
        return True
    except ValueError:
        return False


def filter_directory_entries(
    path: str,
    entries: Iterable[dict],
    *,
    project_ids: Iterable[str] | None,
    legacy_names: Iterable[str] | None = None,
    fail_closed: bool = True,
) -> list[dict]:
    """Drop project directories that are not on the allow-list.

    Non-project paths are left unchanged. Files (README, etc.) in the
    projects root stay visible. Directories that look like project ids or
    legacy capitalized names are filtered.
    """
    allowed_ids = {str(x) for x in (project_ids or [])}
    allowed_legacy = {str(x) for x in (legacy_names or [])}
    parent = _normalize_path(path)
    looking_at_projects = parent in ("codes/projects",)

    out = []
    for entry in entries:
        name = str(entry.get("name") or entry.get("path") or "")
        kind = entry.get("type") or entry.get("content_type")
        if looking_at_projects and kind in (None, "directory", "dir"):
            if not name or name in (".", ".."):
                continue
            if is_uuid_name(name):
                if name in allowed_ids:
                    out.append(entry)
                elif not fail_closed and not allowed_ids:
                    out.append(entry)
                continue
            if name in allowed_legacy or name in allowed_ids:
                out.append(entry)
                continue
            # Unknown directory name under projects/: hide unless it is clearly
            # not a project folder (we treat every directory as a project).
            continue
        nested_id = project_id_from_path(f"{parent}/{name}") if parent else None
        if nested_id and is_uuid_name(nested_id) and nested_id not in allowed_ids:
            if fail_closed or allowed_ids:
                continue
        out.append(entry)
    return out


def path_is_allowed(
    path: str,
    *,
    project_ids: Iterable[str] | None,
    legacy_names: Iterable[str] | None = None,
    fail_closed: bool = True,
) -> bool:
    project_id = project_id_from_path(path)
    if not project_id:
        return True
    allowed_ids = {str(x) for x in (project_ids or [])}
    allowed_legacy = {str(x) for x in (legacy_names or [])}
    if project_id in allowed_ids or project_id in allowed_legacy:
        return True
    if not fail_closed and not allowed_ids:
        return True
    return False


def _backend_url() -> str:
    return os.environ.get("NEUROWORKFLOW_BACKEND_URL", "http://backend:3000").rstrip("/")


def fetch_allowlist(token: str | None) -> dict:
    import time

    if not token:
        return {
            "project_ids": [],
            "legacy_names": [],
            "hide_unlisted_projects": True,
        }
    now = time.time()
    cached = _allowlist_cache.get(token)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    url = f"{_backend_url()}/api/workflow/jupyter/visible-paths/"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Viewer {token}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        payload = {
            "project_ids": [],
            "legacy_names": [],
            "hide_unlisted_projects": True,
        }
    _allowlist_cache[token] = (now, payload)
    return payload


def _load_jupyter_server_extension(serverapp):
    """Register contents handler wrapper + session cookie endpoint."""
    try:
        from jupyter_server.base.handlers import APIHandler
        from jupyter_server.services.contents.handlers import ContentsHandler
        from jupyter_server.utils import url_path_join
        import tornado.web
    except ImportError:
        serverapp.log.warning("jupyter_tenant_filter: jupyter_server not available")
        return

    class NWSessionHandler(APIHandler):
        auth_resource = "contents"

        @tornado.web.authenticated
        def post(self):
            body = self.get_json_body() or {}
            token = (
                body.get("token")
                or self.get_argument("nw_viewer", default=None)
                or self.get_cookie("nw_viewer")
            )
            if not token:
                raise tornado.web.HTTPError(400, "Missing viewer token")
            self.set_secure_cookie(
                "nw_viewer",
                token,
                httponly=True,
                path="/",
            )
            self.finish({"ok": True})

        def check_xsrf_cookie(self):
            return

    class NWContentsHandler(ContentsHandler):
        async def prepare(self):
            await super().prepare()
            token = (
                self.get_argument("nw_viewer", default=None)
                or self.get_cookie("nw_viewer")
            )
            _viewer_token.set(token)

    original_get = serverapp.contents_manager.get

    def filtered_get(path, content=True, type=None, format=None, **kwargs):
        token = _viewer_token.get()
        allow = fetch_allowlist(token)
        project_ids = allow.get("project_ids") or []
        legacy_names = allow.get("legacy_names") or []
        if not path_is_allowed(path, project_ids=project_ids, legacy_names=legacy_names):
            from tornado.web import HTTPError

            raise HTTPError(404, "Not found")
        model = original_get(path, content=content, type=type, format=format, **kwargs)
        if (
            content
            and isinstance(model, dict)
            and model.get("type") == "directory"
            and model.get("content")
        ):
            model["content"] = filter_directory_entries(
                path,
                model["content"],
                project_ids=project_ids,
                legacy_names=legacy_names,
            )
        return model

    serverapp.contents_manager.get = filtered_get

    base = serverapp.base_url
    handlers = [
        (url_path_join(base, "api/nw-session"), NWSessionHandler),
        (url_path_join(base, r"api/contents(.*)"), NWContentsHandler),
    ]
    try:
        serverapp.web_app.add_handlers(".*$", handlers)
        # Prefer our contents handler over the default by inserting first.
        rules = serverapp.web_app.wildcard_router.rules
        ours = [r for r in rules if getattr(r, "target", None) is NWContentsHandler]
        rest = [r for r in rules if getattr(r, "target", None) is not NWContentsHandler]
        if ours:
            serverapp.web_app.wildcard_router.rules = ours + rest
    except Exception as exc:
        serverapp.log.warning("jupyter_tenant_filter: failed to add handlers: %s", exc)


def _jupyter_server_extension_points():
    return [{"module": "jupyter_tenant_filter"}]
