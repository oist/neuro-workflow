"""Jupyter ContentsManager filter: hide project dirs the opener may not see.

This is a *visual* filter for the Lab file browser. The kernel and terminal
still see every path mounted in this container. Isolation between the
internal and hackathon Labs is done with separate bind-mounts, not here.

The opener's identity comes from a short-lived NeuroWorkflow viewer token
(query ``nw_viewer`` or cookie ``nw_viewer``). The Lab page URL carries
``?nw_viewer=``; ``prepare()`` copies it onto a cookie so later
``/api/contents`` XHRs (which do not keep the query string) still identify
the opener.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import urllib.error
import urllib.request
from collections import OrderedDict
from contextvars import ContextVar
from typing import Iterable
from uuid import UUID

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_viewer_token: ContextVar[str | None] = ContextVar("nw_viewer_token", default=None)
_allowlist_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_CACHE_TTL_SECONDS = 30.0
_CACHE_MAX = 64

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


def _open_allowlist() -> dict:
    """Do not hide project dirs (used when the opener is unknown)."""
    return {
        "project_ids": [],
        "legacy_names": [],
        "hide_unlisted_projects": False,
    }


def fetch_allowlist(token: str | None) -> dict:
    import time

    if not token:
        return _open_allowlist()
    now = time.time()
    cached = _allowlist_cache.get(token)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        _allowlist_cache.move_to_end(token)
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
        # Unknown opener / backend hiccup: show the Lab tree rather than 404
        # every project. Kernel/terminal already see the same paths.
        payload = _open_allowlist()
    if "hide_unlisted_projects" not in payload:
        payload["hide_unlisted_projects"] = True
    while len(_allowlist_cache) >= _CACHE_MAX:
        _allowlist_cache.popitem(last=False)
    _allowlist_cache[token] = (now, payload)
    return payload


def _apply_listing_filter(model, path, content, allow: dict):
    if not allow.get("hide_unlisted_projects", True):
        return model
    if (
        content
        and isinstance(model, dict)
        and model.get("type") == "directory"
        and model.get("content")
    ):
        model["content"] = filter_directory_entries(
            path,
            model["content"],
            project_ids=allow.get("project_ids") or [],
            legacy_names=allow.get("legacy_names") or [],
        )
    return model


def _capture_viewer_token(handler, serverapp) -> None:
    """Copy ``nw_viewer`` from the Lab URL onto a cookie for later API calls."""
    token = None
    try:
        token = handler.get_query_argument("nw_viewer", default=None)
        if token:
            base = getattr(serverapp, "base_url", None) or "/"
            handler.set_cookie("nw_viewer", token, path=base, httponly=True)
        else:
            token = handler.get_cookie("nw_viewer")
    except Exception:
        token = None
    try:
        _viewer_token.set(token)
    except Exception:
        pass


def _load_jupyter_server_extension(serverapp):
    """Register contents listing wrapper + session cookie endpoint."""
    try:
        from jupyter_server.base.handlers import APIHandler
        from jupyter_server.utils import url_path_join
        import tornado.web
    except ImportError:
        serverapp.log.warning("jupyter_tenant_filter: jupyter_server not available")
        return

    orig_prepare = tornado.web.RequestHandler.prepare

    def prepare(self, *args, **kwargs):
        _capture_viewer_token(self, serverapp)
        return orig_prepare(self, *args, **kwargs)

    tornado.web.RequestHandler.prepare = prepare

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
            base = getattr(serverapp, "base_url", None) or "/"
            self.set_cookie("nw_viewer", token, path=base, httponly=True)
            _viewer_token.set(token)
            self.finish({"ok": True})

        def check_xsrf_cookie(self):
            return

    original_get = serverapp.contents_manager.get

    def filtered_get(path, content=True, type=None, format=None, **kwargs):
        # Listing filter only — never 404 a path here. JupyterLab also
        # POSTs /api/contents/<file>/checkpoints; a catch-all ContentsHandler
        # stole that route and broke notebook saves.
        result = original_get(path, content=content, type=type, format=format, **kwargs)
        token = _viewer_token.get()
        if inspect.isawaitable(result):

            async def _wrapped():
                import asyncio

                model = await result
                loop = asyncio.get_running_loop()
                allow = await loop.run_in_executor(None, fetch_allowlist, token)
                return _apply_listing_filter(model, path, content, allow)

            return _wrapped()
        allow = fetch_allowlist(token)
        return _apply_listing_filter(result, path, content, allow)

    serverapp.contents_manager.get = filtered_get

    base = serverapp.base_url
    handlers = [
        (url_path_join(base, "api/nw-session"), NWSessionHandler),
    ]
    try:
        serverapp.web_app.add_handlers(".*$", handlers)
    except Exception as exc:
        serverapp.log.warning("jupyter_tenant_filter: failed to add handlers: %s", exc)


def _jupyter_server_extension_points():
    return [{"module": "jupyter_tenant_filter"}]
