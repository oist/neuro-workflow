import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone as dt_timezone

import httpx
import jwt
from django.core.cache import cache
from django.db import IntegrityError, transaction
from django.db.models import Count
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from rest_framework import authentication, exceptions, status
from rest_framework.permissions import SAFE_METHODS, IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.auth.authentication import KeycloakAuthentication
from app.tenants import get_user_tenant, hub_username_for_tenant
from app.workflow.jupyter_execution_service import JUPYTERHUB_API_URL
from app.workflow.models import FlowProject
from app.workflow.permissions import get_accessible_project

from .models import ChatProfile, Conversation, Message, NotebookToken
from .serializers import (
    ChatProfileSerializer,
    ConversationSerializer,
    ConversationListSerializer,
    NotebookConversationListSerializer,
    NotebookConversationSerializer,
    NotebookTokenSerializer,
    NotebookTurnSerializer,
    SendMessageSerializer,
)
from .services.chat_orchestrator import orchestrate_chat
from .services.llm_providers import chat_models, provider_for_model
from .services.mcp_client import MCPClient, mcp_tools_to_openai_functions

logger = logging.getLogger(__name__)


def _extract_bearer_token(request) -> str | None:
    """Extract the bearer token from the Authorization header so it can be
    forwarded to the MCP server (and onward to the Django API)."""
    parts = authentication.get_authorization_header(request).split()
    if len(parts) == 2 and parts[0].lower() == b"bearer":
        try:
            return parts[1].decode("utf-8")
        except UnicodeError:
            return None
    return None


_KERNEL_SPACE_TTL = 60  # seconds a hub token lookup is cached


def _hub_user_for_token(hub_token: str) -> str | None:
    """Ask JupyterHub which user a per-server token belongs to.

    Returns None when the hub rejects it or it is not a user token (the hub's
    own service token is refused). Raises ``httpx.HTTPError`` when the hub
    cannot be reached.
    """
    resp = httpx.get(
        f"{JUPYTERHUB_API_URL}/hub/api/user",
        headers={"Authorization": f"token {hub_token}"},
        timeout=10.0,
    )
    if resp.status_code != 200:
        return None
    try:
        identity = resp.json()
    except ValueError:
        return None
    if identity.get("kind") != "user":
        return None
    return identity.get("name") or None


def _kernel_space(request) -> str | None:
    """Return the JupyterHub user whose kernel is calling, or None.

    A kernel presents its own per-server hub token: in ``x-jupyterhub-token``,
    or as its API key (``x-api-key``) when the ``claude`` CLI calls the
    Anthropic proxy. That token only works for its own Jupyter space, so a
    kernel can only use tokens relayed for that space. Answers are cached
    briefly per token to spare the hub a round trip on every call.
    """
    hub_token = request.headers.get("x-jupyterhub-token") or request.headers.get(
        "x-api-key"
    )
    if not hub_token:
        return None
    key = "kernel-space:" + hashlib.sha256(hub_token.encode()).hexdigest()
    cached = cache.get(key)
    if cached is not None:
        return cached or None
    try:
        hub_user = _hub_user_for_token(hub_token)
    except httpx.HTTPError as exc:
        logger.error("JupyterHub user lookup failed: %s", exc)
        return None  # not cached: the next call asks the hub again
    cache.set(key, hub_user or "", _KERNEL_SPACE_TTL)
    return hub_user


def _relayed_token(project_id: str, hub_user: str) -> str:
    """Return the browser token relayed for ``project_id`` in ``hub_user``'s space."""
    stored = NotebookToken.objects.filter(
        project_id=project_id, hub_user=hub_user, expires_at__gt=timezone.now()
    ).first()
    if stored is None:
        raise exceptions.AuthenticationFailed(
            "No relayed user token for this project; open its Jupyter tab in the app."
        )
    return stored.access_token


@method_decorator(csrf_exempt, name="dispatch")
class ConversationListCreateView(APIView):
    """List and create conversations."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        conversations = Conversation.objects.filter(
            user=request.user, is_active=True, source=Conversation.Source.BROWSER,
        )
        serializer = ConversationListSerializer(conversations, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = ConversationSerializer(data=request.data)
        if serializer.is_valid():
            conversation = serializer.save(user=request.user)
            return Response(
                ConversationSerializer(conversation).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@method_decorator(csrf_exempt, name="dispatch")
class ConversationDetailView(APIView):
    """Retrieve or delete a conversation."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, conversation_id):
        try:
            conversation = Conversation.objects.get(
                id=conversation_id, user=request.user, is_active=True,
                source=Conversation.Source.BROWSER,
            )
        except Conversation.DoesNotExist:
            return Response(
                {"error": "Conversation not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        serializer = ConversationSerializer(conversation)
        return Response(serializer.data)

    def delete(self, request, conversation_id):
        try:
            conversation = Conversation.objects.get(
                id=conversation_id, user=request.user,
                source=Conversation.Source.BROWSER,
            )
        except Conversation.DoesNotExist:
            return Response(
                {"error": "Conversation not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        conversation.is_active = False
        conversation.save()
        return Response({"status": "deleted"}, status=status.HTTP_200_OK)


DUPLICATE_PROFILE_NAME = {"name": ["A profile with this name already exists."]}


class _ChatProfilePermissions:
    """Everyone signed in may read profiles; only staff may change them."""

    def get_permissions(self):
        if self.request.method in SAFE_METHODS:
            return [IsAuthenticated()]
        return [IsAdminUser()]


@method_decorator(csrf_exempt, name="dispatch")
class ChatProfileListCreateView(_ChatProfilePermissions, APIView):
    """List the shared chat profiles; create one (staff only)."""

    authentication_classes = [KeycloakAuthentication]

    def get(self, request):
        profiles = ChatProfile.objects.all()
        return Response(ChatProfileSerializer(profiles, many=True).data)

    def post(self, request):
        serializer = ChatProfileSerializer(data=request.data)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    profile = serializer.save(created_by=request.user)
            except IntegrityError:
                return Response(
                    DUPLICATE_PROFILE_NAME, status=status.HTTP_400_BAD_REQUEST
                )
            return Response(
                ChatProfileSerializer(profile).data, status=status.HTTP_201_CREATED
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@method_decorator(csrf_exempt, name="dispatch")
class ChatProfileDetailView(_ChatProfilePermissions, APIView):
    """Retrieve a shared chat profile; update or delete it (staff only)."""

    authentication_classes = [KeycloakAuthentication]

    def _get_profile(self, profile_id):
        return ChatProfile.objects.get(id=profile_id)

    def get(self, request, profile_id):
        try:
            profile = self._get_profile(profile_id)
        except ChatProfile.DoesNotExist:
            return Response(
                {"error": "Chat profile not found"}, status=status.HTTP_404_NOT_FOUND
            )
        return Response(ChatProfileSerializer(profile).data)

    def put(self, request, profile_id):
        try:
            profile = self._get_profile(profile_id)
        except ChatProfile.DoesNotExist:
            return Response(
                {"error": "Chat profile not found"}, status=status.HTTP_404_NOT_FOUND
            )
        serializer = ChatProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            try:
                with transaction.atomic():
                    profile = serializer.save()
            except IntegrityError:
                return Response(
                    DUPLICATE_PROFILE_NAME, status=status.HTTP_400_BAD_REQUEST
                )
            return Response(ChatProfileSerializer(profile).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, profile_id):
        try:
            profile = self._get_profile(profile_id)
        except ChatProfile.DoesNotExist:
            return Response(
                {"error": "Chat profile not found"}, status=status.HTTP_404_NOT_FOUND
            )
        profile.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ChatModelsView(APIView):
    """List the models the browser chat may use; the first is the default.

    The notebook agent reads it too (with its kernel's hub token) to offer
    the models marked ``notebook`` in its own model picker.
    """

    authentication_classes = [KeycloakAuthentication]

    def get_permissions(self):
        return [] if _kernel_space(self.request) else [IsAuthenticated()]

    def get(self, request):
        return Response({"models": chat_models()})


@method_decorator(csrf_exempt, name="dispatch")
class ChatStreamView(APIView):
    """Handle chat messages with SSE streaming response."""

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = SendMessageSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        user = request.user
        auth_token = _extract_bearer_token(request)
        user_message = serializer.validated_data["message"]
        conversation_id = serializer.validated_data.get("conversation_id")
        project_id = serializer.validated_data.get("project_id")
        viewer_context = serializer.validated_data.get("viewer_context")
        profile_id = serializer.validated_data.get("profile_id")

        # Only models on offer may be used; without one the default applies.
        available = [m["id"] for m in chat_models()]
        model = serializer.validated_data.get("model") or None
        if model is None:
            model = available[0] if available else None
        elif model not in available:
            return Response(
                {"error": "Model not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Resolve the chat profile first so a bad id never creates an orphan
        # conversation. Without a profile, non-staff users get the admin
        # default profile (if one is set); staff get all tools.
        profile = None
        if profile_id:
            try:
                profile = ChatProfile.objects.get(id=profile_id)
            except ChatProfile.DoesNotExist:
                return Response(
                    {"error": "Chat profile not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
        elif not user.is_staff:
            profile = ChatProfile.objects.filter(is_default=True).first()

        # Get or create conversation
        if conversation_id:
            try:
                conversation = Conversation.objects.get(
                    id=conversation_id, user=user, is_active=True,
                    source=Conversation.Source.BROWSER,
                )
                # Update project context if the active project changed
                if project_id and str(conversation.project_id) != str(project_id):
                    conversation.project_id = project_id
                    conversation.save(update_fields=["project_id"])
            except Conversation.DoesNotExist:
                return Response(
                    {"error": "Conversation not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )
        else:
            # Create new conversation
            title = user_message[:50] + ("..." if len(user_message) > 50 else "")
            conversation = Conversation.objects.create(
                title=title,
                user=user,
                project_id=project_id,
            )

        response = StreamingHttpResponse(
            self._sync_event_generator(
                conversation, user_message, auth_token, viewer_context, profile, model
            ),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        response["X-Conversation-Id"] = str(conversation.id)
        return response

    def _sync_event_generator(
        self, conversation, user_message, auth_token, viewer_context=None,
        profile=None, model=None,
    ):
        """Wrap the async orchestrator into a sync generator for WSGI."""
        loop = asyncio.new_event_loop()

        try:
            # Send the conversation ID as the first event
            yield _format_sse("conversation_id", {"id": str(conversation.id)})

            agen = orchestrate_chat(
                conversation,
                user_message,
                auth_token=auth_token,
                viewer_context=viewer_context,
                profile=profile,
                model=model,
            )

            while True:
                try:
                    event = loop.run_until_complete(agen.__anext__())
                    event_type = event.get("type", "unknown")
                    event_data = event.get("data", {})
                    yield _format_sse(event_type, event_data)
                except StopAsyncIteration:
                    break
                except Exception as e:
                    logger.error("Stream error: %s", e, exc_info=True)
                    yield _format_sse("error", {"message": str(e)})
                    break
        finally:
            loop.close()


def _project_scoped(tools: list[dict]) -> list[dict]:
    """MCP tools that act on a single project, i.e. take a ``workflow_id``."""
    return [
        t
        for t in tools
        if "workflow_id" in (t.get("inputSchema") or {}).get("properties", {})
    ]


def _format_sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _NotebookMCPView(APIView):
    """Base for the notebook MCP proxies; resolves the JWT forwarded to MCP.

    User path: a real Keycloak JWT in ``Authorization``, forwarded as-is so
    per-user workflow data is scoped correctly. Kernel path: the kernel's own
    JupyterHub token (``x-jupyterhub-token``, which identifies its Jupyter space)
    and a ``project_id``, resolved to the token the browser relayed for that
    project and space (``NotebookTokenView``). The kernel never sees the JWT,
    and only gets the tools that act on that project (``_project_scoped``).
    """

    authentication_classes = [KeycloakAuthentication]

    def get_permissions(self):
        return [] if _kernel_space(self.request) else [IsAuthenticated()]

    def _user_token(self, project_id_source) -> tuple[str, str | None]:
        """Return ``(jwt, relayed_project_id)``; the id is None on the user path."""
        hub_user = _kernel_space(self.request)
        if not hub_user:
            return _extract_bearer_token(self.request), None
        serializer = NotebookTokenSerializer(data=project_id_source)
        serializer.is_valid(raise_exception=True)
        project_id = str(serializer.validated_data["project_id"])
        return _relayed_token(project_id, hub_user), project_id


@method_decorator(csrf_exempt, name="dispatch")
class NotebookMCPToolsView(_NotebookMCPView):
    """List MCP workflow tools (OpenAI function format) for the notebook agent."""

    def get(self, request):
        token, relayed_project_id = self._user_token(request.query_params)

        async def _run():
            mcp = MCPClient(auth_token=token)
            await mcp.initialize()
            tools = await mcp.list_tools()
            if relayed_project_id is not None:
                tools = _project_scoped(tools)
            return mcp_tools_to_openai_functions(tools)

        try:
            tools = asyncio.run(_run())
        except Exception as e:
            logger.error("mcp-tools error: %s", e, exc_info=True)
            return Response(
                {"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY
            )
        return Response({"tools": tools})


@method_decorator(csrf_exempt, name="dispatch")
class NotebookMCPCallView(_NotebookMCPView):
    """Execute a single MCP workflow tool on behalf of the notebook agent.

    On the kernel path only tools that take a ``workflow_id`` are served, and
    the ``workflow_id`` is always the project the token was relayed for.
    """

    def post(self, request):
        tool_name = request.data.get("tool_name")
        arguments = request.data.get("arguments") or {}
        if not tool_name:
            return Response(
                {"error": "'tool_name' is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not isinstance(arguments, dict):
            return Response(
                {"error": "'arguments' must be an object"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        token, relayed_project_id = self._user_token(request.data)
        if relayed_project_id is not None:
            arguments = {**arguments, "workflow_id": relayed_project_id}

        async def _run():
            mcp = MCPClient(auth_token=token)
            await mcp.initialize()
            if relayed_project_id is not None:
                scoped = _project_scoped(await mcp.list_tools())
                if tool_name not in {t["name"] for t in scoped}:
                    raise exceptions.PermissionDenied(
                        f"'{tool_name}' is not available to the notebook agent."
                    )
            return await mcp.call_tool(tool_name, arguments)

        try:
            result = asyncio.run(_run())
        except exceptions.PermissionDenied:
            raise
        except Exception as e:
            logger.error("mcp-call error: %s", e, exc_info=True)
            return Response(
                {"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY
            )
        return Response({"result": result})


@method_decorator(csrf_exempt, name="dispatch")
class NotebookTokenView(APIView):
    """Relay the browser's Keycloak access token for a project's notebook kernel.

    The frontend posts this while the project's Jupyter tab is open (again on
    every token refresh). The caller's own bearer token is what gets stored,
    bound to the JupyterHub user of the caller's Jupyter space; the kernel-side
    MCP proxies then use it server-side (``_NotebookMCPView``). Only short-lived
    access tokens are stored, never refresh tokens. Relaying needs write access,
    so a visitor who may only read the project cannot replace the token its
    own users' kernels rely on.
    """

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = NotebookTokenSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        project = get_accessible_project(
            request, serializer.validated_data["project_id"], write=True
        )
        token = _extract_bearer_token(request)
        # Already verified by KeycloakAuthentication; only the expiry is needed.
        exp = jwt.decode(token, options={"verify_signature": False})["exp"]
        NotebookToken.objects.filter(expires_at__lte=timezone.now()).delete()
        stored, _ = NotebookToken.objects.update_or_create(
            project=project,
            defaults={
                "user": request.user,
                "hub_user": hub_username_for_tenant(get_user_tenant(request.user)),
                "access_token": token,
                "expires_at": datetime.fromtimestamp(exp, tz=dt_timezone.utc),
            },
        )
        return Response({"expires_at": stored.expires_at})


def _notebook_conversations():
    return (
        Conversation.objects.filter(
            source=Conversation.Source.NOTEBOOK, is_active=True
        )
        .select_related("user", "project")
        .annotate(message_count=Count("messages"))
    )


def _turn_title(messages) -> str:
    """First user message, cut like browser chat titles (``ChatStreamView``)."""
    text = next((m["content"] for m in messages if m["role"] == "user"), "")
    text = text.strip() or "Notebook chat"
    return text[:50] + ("..." if len(text) > 50 else "")


@method_decorator(csrf_exempt, name="dispatch")
class NotebookConversationListView(APIView):
    """Saved notebook agent conversations of a project.

    GET (browser): list them; anyone who can open the project may read them.
    POST: record one agent turn. Kernel path: the shared service token plus the
    kernel's JupyterHub token, accepted for active projects of that Jupyter
    space — no live relayed token is required, so recording keeps working after
    the browser token expires. The recorded user is only a label: whoever last
    relayed a token for the project in that space. User path: a Keycloak JWT
    (manual ``NEUROWORKFLOW_USER_TOKEN``) with write access to the project.
    """

    authentication_classes = [KeycloakAuthentication]

    def get_permissions(self):
        if self.request.method == "POST" and _service_token_ok(self.request):
            return []
        return [IsAuthenticated()]

    def get(self, request):
        serializer = NotebookTokenSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        project = get_accessible_project(
            request, serializer.validated_data["project_id"]
        )
        conversations = _notebook_conversations().filter(project=project)
        return Response(
            NotebookConversationListSerializer(
                conversations, many=True, context={"request": request}
            ).data
        )

    def _recording_target(self, request, project_id):
        """Return ``(project, user label or None, hub_user)`` for a new turn."""
        if not _service_token_ok(request):
            project = get_accessible_project(request, project_id, write=True)
            return project, request.user, ""
        hub_user = _kernel_hub_user(request)
        project = FlowProject.objects.filter(id=project_id, is_active=True).first()
        if project is None or hub_username_for_tenant(project.tenant) != hub_user:
            raise exceptions.NotFound("Project not found.")
        relay = (
            NotebookToken.objects.filter(project=project, hub_user=hub_user)
            .select_related("user")
            .first()
        )
        return project, (relay.user if relay else None), hub_user

    def post(self, request):
        serializer = NotebookTurnSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        project, user, hub_user = self._recording_target(request, data["project_id"])
        with transaction.atomic():
            conversation = None
            if data.get("conversation_id"):
                conversation = Conversation.objects.filter(
                    id=data["conversation_id"],
                    project=project,
                    source=Conversation.Source.NOTEBOOK,
                    is_active=True,
                ).first()
            created = conversation is None
            if created:
                conversation = Conversation.objects.create(
                    source=Conversation.Source.NOTEBOOK,
                    project=project,
                    user=user,
                    title=_turn_title(data["messages"]),
                )
            elif conversation.user_id is None and user is not None:
                conversation.user = user
            for key in ("model", "session_id"):
                if data.get(key):
                    conversation.metadata[key] = data[key]
            if hub_user:
                conversation.metadata["hub_user"] = hub_user
            conversation.save(update_fields=["metadata", "user", "updated_at"])
            for message in data["messages"]:
                Message.objects.create(
                    conversation=conversation,
                    role=message["role"],
                    content=message["content"],
                    tool_calls=message.get("tool_calls") or None,
                    tool_call_id=message["tool_call_id"],
                    tool_name=message["tool_name"],
                    raw_response={"is_error": True} if message["is_error"] else None,
                )
        return Response(
            {"id": str(conversation.id)},
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


@method_decorator(csrf_exempt, name="dispatch")
class NotebookConversationDetailView(APIView):
    """Read or delete one saved notebook agent conversation.

    Readable by anyone who can open its project; deletable (soft delete) by
    the project owner or the recorded user.
    """

    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def _get(self, request, conversation_id):
        conversation = _notebook_conversations().filter(id=conversation_id).first()
        if conversation is None or conversation.project_id is None:
            raise exceptions.NotFound("Conversation not found.")
        project = get_accessible_project(request, conversation.project_id)
        return conversation, project

    def get(self, request, conversation_id):
        conversation, _ = self._get(request, conversation_id)
        return Response(
            NotebookConversationSerializer(
                conversation, context={"request": request}
            ).data
        )

    def delete(self, request, conversation_id):
        conversation, project = self._get(request, conversation_id)
        if request.user.id not in (project.owner_id, conversation.user_id):
            raise exceptions.PermissionDenied(
                "Only the project owner or the recorded user can delete it."
            )
        conversation.is_active = False
        conversation.save(update_fields=["is_active"])
        return Response({"status": "deleted"})


ANTHROPIC_API_BASE = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com")

# Hop-by-hop / auth headers we must not forward upstream verbatim.
_PROXY_SKIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "x-api-key",
    "x-jupyterhub-token",
    "authorization",
    "cookie",
    "accept-encoding",
}


@method_decorator(csrf_exempt, name="dispatch")
class AnthropicProxyView(View):
    """Passthrough proxy: notebook kernel -> backend -> Anthropic API.

    The notebook Claude agent (claude-agent-sdk) runs inside a user kernel with
    ``ANTHROPIC_BASE_URL`` pointing here and ``ANTHROPIC_API_KEY`` set to the
    kernel's own hub token. We check that token with the hub and swap in the
    real Anthropic key, so the key is never exposed inside user-accessible
    kernels (mirroring why the OpenAI proxy keeps its key on the backend).
    The user's Keycloak JWT
    is not involved here — per-user workflow actions still go through the MCP
    proxies (``/api/chat/mcp-call/``) with the user's token.

    This is a plain Django ``View`` (not a DRF ``APIView``) so it relays the raw
    request/response body untouched, including the streaming ``/v1/messages`` SSE.
    Only ``POST`` to ``v1/messages`` and ``v1/messages/count_tokens`` is served
    (see ``urls.py``), so the key cannot reach the rest of the API.

    With ``provider="compat"`` (a second route) the same passthrough targets the
    Anthropic-compatible API of the OpenAI-compatible provider offering the
    requested model (see ``llm_providers``), with that provider's key. Only
    models on offer are forwarded.
    """

    provider = "anthropic"

    def dispatch(self, request, subpath=""):
        if request.method != "POST":
            return JsonResponse({"error": "Method not allowed"}, status=405)
        space = _kernel_space(request)
        if not space:
            return JsonResponse({"error": "Invalid kernel token"}, status=401)

        try:
            model = json.loads(request.body).get("model")
        except (ValueError, AttributeError):
            model = None
        if self.provider == "compat":
            # Only offered providers (which always have a key) are returned.
            provider = provider_for_model(model)
            if not provider or not provider.anthropic_base_url:
                return JsonResponse(
                    {"error": f"Model {model!r} is not available to the notebook"},
                    status=400,
                )
            base_url, real_key = provider.anthropic_base_url, provider.api_key
        else:
            base_url = ANTHROPIC_API_BASE
            real_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not real_key:
                return JsonResponse(
                    {"error": "ANTHROPIC_API_KEY is not configured on the backend"},
                    status=500,
                )

        url = f"{base_url}/{subpath}"
        query_string = request.META.get("QUERY_STRING", "")
        if query_string:
            url = f"{url}?{query_string}"

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _PROXY_SKIP_REQUEST_HEADERS
        }
        if self.provider == "compat":
            headers["authorization"] = f"Bearer {real_key}"
        else:
            headers["x-api-key"] = real_key
        headers.setdefault("anthropic-version", "2023-06-01")
        # Force an uncompressed upstream response: we forward the body without a
        # content-encoding header, so compressed bytes would reach the CLI as
        # unparseable garbage ("API Error: Failed to parse JSON"). identity also
        # keeps the SSE stream flushing in real time.
        headers["accept-encoding"] = "identity"

        client = httpx.Client(timeout=None)
        try:
            upstream = client.send(
                client.build_request(
                    request.method, url, headers=headers, content=request.body,
                ),
                stream=True,
            )
        except httpx.HTTPError as e:
            client.close()
            logger.error("%s proxy connection error: %s", self.provider, e)
            return JsonResponse({"error": f"Upstream error: {e}"}, status=502)
        logger.info(
            "%s proxy: space=%s model=%s path=%s status=%s",
            self.provider,
            space,
            model,
            subpath,
            upstream.status_code,
        )

        def _stream():
            try:
                # iter_bytes() yields content-decoded bytes (matches the absent
                # content-encoding header we send to the client).
                for chunk in upstream.iter_bytes():
                    yield chunk
            finally:
                upstream.close()
                client.close()

        response = StreamingHttpResponse(
            _stream(),
            status=upstream.status_code,
            content_type=upstream.headers.get("content-type", "application/json"),
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response
