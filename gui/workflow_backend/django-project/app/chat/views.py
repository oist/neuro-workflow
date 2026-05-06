import asyncio
import json
import logging

from django.http import StreamingHttpResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import authentication, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.auth.authentication import CombinedJWTAuthentication

from .models import Conversation, Message
from .serializers import (
    ConversationSerializer,
    ConversationListSerializer,
    SendMessageSerializer,
)
from .services.chat_orchestrator import orchestrate_chat

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


@method_decorator(csrf_exempt, name="dispatch")
class ConversationListCreateView(APIView):
    """List and create conversations."""

    authentication_classes = [CombinedJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        conversations = Conversation.objects.filter(user=request.user, is_active=True)
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

    authentication_classes = [CombinedJWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request, conversation_id):
        try:
            conversation = Conversation.objects.get(
                id=conversation_id, user=request.user, is_active=True,
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
            )
        except Conversation.DoesNotExist:
            return Response(
                {"error": "Conversation not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        conversation.is_active = False
        conversation.save()
        return Response({"status": "deleted"}, status=status.HTTP_200_OK)


@method_decorator(csrf_exempt, name="dispatch")
class ChatStreamView(APIView):
    """Handle chat messages with SSE streaming response."""

    authentication_classes = [CombinedJWTAuthentication]
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

        # Get or create conversation
        if conversation_id:
            try:
                conversation = Conversation.objects.get(
                    id=conversation_id, user=user, is_active=True,
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
            self._sync_event_generator(conversation, user_message, auth_token),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        response["X-Conversation-Id"] = str(conversation.id)
        return response

    def _sync_event_generator(self, conversation, user_message, auth_token):
        """Wrap the async orchestrator into a sync generator for WSGI."""
        loop = asyncio.new_event_loop()

        try:
            # Send the conversation ID as the first event
            yield _format_sse("conversation_id", {"id": str(conversation.id)})

            agen = orchestrate_chat(conversation, user_message, auth_token=auth_token)

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


def _format_sse(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
