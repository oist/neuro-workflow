from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response


@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def health_check(request):
    """Authentication-free health checks"""
    return Response({"status": "ok", "message": "Django server is running"})


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def protected_view(request):
    """Endpoints that require authentication"""
    return Response(
        {
            "message": "Hello authenticated user!",
            "user": {
                "id": request.user.id,
                "username": request.user.username,
                "email": request.user.email,
                "is_authenticated": request.user.is_authenticated,
            },
        }
    )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """User information acquisition"""
    return Response(
        {
            "user": {
                "id": request.user.id,
                "username": request.user.username,
                "email": request.user.email,
                "first_name": request.user.first_name,
                "last_name": request.user.last_name,
                "date_joined": request.user.date_joined.isoformat(),
                "last_login": (
                    request.user.last_login.isoformat()
                    if request.user.last_login
                    else None
                ),
            }
        }
    )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def test_post(request):
    """POST Testing the request"""
    return Response(
        {
            "message": "POST request successful",
            "data": request.data,
            "user": request.user.username,
        }
    )
