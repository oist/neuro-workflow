from django.core.exceptions import ValidationError as DjangoValidationError
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from app.auth.authentication import KeycloakAuthentication

from .models import UserSecret
from .serializers import UserSecretSerializer
from .services import (
    client_ip,
    create_user_secret,
    get_owned_secret,
    owner_secrets_qs,
    revoke_user_secret,
    rotate_user_secret,
)


def _not_found():
    return Response({"error": "Secret not found."}, status=status.HTTP_404_NOT_FOUND)


class UserSecretListCreateView(APIView):
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = owner_secrets_qs(request.user)
        return Response(UserSecretSerializer(qs, many=True).data)

    def post(self, request):
        ser = UserSecretSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        value = ser.validated_data.get("value")
        if not value:
            return Response({"error": "value is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            secret = create_user_secret(
                request.user,
                name=ser.validated_data["name"],
                value=value,
                description=ser.validated_data.get("description") or "",
                actor=request.user,
                ip=client_ip(request),
            )
        except DjangoValidationError as exc:
            return Response({"error": exc.message_dict if hasattr(exc, "message_dict") else str(exc)}, status=400)
        return Response(UserSecretSerializer(secret).data, status=status.HTTP_201_CREATED)


class UserSecretDetailView(APIView):
    authentication_classes = [KeycloakAuthentication]
    permission_classes = [IsAuthenticated]

    def _load(self, request, secret_id) -> UserSecret | None:
        return get_owned_secret(request.user, secret_id)

    def get(self, request, secret_id):
        secret = self._load(request, secret_id)
        if secret is None or secret.revoked_at is not None:
            return _not_found()
        return Response(UserSecretSerializer(secret).data)

    def patch(self, request, secret_id):
        secret = self._load(request, secret_id)
        if secret is None:
            return _not_found()
        value = request.data.get("value")
        description = request.data.get("description")
        if value is None and description is None:
            return Response({"error": "value or description is required."}, status=400)
        try:
            secret = rotate_user_secret(
                secret,
                value=value,
                description=description,
                actor=request.user,
                ip=client_ip(request),
            )
        except DjangoValidationError as exc:
            return Response({"error": str(exc)}, status=400)
        return Response(UserSecretSerializer(secret).data)

    def delete(self, request, secret_id):
        secret = self._load(request, secret_id)
        if secret is None:
            return _not_found()
        revoke_user_secret(secret, actor=request.user, ip=client_ip(request))
        return Response(status=status.HTTP_204_NO_CONTENT)
