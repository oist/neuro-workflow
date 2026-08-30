"""Authenticate Jupyter contents-filter calls with a signed viewer token."""

from rest_framework import authentication, exceptions

from .viewer_tokens import ViewerTokenError, user_from_viewer_token


class JupyterViewerTokenAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        token = _extract_viewer_token(request)
        if not token:
            return None
        try:
            user, payload = user_from_viewer_token(token)
        except ViewerTokenError as exc:
            raise exceptions.AuthenticationFailed(str(exc)) from exc
        request.viewer_payload = payload
        return (user, token)

    def authenticate_header(self, request):
        return 'Viewer realm="jupyter"'


def _extract_viewer_token(request) -> str | None:
    header = authentication.get_authorization_header(request).decode("utf-8")
    if header.lower().startswith("viewer "):
        return header.split(" ", 1)[1].strip() or None
    token = request.META.get("HTTP_X_NW_VIEWER_TOKEN") or request.GET.get("token")
    if token:
        return token
    return None
