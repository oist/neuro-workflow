import jwt
import requests
from django.contrib.auth import get_user_model
from rest_framework import authentication, exceptions
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_jwks_cache: dict = {}


def _fetch_jwks(jwks_url: str) -> list:
    """Fetch and cache JWKS keys from a remote endpoint."""
    cached = _jwks_cache.get(jwks_url)
    if cached:
        return cached
    try:
        resp = requests.get(jwks_url, timeout=5)
        resp.raise_for_status()
        keys = resp.json().get("keys", [])
    except Exception as e:
        raise jwt.InvalidTokenError(f"Failed to fetch JWKS from {jwks_url}: {e}")
    if keys:
        _jwks_cache[jwks_url] = keys
    return keys


def _verify_rs256(token: str, jwks_url: str, **decode_kwargs) -> dict:
    """Verify a JWT using RS256 against a JWKS endpoint."""
    keys = _fetch_jwks(jwks_url)
    unverified = jwt.get_unverified_header(token)
    kid = unverified.get("kid")
    public_key = None
    for key in keys:
        if key.get("kid") == kid:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            break
    if not public_key:
        _jwks_cache.pop(jwks_url, None)
        keys = _fetch_jwks(jwks_url)
        for key in keys:
            if key.get("kid") == kid:
                public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
                break
    if not public_key:
        raise jwt.InvalidTokenError("No matching key found in JWKS")
    return jwt.decode(token, public_key, algorithms=["RS256"], **decode_kwargs)


def _get_or_create_user(user_id: str, email: str, extra: dict | None = None):
    """Get-or-create a Django User from an external identity provider."""
    User = get_user_model()
    extra = extra or {}

    try:
        return User.objects.get(username=user_id)
    except User.DoesNotExist:
        pass

    if email:
        try:
            user = User.objects.get(email=email)
            user.username = user_id
            user.save(update_fields=["username"])
            return user
        except User.DoesNotExist:
            pass

    return User.objects.create_user(
        username=user_id,
        email=email or "",
        first_name=extra.get("first_name", ""),
        last_name=extra.get("last_name", ""),
    )


# ---------------------------------------------------------------------------
# Base bearer-token authenticator
# ---------------------------------------------------------------------------

class _BearerAuthentication(authentication.BaseAuthentication):
    """Extract a Bearer token from the Authorization header."""

    def authenticate(self, request):
        auth_header = authentication.get_authorization_header(request).split()
        if not auth_header or auth_header[0].lower() != b"bearer":
            return None
        if len(auth_header) != 2:
            raise exceptions.AuthenticationFailed("Malformed Authorization header.")
        try:
            token = auth_header[1].decode("utf-8")
        except UnicodeError:
            raise exceptions.AuthenticationFailed("Invalid characters in token.")
        return self.authenticate_credentials(token)

    def authenticate_credentials(self, token):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Keycloak OIDC authentication
# ---------------------------------------------------------------------------

class KeycloakAuthentication(_BearerAuthentication):
    """Verify JWTs issued by a Keycloak realm (RS256 via JWKS)."""

    def authenticate_credentials(self, token):
        kc_url = getattr(settings, "KEYCLOAK_URL", None)
        kc_realm = getattr(settings, "KEYCLOAK_REALM", None)
        if not kc_url or not kc_realm:
            return None

        jwks_url = f"{kc_url}/realms/{kc_realm}/protocol/openid-connect/certs"
        issuer = f"{kc_url}/realms/{kc_realm}"
        kc_client = getattr(settings, "KEYCLOAK_CLIENT_ID", None)

        decode_kwargs: dict = {"issuer": issuer}
        if kc_client:
            decode_kwargs["audience"] = kc_client

        try:
            payload = _verify_rs256(token, jwks_url, **decode_kwargs)
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("Token has expired.")
        except jwt.InvalidTokenError as e:
            raise exceptions.AuthenticationFailed(f"Invalid token: {e}")

        sub = payload.get("sub")
        email = payload.get("email", "")
        preferred = payload.get("preferred_username", "")
        user = _get_or_create_user(
            user_id=sub,
            email=email,
            extra={
                "first_name": payload.get("given_name", ""),
                "last_name": payload.get("family_name", ""),
            },
        )
        return (user, token)


# ---------------------------------------------------------------------------
# Supabase authentication (kept for backward compatibility)
# ---------------------------------------------------------------------------

class SupabaseAuthentication(_BearerAuthentication):
    """Verify JWTs issued by Supabase (HS256 secret or RS256 JWKS)."""

    def authenticate_credentials(self, token):
        supabase_url = getattr(settings, "SUPABASE_URL", None)
        jwt_secret = getattr(settings, "SUPABASE_JWT_SECRET", None)
        if not supabase_url:
            return None

        decode_kwargs = dict(
            audience="authenticated",
            issuer=f"{supabase_url}/auth/v1",
        )

        try:
            if jwt_secret:
                try:
                    payload = jwt.decode(
                        token, jwt_secret, algorithms=["HS256"], **decode_kwargs,
                    )
                    return self._resolve_user(payload, token)
                except jwt.InvalidTokenError:
                    logger.debug("HS256 failed, trying JWKS")

            jwks_url = f"{supabase_url}/auth/v1/jwks"
            payload = _verify_rs256(token, jwks_url, **decode_kwargs)
            return self._resolve_user(payload, token)

        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("Token has expired.")
        except jwt.InvalidTokenError as e:
            raise exceptions.AuthenticationFailed(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Supabase auth error: {e}")
            raise exceptions.AuthenticationFailed("Authentication failed.")

    @staticmethod
    def _resolve_user(payload, token):
        sub = payload.get("sub")
        email = payload.get("email")
        if not sub or not email:
            raise exceptions.AuthenticationFailed("Invalid token payload.")
        user = _get_or_create_user(
            user_id=sub,
            email=email,
            extra={
                "first_name": payload.get("user_metadata", {}).get("first_name", ""),
                "last_name": payload.get("user_metadata", {}).get("last_name", ""),
            },
        )
        return (user, token)


# ---------------------------------------------------------------------------
# Combined authentication: try Keycloak first, then Supabase
# ---------------------------------------------------------------------------

class CombinedJWTAuthentication(authentication.BaseAuthentication):
    """Attempt Keycloak first, then Supabase. Returns None if both decline so
    that ``permission_classes = [IsAuthenticated]`` produces a 401."""

    def __init__(self):
        self._backends = (KeycloakAuthentication(), SupabaseAuthentication())

    def authenticate(self, request):
        for backend in self._backends:
            try:
                result = backend.authenticate(request)
            except exceptions.AuthenticationFailed:
                continue
            if result is not None:
                return result
        return None

    def authenticate_header(self, request):
        return 'Bearer realm="api"'
