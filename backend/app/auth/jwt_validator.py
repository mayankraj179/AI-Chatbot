from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jwt
from jwt import PyJWKClient

from app.auth.principal import (
    PERMISSION_ADMIN,
    PERMISSION_CHAT,
    PERMISSION_CONVERSATION_READ,
    PERMISSION_CONVERSATION_WRITE,
    Principal,
)
from app.config.settings import Settings

DEFAULT_TENANT = "local-development"
_USER_PERMISSIONS = frozenset(
    {PERMISSION_CHAT, PERMISSION_CONVERSATION_READ, PERMISSION_CONVERSATION_WRITE}
)


class AuthenticationError(Exception):
    """Token missing or invalid -> HTTP 401. ``detail`` is safe to return to clients."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class IdentityProviderUnavailableError(Exception):
    """Signing keys could not be fetched -> HTTP 503."""


class JwtValidator:
    """Validates bearer tokens locally against the IdP's cached JWKS."""

    def __init__(self, settings: Settings, jwks_client: PyJWKClient | None = None) -> None:
        self._settings = settings
        self._algorithms = list(settings.auth_algorithms)
        self._jwks_client = jwks_client or PyJWKClient(
            settings.auth_jwks_url,
            cache_jwk_set=True,
            lifespan=settings.auth_jwks_cache_seconds,
            cache_keys=True,
            timeout=5,
        )

    def validate(self, token: str) -> dict[str, Any]:
        if not token:
            raise AuthenticationError("missing bearer token")
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") not in self._algorithms:
                raise AuthenticationError("token signing algorithm not allowed")
            key = self._jwks_client.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key=key,
                algorithms=self._algorithms,
                audience=self._settings.auth_audience,
                issuer=self._settings.auth_issuer,
                leeway=self._settings.auth_leeway_seconds,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except AuthenticationError:
            raise
        except jwt.PyJWKClientConnectionError as exc:
            raise IdentityProviderUnavailableError("could not fetch signing keys") from exc
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token has expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthenticationError("invalid token audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthenticationError("invalid token issuer") from exc
        except (jwt.InvalidTokenError, jwt.PyJWKClientError) as exc:
            raise AuthenticationError("invalid token") from exc
        # Deeply nested or malformed payloads can escape PyJWT's own handling.
        except (RecursionError, TypeError, ValueError, KeyError) as exc:
            raise AuthenticationError("invalid token") from exc

        token_type = claims.get("typ")
        if token_type is not None and str(token_type).lower() != "bearer":
            raise AuthenticationError("not an access token")
        return claims

    def principal_from_claims(self, claims: Mapping[str, Any]) -> Principal:
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise AuthenticationError("token has no subject")
        tenant = claims.get("tenant_id")
        tenant_id = tenant if isinstance(tenant, str) and tenant else DEFAULT_TENANT

        roles = _string_set(_nested(claims, "realm_access", "roles"))
        groups = _string_set(claims.get("groups"))
        labels = {f"user:{subject}", f"tenant:{tenant_id}"}
        labels |= {f"role:{role}" for role in roles}
        labels |= {f"group:{group.strip('/')}" for group in groups if group.strip("/")}

        permissions = set(_USER_PERMISSIONS)
        if self._settings.auth_admin_role in roles:
            permissions.add(PERMISSION_ADMIN)

        username = claims.get("preferred_username")
        return Principal(
            tenant_id=tenant_id,
            principal_id=subject,
            labels=frozenset(labels),
            permissions=frozenset(permissions),
            username=username if isinstance(username, str) else None,
        )


def _nested(claims: Mapping[str, Any], *keys: str) -> Any:
    node: Any = claims
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _string_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list | tuple):
        return {item for item in value if isinstance(item, str) and item and " " not in item}
    return set()
