from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status

from app.auth.jwt_validator import (
    AuthenticationError,
    IdentityProviderUnavailableError,
    JwtValidator,
)
from app.auth.principal import Principal, ZeroLabelPrincipalError

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def get_jwt_validator(request: Request) -> JwtValidator:
    return request.app.state.jwt_validator


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization")
    if not header:
        raise AuthenticationError("missing Authorization header")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Authorization header must be 'Bearer <token>'")
    return token.strip()


# Sync on purpose: a JWKS cache miss does blocking I/O, so FastAPI runs this in its
# threadpool instead of on the event loop.
def get_principal(
    request: Request, validator: JwtValidator = Depends(get_jwt_validator)
) -> Principal:
    try:
        claims = validator.validate(_bearer_token(request))
        return validator.principal_from_claims(claims)
    except AuthenticationError as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail=exc.detail, headers=_UNAUTHORIZED_HEADERS
        ) from None
    except ZeroLabelPrincipalError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="no access labels") from None
    except IdentityProviderUnavailableError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="identity provider unavailable"
        ) from None


def require_permission(permission: str) -> Callable[[Principal], Principal]:
    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.has_permission(permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="permission denied")
        return principal

    return dependency
