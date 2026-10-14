from __future__ import annotations

import base64
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth.dependencies import get_principal, require_permission
from app.auth.jwt_validator import AuthenticationError, JwtValidator
from app.auth.principal import PERMISSION_ADMIN, Principal, ZeroLabelPrincipalError
from app.config.settings import Settings
from tests.conftest import AUDIENCE, ISSUER, FakeJwksClient, TokenFactory


@pytest.fixture
def validator(settings: Settings, tokens: TokenFactory) -> JwtValidator:
    return JwtValidator(settings, jwks_client=FakeJwksClient(tokens.public_key))


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def test_valid_token_resolves_principal(validator: JwtValidator, tokens: TokenFactory) -> None:
    claims = validator.validate(tokens.make("alice", realm_access={"roles": ["employee"]}))
    principal = validator.principal_from_claims(claims)
    assert principal.principal_id == "alice"
    assert principal.tenant_id == "local-development"
    assert {"user:alice", "tenant:local-development", "role:employee"} <= principal.labels
    assert not principal.has_permission(PERMISSION_ADMIN)


def test_admin_role_grants_admin_permission(validator: JwtValidator, tokens: TokenFactory) -> None:
    claims = validator.validate(tokens.make("root", realm_access={"roles": ["chatbot-admin"]}))
    assert validator.principal_from_claims(claims).has_permission(PERMISSION_ADMIN)


@pytest.mark.parametrize(
    ("overrides", "detail"),
    [
        ({"exp": int(time.time()) - 3600}, "expired"),
        ({"aud": "someone-else"}, "audience"),
        ({"iss": "http://evil.test"}, "issuer"),
        ({"sub": None}, "invalid token"),
        ({"exp": None}, "invalid token"),
        ({"typ": "Refresh"}, "not an access token"),
    ],
)
def test_bad_claims_rejected(
    validator: JwtValidator, tokens: TokenFactory, overrides: dict, detail: str
) -> None:
    with pytest.raises(AuthenticationError, match=detail):
        validator.validate(tokens.make(**overrides))


def test_token_signed_by_other_key_rejected(validator: JwtValidator) -> None:
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = int(time.time())
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "iat": now, "exp": now + 60},
        other,
        algorithm="RS256",
    )
    with pytest.raises(AuthenticationError):
        validator.validate(token)


def test_alg_none_and_hs256_rejected(validator: JwtValidator) -> None:
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": "x", "iat": now, "exp": now + 60}
    unsigned = jwt.encode(claims, key=None, algorithm="none")
    hs = jwt.encode(claims, "shared-secret-shared-secret-shared-secret", algorithm="HS256")
    for token in (unsigned, hs):
        with pytest.raises(AuthenticationError, match="algorithm not allowed"):
            validator.validate(token)


@pytest.mark.parametrize("token", ["", "garbage", "a.b.c", "eyJhbGciOiJSUzI1NiJ9.!!!.sig"])
def test_malformed_tokens_rejected(validator: JwtValidator, token: str) -> None:
    with pytest.raises(AuthenticationError):
        validator.validate(token)


def test_deeply_nested_token_is_401_not_crash(validator: JwtValidator) -> None:
    header = _b64(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _b64(b'{"a":' * 100_000 + b"1" + b"}" * 100_000)
    nested_header = _b64(b'{"alg":' + b"[" * 100_000 + b"]" * 100_000 + b"}")
    for token in (f"{header}.{payload}.c2ln", f"{nested_header}.{payload}.c2ln"):
        with pytest.raises(AuthenticationError):
            validator.validate(token)


def test_zero_label_principal_refused() -> None:
    with pytest.raises(ZeroLabelPrincipalError):
        Principal(tenant_id="t", principal_id="p", labels=frozenset())
    with pytest.raises(ValueError, match="namespaced"):
        Principal(tenant_id="t", principal_id="p", labels=frozenset({"employee"}))


def _protected_app(validator: JwtValidator) -> TestClient:
    app = FastAPI()
    app.state.jwt_validator = validator

    @app.get("/me")
    def me(principal: Principal = Depends(get_principal)) -> dict:
        return {"sub": principal.principal_id}

    @app.get("/admin-only")
    def admin(principal: Principal = Depends(require_permission(PERMISSION_ADMIN))) -> dict:
        return {"ok": True}

    return TestClient(app)


def test_http_status_codes(validator: JwtValidator, tokens: TokenFactory) -> None:
    client = _protected_app(validator)
    assert client.get("/me").status_code == 401
    assert client.get("/me").headers["www-authenticate"] == "Bearer"
    assert client.get("/me", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.get("/me", headers={"Authorization": "Bearer "}).status_code == 401
    user = {"Authorization": f"Bearer {tokens.make('alice')}"}
    assert client.get("/me", headers=user).json() == {"sub": "alice"}
    assert client.get("/admin-only", headers=user).status_code == 403
    admin_token = tokens.make("root", realm_access={"roles": ["chatbot-admin"]})
    admin = {"Authorization": f"Bearer {admin_token}"}
    assert client.get("/admin-only", headers=admin).status_code == 200
