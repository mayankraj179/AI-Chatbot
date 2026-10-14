from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

import jwt
import pytest
from alembic.config import Config
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from alembic import command
from app.auth.jwt_validator import JwtValidator
from app.config.loader import load_all_assistant_configs
from app.config.settings import Settings
from app.ingestion.service import IngestReport, ingest_assistant
from app.main import create_app
from app.providers.base import (
    ModelReply,
    ModelRequest,
    ProviderError,
    StreamDelta,
    StreamEnd,
    StreamEvent,
    TokenUsage,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Tests always use their own database, never the dev database.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://chatbot:chatbot_dev_password@localhost:5432/chatbot_test",
)
ISSUER = "http://issuer.test/realms/test"
AUDIENCE = "ai-chatbot-api"


class FakeSigningKey:
    def __init__(self, key: Any) -> None:
        self.key = key


class FakeJwksClient:
    def __init__(self, public_key: Any) -> None:
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token: str) -> FakeSigningKey:
        return FakeSigningKey(self._public_key)


class TokenFactory:
    def __init__(self) -> None:
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()

    def make(self, sub: str = "user-a", **overrides: Any) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": sub,
            "iat": now,
            "exp": now + 3600,
            "typ": "Bearer",
            "preferred_username": sub,
        }
        claims.update(overrides)
        claims = {k: v for k, v in claims.items() if v is not None}
        return jwt.encode(claims, self.private_key, algorithm="RS256")


class FakeEmbedder:
    """Deterministic bag-of-words embedding: shared words give high cosine similarity."""

    name = "fake"
    model_id = "fake:bow-64"

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def vector(text: str) -> list[float]:
        values = [0.0] * 64
        for word in re.findall(r"[a-z]+", text.lower()):
            if len(word) > 3:
                values[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0  # noqa: S324
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self.vector(text)


class FakeProvider:
    name = "fake"
    model_name = "fake-model"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.fail_with: ProviderError | None = None
        self.fail_mid_stream = False
        self.chunks = ["I don't know ", "based on the available documents."]

    async def generate(self, request: ModelRequest) -> ModelReply:
        self.requests.append(request)
        if self.fail_with:
            raise self.fail_with
        return ModelReply("".join(self.chunks), TokenUsage(10, 5, 0))

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if self.fail_with and not self.fail_mid_stream:
            raise self.fail_with
        for index, chunk in enumerate(self.chunks):
            if self.fail_mid_stream and index == 1:
                raise self.fail_with or ProviderError("OVERLOADED", "boom", provider="fake")
            yield StreamDelta(chunk)
        yield StreamEnd(TokenUsage(10, 5, 0))


@pytest.fixture(scope="session")
def tokens() -> TokenFactory:
    return TokenFactory()


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        _env_file=None,
        database_url=TEST_DATABASE_URL,
        assistant_configs_dir=str(FIXTURES / "configs"),
        content_root=str(FIXTURES / "content"),
        auth_issuer=ISSUER,
        auth_audience=AUDIENCE,
        auth_jwks_url="http://unused.test/certs",
        gemini_api_key=None,
    )


@pytest.fixture(scope="session")
def migrated_db() -> str:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    cfg.attributes["configure_logger"] = False
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    return TEST_DATABASE_URL


def _truncate(url: str) -> None:
    async def run() -> None:
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            await conn.exec_driver_sql(
                "TRUNCATE messages, conversations, chunks, documents CASCADE"
            )
        await engine.dispose()

    asyncio.run(run())


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def ingest_fixtures(url: str, settings: Settings, embedder: FakeEmbedder) -> IngestReport:
    async def run() -> IngestReport:
        engine = create_async_engine(url)
        try:
            configs = load_all_assistant_configs(settings.assistant_configs_dir)
            return await ingest_assistant(
                configs["leave_helper"],
                embedder,
                async_sessionmaker(engine, expire_on_commit=False),
                Path(settings.content_root),
            )
        finally:
            await engine.dispose()

    return asyncio.run(run())


@pytest.fixture
def client(
    migrated_db: str,
    settings: Settings,
    tokens: TokenFactory,
    provider: FakeProvider,
    embedder: FakeEmbedder,
) -> Iterator[TestClient]:
    _truncate(migrated_db)
    ingest_fixtures(migrated_db, settings, embedder)
    validator = JwtValidator(settings, jwks_client=FakeJwksClient(tokens.public_key))
    app = create_app(
        settings,
        jwt_validator=validator,
        provider_factory=lambda _cfg: provider,
        embedder_factory=lambda _cfg: embedder,
    )
    with TestClient(app) as test_client:
        yield test_client


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def random_uuid() -> str:
    return str(uuid.uuid4())
