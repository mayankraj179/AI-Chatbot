from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Asymmetric algorithms only: "none" and HS* (shared-secret) tokens are always refused.
SUPPORTED_JWT_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "ES256", "ES384", "PS256"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://chatbot:chatbot_dev_password@localhost:5432/chatbot"
    assistant_configs_dir: str = "./configs"

    auth_issuer: str = "http://localhost:8080/realms/ai-chatbot-dev"
    auth_audience: str = "ai-chatbot-api"
    auth_jwks_url: str = "http://localhost:8080/realms/ai-chatbot-dev/protocol/openid-connect/certs"
    auth_algorithms: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["RS256"])
    auth_jwks_cache_seconds: int = Field(default=300, gt=0)
    auth_leeway_seconds: int = Field(default=30, ge=0, le=120)
    auth_admin_role: str = "chatbot-admin"

    gemini_api_key: str | None = None
    # Knowledge source paths in assistant YAML are relative to this directory.
    content_root: str = "../content"
    embedding_cache_dir: str = ".cache/fastembed"

    @field_validator("auth_algorithms", mode="before")
    @classmethod
    def _split_algorithms(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("auth_algorithms")
    @classmethod
    def _asymmetric_only(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("AUTH_ALGORITHMS must list at least one algorithm")
        unsupported = sorted(set(value) - SUPPORTED_JWT_ALGORITHMS)
        if unsupported:
            raise ValueError(
                f"unsupported AUTH_ALGORITHMS {unsupported}; allowed: "
                f"{sorted(SUPPORTED_JWT_ALGORITHMS)}"
            )
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
