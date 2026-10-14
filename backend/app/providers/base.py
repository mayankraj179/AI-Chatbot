from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    role: Role
    content: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True)
class ToolExchange:
    """One executed tool call and its result, sent back to the model."""

    call: ToolCall
    result: dict[str, Any]


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    history: Sequence[ChatTurn]
    user_message: str
    tools: Sequence[ToolSpec] = ()
    tool_exchanges: Sequence[ToolExchange] = ()


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class ModelReply:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class StreamDelta:
    text: str


@dataclass(frozen=True)
class StreamEnd:
    usage: TokenUsage


StreamEvent = StreamDelta | StreamEnd

ProviderErrorKind = Literal[
    "INVALID_KEY", "QUOTA_EXCEEDED", "RATE_LIMITED", "OVERLOADED", "BAD_REQUEST", "PROVIDER_ERROR"
]


class ProviderError(Exception):
    """A classified model/embedding provider failure. ``message`` is safe to log."""

    def __init__(self, kind: ProviderErrorKind, message: str, *, provider: str) -> None:
        super().__init__(f"{provider} {kind}: {message}")
        self.kind = kind
        self.message = message
        self.provider = provider


class ModelProvider(Protocol):
    name: str
    model_name: str

    async def generate(self, request: ModelRequest) -> ModelReply: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]: ...


class EmbeddingProvider(Protocol):
    """Implemented in Phase 2. Queries and documents use different task types."""

    name: str
    model_id: str

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
