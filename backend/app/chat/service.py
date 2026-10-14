from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from app.auth.principal import Principal
from app.config.assistant_config import AssistantConfig
from app.providers.base import (
    ChatTurn,
    ModelProvider,
    ModelRequest,
    ProviderError,
    StreamDelta,
    StreamEnd,
)
from app.services.conversations import ConversationStore, SavedTurn

GROUNDING_RULES = """
Rules that always apply:
- Answer ONLY from the CONTEXT section of the user's message. Never use general
  knowledge, even if you know the answer.
- If the CONTEXT is empty or does not answer the question, say plainly that you
  don't know based on the available documents, and stop.
- Do not write citation markers, source tags or bracketed references; the
  application shows sources separately.
- You are read-only: never claim to have changed, booked, sent or deleted anything.
""".strip()

NO_CONTEXT = "(no relevant documents were found)"


class AssistantNotFoundError(Exception):
    pass


class InputRejectedError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class ChatResult:
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    grounded: bool
    citations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: dict[str, Any]


@dataclass
class PreparedTurn:
    config: AssistantConfig
    provider: ModelProvider
    request: ModelRequest
    session_id: uuid.UUID | None
    message: str


ProviderFactory = Callable[[AssistantConfig], ModelProvider]


class ChatService:
    def __init__(
        self,
        assistants: dict[str, AssistantConfig],
        store: ConversationStore,
        provider_factory: ProviderFactory,
    ) -> None:
        self._assistants = assistants
        self._store = store
        self._provider_factory = provider_factory
        self._providers: dict[str, ModelProvider] = {}

    def assistants_for(self, principal: Principal) -> list[AssistantConfig]:
        return [a for a in self._assistants.values() if a.tenant_id == principal.tenant_id]

    def assistant(self, principal: Principal, assistant_id: str) -> AssistantConfig:
        config = self._assistants.get(assistant_id)
        if config is None or config.tenant_id != principal.tenant_id:
            raise AssistantNotFoundError
        return config

    def _provider(self, config: AssistantConfig) -> ModelProvider:
        if config.assistant_id not in self._providers:
            self._providers[config.assistant_id] = self._provider_factory(config)
        return self._providers[config.assistant_id]

    async def prepare(
        self,
        principal: Principal,
        assistant_id: str,
        message: str,
        session_id: uuid.UUID | None,
    ) -> PreparedTurn:
        """All checks that can fail with an HTTP status, before any model call."""
        config = self.assistant(principal, assistant_id)
        message = message.strip()
        if not message:
            raise InputRejectedError("message is empty")
        if len(message) > config.guardrails.max_input_chars:
            raise InputRejectedError(
                f"message exceeds {config.guardrails.max_input_chars} characters"
            )
        history = []
        if session_id is not None:
            history = await self._store.recent_history(principal, session_id, assistant_id)
        request = ModelRequest(
            system_prompt=f"{config.system_prompt.strip()}\n\n{GROUNDING_RULES}",
            history=[ChatTurn(m.role, m.content) for m in history],  # type: ignore[arg-type]
            # Retrieval arrives in Phase 2; until then the context is always empty.
            user_message=f"CONTEXT:\n{NO_CONTEXT}\n\nQUESTION:\n{message}",
        )
        return PreparedTurn(config, self._provider(config), request, session_id, message)

    async def _save(self, principal: Principal, turn: PreparedTurn, answer: str) -> SavedTurn:
        return await self._store.save_turn(
            principal,
            turn.config.assistant_id,
            turn.session_id,
            turn.message,
            answer,
            citations=[],
            grounded=False,
        )

    async def complete(self, principal: Principal, turn: PreparedTurn) -> ChatResult:
        reply = await turn.provider.generate(turn.request)
        saved = await self._save(principal, turn, reply.text)
        return ChatResult(saved.conversation_id, saved.assistant_message_id, reply.text, False)

    async def stream(self, principal: Principal, turn: PreparedTurn) -> AsyncIterator[SseEvent]:
        parts: list[str] = []
        try:
            async for event in turn.provider.stream(turn.request):
                if isinstance(event, StreamDelta):
                    parts.append(event.text)
                    yield SseEvent("delta", {"text": event.text})
                elif isinstance(event, StreamEnd):
                    break
            # Only a fully streamed answer is persisted.
            saved = await self._save(principal, turn, "".join(parts))
        except ProviderError as exc:
            yield SseEvent("error", {"kind": exc.kind, "message": "model provider error"})
            return
        except Exception:
            yield SseEvent("error", {"kind": "INTERNAL", "message": "internal error"})
            raise
        yield SseEvent(
            "done",
            {
                "session_id": str(saved.conversation_id),
                "message_id": str(saved.assistant_message_id),
                "grounded": False,
                "citations": [],
            },
        )
