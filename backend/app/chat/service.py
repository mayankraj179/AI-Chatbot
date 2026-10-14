from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from app.auth.principal import Principal
from app.config.assistant_config import AssistantConfig
from app.providers.base import (
    ChatTurn,
    EmbeddingProvider,
    ModelProvider,
    ModelRequest,
    ProviderError,
    StreamDelta,
    StreamEnd,
)
from app.services.conversations import ConversationStore, SavedTurn
from app.services.retrieval import RetrievedChunk, Retriever

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
# Sent without a model call when retrieval finds nothing the user may see.
DONT_KNOW = "I don't know. I couldn't find anything about that in the documents available to you."


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
    provider: ModelProvider | None
    request: ModelRequest
    session_id: uuid.UUID | None
    message: str
    grounded: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    canned_answer: str | None = None


ProviderFactory = Callable[[AssistantConfig], ModelProvider]
EmbedderFactory = Callable[[AssistantConfig], EmbeddingProvider]


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        location = " > ".join((chunk.title, *chunk.heading_path))
        if chunk.page is not None:
            location += f" (page {chunk.page})"
        blocks.append(f"[{number}] {location}\n{chunk.text}")
    return "\n\n".join(blocks)


class ChatService:
    def __init__(
        self,
        assistants: dict[str, AssistantConfig],
        store: ConversationStore,
        provider_factory: ProviderFactory,
        retriever: Retriever | None = None,
        embedder_factory: EmbedderFactory | None = None,
    ) -> None:
        self._assistants = assistants
        self._store = store
        self._provider_factory = provider_factory
        self._providers: dict[str, ModelProvider] = {}
        self._retriever = retriever
        self._embedder_factory = embedder_factory
        self._embedders: dict[str, EmbeddingProvider] = {}

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

    def _embedder(self, config: AssistantConfig) -> EmbeddingProvider:
        if self._embedder_factory is None:
            raise RuntimeError("retrieval is enabled but no embedder factory is configured")
        key = f"{config.retrieval.embedding_provider}:{config.retrieval.embedding_model}"
        if key not in self._embedders:
            self._embedders[key] = self._embedder_factory(config)
        return self._embedders[key]

    async def _retrieve(
        self, principal: Principal, config: AssistantConfig, message: str
    ) -> list[RetrievedChunk]:
        if not config.retrieval.enabled or self._retriever is None:
            return []
        embedder = self._embedder(config)
        query_vector = await embedder.embed_query(message)
        return await self._retriever.search(
            tenant_id=principal.tenant_id,
            assistant_id=config.assistant_id,
            labels=sorted(principal.labels),
            embedding_model=embedder.model_id,
            query_vector=query_vector,
            top_k=config.retrieval.top_k,
            min_similarity=config.retrieval.min_similarity or 0.0,
        )

    async def search(
        self, principal: Principal, assistant_id: str, query: str
    ) -> list[RetrievedChunk]:
        config = self.assistant(principal, assistant_id)
        query = query.strip()
        if not query:
            raise InputRejectedError("query is empty")
        if len(query) > config.guardrails.max_input_chars:
            raise InputRejectedError(
                f"query exceeds {config.guardrails.max_input_chars} characters"
            )
        return await self._retrieve(principal, config, query)

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
        chunks = await self._retrieve(principal, config, message)
        context = build_context(chunks) if chunks else NO_CONTEXT
        request = ModelRequest(
            system_prompt=f"{config.system_prompt.strip()}\n\n{GROUNDING_RULES}",
            history=[ChatTurn(m.role, m.content) for m in history],  # type: ignore[arg-type]
            user_message=f"CONTEXT:\n{context}\n\nQUESTION:\n{message}",
        )
        if config.retrieval.enabled and not chunks:
            return PreparedTurn(config, None, request, session_id, message, canned_answer=DONT_KNOW)
        return PreparedTurn(
            config,
            self._provider(config),
            request,
            session_id,
            message,
            grounded=bool(chunks),
            citations=[c.citation(n) for n, c in enumerate(chunks, start=1)],
        )

    async def _save(self, principal: Principal, turn: PreparedTurn, answer: str) -> SavedTurn:
        return await self._store.save_turn(
            principal,
            turn.config.assistant_id,
            turn.session_id,
            turn.message,
            answer,
            citations=turn.citations,
            grounded=turn.grounded,
        )

    async def complete(self, principal: Principal, turn: PreparedTurn) -> ChatResult:
        if turn.canned_answer is not None:
            answer = turn.canned_answer
        else:
            assert turn.provider is not None
            answer = (await turn.provider.generate(turn.request)).text
        saved = await self._save(principal, turn, answer)
        return ChatResult(
            saved.conversation_id, saved.assistant_message_id, answer, turn.grounded, turn.citations
        )

    async def stream(self, principal: Principal, turn: PreparedTurn) -> AsyncIterator[SseEvent]:
        parts: list[str] = []
        try:
            if turn.canned_answer is not None:
                parts.append(turn.canned_answer)
                yield SseEvent("delta", {"text": turn.canned_answer})
            else:
                assert turn.provider is not None
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
                "grounded": turn.grounded,
                "citations": turn.citations,
            },
        )
