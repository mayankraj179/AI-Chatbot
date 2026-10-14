from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from app.auth.principal import Principal
from app.chat.greeting import greeting_reply, is_greeting
from app.config.assistant_config import AssistantConfig
from app.guardrails.input import screen_input
from app.guardrails.output import MarkerStripper, is_refusal, screen_output, strip_markers
from app.providers.base import (
    ChatTurn,
    EmbeddingProvider,
    ModelProvider,
    ModelRequest,
    ProviderError,
    StreamDelta,
    StreamEnd,
    ToolExchange,
)
from app.services.conversations import ConversationStore, SavedTurn
from app.services.retrieval import RetrievedChunk, Retriever
from app.tools.registry import TOOLS, may_need_tool, run_tool

logger = logging.getLogger(__name__)

GROUNDING_RULES = """
Rules that always apply:
- Answer ONLY from the CONTEXT section of the user's message or from results of the
  tools you call. Never use general knowledge, even if you know the answer.
- If neither the CONTEXT nor a tool result answers the question, say plainly that you
  don't know based on the available documents, and stop.
- For yes/no questions, start with "Yes" or "No" and name the specific status or value
  from the CONTEXT (for example "No. 3 March 2026 is a Floating Holiday.").
- Use the calculate tool for arithmetic and the current_datetime tool for today's
  date or time, when those tools are available; do not compute or guess them yourself.
- Do not write citation markers, source tags or bracketed references; the
  application shows sources separately.
- You are read-only: never claim to have changed, booked, sent or deleted anything.
""".strip()

NO_CONTEXT = "(no relevant documents were found)"
# Sent without a model call when retrieval finds nothing the user may see.
DONT_KNOW = "I don't know. I couldn't find anything about that in the documents available to you."
MAX_TOOL_ROUNDS = 3


class AssistantNotFoundError(Exception):
    pass


class InputRejectedError(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class ChatResult:
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    grounded: bool
    citations: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "answered"
    tools_used: list[str] = field(default_factory=list)


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
    chunks: list[RetrievedChunk] = field(default_factory=list)
    canned_answer: str | None = None
    canned_outcome: str = "no_context"


@dataclass(frozen=True)
class FinalAnswer:
    text: str
    grounded: bool
    citations: list[dict[str, Any]]
    outcome: str
    tools_used: list[str]


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


def finalize(turn: PreparedTurn, raw_text: str, tools_used: list[str]) -> FinalAnswer:
    """Output guardrails on the full answer text."""
    text = strip_markers(raw_text).strip()
    if not turn.chunks and turn.request.tools and not tools_used:
        # A tool-enabled assistant with zero chunks may only answer through a tool.
        return FinalAnswer(DONT_KNOW, False, [], "no_context", [])
    if not text:
        return FinalAnswer(DONT_KNOW, False, [], "no_context", tools_used)
    verdict = screen_output(text, turn.config.guardrails)
    if verdict is not None:
        return FinalAnswer(verdict.message, False, [], verdict.reason, tools_used)
    if is_refusal(text):
        return FinalAnswer(text, False, [], "refused", tools_used)
    if not turn.chunks:
        return FinalAnswer(text, False, [], "tool_answer", tools_used)
    citations = [c.citation(n) for n, c in enumerate(turn.chunks, start=1)]
    return FinalAnswer(text, True, citations, "answered", tools_used)


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

    @staticmethod
    def _check_length(config: AssistantConfig, text: str) -> str:
        text = text.strip()
        if not text:
            raise InputRejectedError("empty", "message is empty")
        if len(text) > config.guardrails.max_input_chars:
            raise InputRejectedError(
                "too_long", f"message exceeds {config.guardrails.max_input_chars} characters"
            )
        return text

    async def search(
        self, principal: Principal, assistant_id: str, query: str
    ) -> list[RetrievedChunk]:
        config = self.assistant(principal, assistant_id)
        return await self._retrieve(principal, config, self._check_length(config, query))

    async def prepare(
        self,
        principal: Principal,
        assistant_id: str,
        message: str,
        session_id: uuid.UUID | None,
    ) -> PreparedTurn:
        """All checks that can fail with an HTTP status, before any model call."""
        config = self.assistant(principal, assistant_id)
        message = self._check_length(config, message)
        history = []
        if session_id is not None:
            history = await self._store.recent_history(principal, session_id, assistant_id)
        request = ModelRequest(
            system_prompt=f"{config.system_prompt.strip()}\n\n{GROUNDING_RULES}",
            history=[ChatTurn(m.role, m.content) for m in history],  # type: ignore[arg-type]
            user_message="",
        )
        if is_greeting(message, config.display_name):
            return PreparedTurn(
                config,
                None,
                request,
                session_id,
                message,
                canned_answer=greeting_reply(config.display_name),
                canned_outcome="greeting",
            )
        verdict = screen_input(message, config.guardrails)
        if verdict is not None:
            raise InputRejectedError(verdict.reason, verdict.message)

        chunks = await self._retrieve(principal, config, message)
        context = build_context(chunks) if chunks else NO_CONTEXT
        tools = [TOOLS[name].spec for name in config.enabled_tools]
        request = replace(
            request,
            user_message=f"CONTEXT:\n{context}\n\nQUESTION:\n{message}",
            tools=tools,
        )
        if (
            not chunks
            and (config.retrieval.enabled or tools)
            and not may_need_tool(message, config.enabled_tools)
        ):
            return PreparedTurn(config, None, request, session_id, message, canned_answer=DONT_KNOW)
        return PreparedTurn(config, self._provider(config), request, session_id, message, chunks)

    async def _run_tools(self, turn: PreparedTurn) -> tuple[ModelRequest, str | None, list[str]]:
        """Tool rounds via non-streaming calls.

        Returns the request for the final answer, the answer text if the model already
        answered, and the names of tools used.
        """
        assert turn.provider is not None
        request = turn.request
        used: list[str] = []
        for _ in range(MAX_TOOL_ROUNDS):
            reply = await turn.provider.generate(request)
            if not reply.tool_calls:
                return request, reply.text, used
            exchanges = list(request.tool_exchanges)
            for call in reply.tool_calls:
                if call.name in turn.config.enabled_tools:
                    result = run_tool(call.name, call.arguments)
                    used.append(call.name)
                else:
                    result = {"error": f"tool '{call.name[:64]}' is not available"}
                logger.info("tool call %s for %s", call.name[:64], turn.config.assistant_id)
                exchanges.append(ToolExchange(call, result))
            request = replace(request, tool_exchanges=exchanges)
        # Still asking for tools after the last round: answer without tools.
        return replace(request, tools=()), None, used

    async def _save(
        self, principal: Principal, turn: PreparedTurn, final: FinalAnswer
    ) -> SavedTurn:
        return await self._store.save_turn(
            principal,
            turn.config.assistant_id,
            turn.session_id,
            turn.message,
            final.text,
            citations=final.citations,
            grounded=final.grounded,
        )

    def _canned(self, turn: PreparedTurn) -> FinalAnswer:
        assert turn.canned_answer is not None
        return FinalAnswer(turn.canned_answer, False, [], turn.canned_outcome, [])

    async def complete(self, principal: Principal, turn: PreparedTurn) -> ChatResult:
        if turn.canned_answer is not None:
            final = self._canned(turn)
        else:
            assert turn.provider is not None
            request, text, used = await self._run_tools(turn)
            if text is None:
                text = (await turn.provider.generate(request)).text
            final = finalize(turn, text, used)
        saved = await self._save(principal, turn, final)
        logger.info("turn %s outcome=%s", turn.config.assistant_id, final.outcome)
        return ChatResult(
            saved.conversation_id,
            saved.assistant_message_id,
            final.text,
            final.grounded,
            final.citations,
            final.outcome,
            final.tools_used,
        )

    async def stream(self, principal: Principal, turn: PreparedTurn) -> AsyncIterator[SseEvent]:
        parts: list[str] = []
        try:
            if turn.canned_answer is not None:
                final = self._canned(turn)
                yield SseEvent("delta", {"text": final.text})
            else:
                assert turn.provider is not None
                request, text, used = turn.request, None, []
                if turn.request.tools:
                    request, text, used = await self._run_tools(turn)
                if text is not None:
                    parts.append(text)
                    stripped = strip_markers(text)
                    if stripped:
                        yield SseEvent("delta", {"text": stripped})
                else:
                    stripper = MarkerStripper()
                    async for event in turn.provider.stream(request):
                        if isinstance(event, StreamDelta):
                            parts.append(event.text)
                            visible = stripper.feed(event.text)
                            if visible:
                                yield SseEvent("delta", {"text": visible})
                        elif isinstance(event, StreamEnd):
                            break
                    tail = stripper.flush()
                    if tail:
                        yield SseEvent("delta", {"text": tail})
                # Output checks run on the full text; `done.answer` is authoritative.
                final = finalize(turn, "".join(parts), used)
            # Only a fully streamed answer is persisted.
            saved = await self._save(principal, turn, final)
        except ProviderError as exc:
            yield SseEvent("error", {"kind": exc.kind, "message": "model provider error"})
            return
        except Exception:
            yield SseEvent("error", {"kind": "INTERNAL", "message": "internal error"})
            raise
        logger.info("turn %s outcome=%s", turn.config.assistant_id, final.outcome)
        yield SseEvent(
            "done",
            {
                "session_id": str(saved.conversation_id),
                "message_id": str(saved.assistant_message_id),
                "answer": final.text,
                "grounded": final.grounded,
                "citations": final.citations,
                "outcome": final.outcome,
                "tools_used": final.tools_used,
            },
        )
