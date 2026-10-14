from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth.dependencies import require_permission
from app.auth.principal import (
    PERMISSION_CHAT,
    PERMISSION_CONVERSATION_READ,
    PERMISSION_CONVERSATION_WRITE,
    Principal,
)
from app.chat.service import (
    AssistantNotFoundError,
    ChatService,
    InputRejectedError,
    PreparedTurn,
    SseEvent,
)
from app.providers.base import ProviderError
from app.services.conversations import (
    ConversationNotFoundError,
    ConversationStore,
    parse_conversation_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_store(request: Request) -> ConversationStore:
    return request.app.state.conversation_store


class AssistantOut(BaseModel):
    assistant_id: str
    display_name: str
    description: str


class SessionCreate(BaseModel):
    assistant_id: str


class SessionOut(BaseModel):
    session_id: uuid.UUID
    assistant_id: str
    title: str | None
    created_at: datetime


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    grounded: bool
    created_at: datetime


class ChatRequest(BaseModel):
    assistant_id: str
    message: str = Field(max_length=100_000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    grounded: bool
    citations: list[dict[str, Any]]


def _not_found(what: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail=f"{what} not found")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/assistants", response_model=list[AssistantOut])
async def list_assistants(
    principal: Principal = Depends(require_permission(PERMISSION_CHAT)),
    chat: ChatService = Depends(get_chat_service),
) -> list[AssistantOut]:
    return [
        AssistantOut(
            assistant_id=a.assistant_id, display_name=a.display_name, description=a.description
        )
        for a in chat.assistants_for(principal)
    ]


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    principal: Principal = Depends(require_permission(PERMISSION_CONVERSATION_WRITE)),
    chat: ChatService = Depends(get_chat_service),
    store: ConversationStore = Depends(get_store),
) -> SessionOut:
    try:
        chat.assistant(principal, body.assistant_id)
    except AssistantNotFoundError:
        raise _not_found("assistant") from None
    view = await store.create(principal, body.assistant_id)
    return SessionOut(
        session_id=view.id,
        assistant_id=view.assistant_id,
        title=view.title,
        created_at=view.created_at,
    )


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
async def list_session_messages(
    session_id: str,
    principal: Principal = Depends(require_permission(PERMISSION_CONVERSATION_READ)),
    store: ConversationStore = Depends(get_store),
) -> list[MessageOut]:
    try:
        messages = await store.list_messages(principal, parse_conversation_id(session_id))
    except ConversationNotFoundError:
        raise _not_found("session") from None
    return [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            citations=m.citations,
            grounded=m.grounded,
            created_at=m.created_at,
        )
        for m in messages
    ]


async def _prepare(chat: ChatService, principal: Principal, body: ChatRequest) -> PreparedTurn:
    try:
        session_id = parse_conversation_id(body.session_id) if body.session_id else None
        return await chat.prepare(principal, body.assistant_id, body.message, session_id)
    except AssistantNotFoundError:
        raise _not_found("assistant") from None
    except ConversationNotFoundError:
        raise _not_found("session") from None
    except InputRejectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=exc.detail) from None
    except ProviderError as exc:
        logger.error("provider setup failed: %s %s", exc.provider, exc.kind)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="model provider error") from None


@router.post("/chat", response_model=ChatResponse)
async def chat_once(
    body: ChatRequest,
    principal: Principal = Depends(require_permission(PERMISSION_CHAT)),
    chat: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    turn = await _prepare(chat, principal, body)
    try:
        result = await chat.complete(principal, turn)
    except ProviderError as exc:
        logger.error("provider call failed: %s %s %s", exc.provider, exc.kind, exc.message)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={"kind": exc.kind, "message": "model provider error"},
        ) from None
    except ConversationNotFoundError:
        raise _not_found("session") from None
    return ChatResponse(
        session_id=result.session_id,
        message_id=result.message_id,
        answer=result.answer,
        grounded=result.grounded,
        citations=result.citations,
    )


def _sse(event: SseEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    principal: Principal = Depends(require_permission(PERMISSION_CHAT)),
    chat: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    turn = await _prepare(chat, principal, body)

    async def events() -> AsyncIterator[str]:
        async for event in chat.stream(principal, turn):
            if event.event == "error":
                logger.error("stream failed: %s", event.data.get("kind"))
            yield _sse(event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class SearchRequest(BaseModel):
    assistant_id: str
    query: str


class SearchResponse(BaseModel):
    citations: list[dict[str, Any]]


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    principal: Principal = Depends(require_permission(PERMISSION_CHAT)),
    chat: ChatService = Depends(get_chat_service),
) -> SearchResponse:
    """Permission-filtered retrieval only (no model call); used for checks and evaluation."""
    try:
        chunks = await chat.search(principal, body.assistant_id, body.query)
    except AssistantNotFoundError:
        raise _not_found("assistant") from None
    except InputRejectedError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=exc.detail) from None
    except ProviderError as exc:
        logger.error("embedding failed: %s %s", exc.provider, exc.kind)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, detail="embedding provider error"
        ) from None
    return SearchResponse(citations=[c.citation(n) for n, c in enumerate(chunks, start=1)])
