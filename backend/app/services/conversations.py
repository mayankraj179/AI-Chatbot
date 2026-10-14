from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.auth.principal import Principal
from app.db.models import Conversation, Message

HISTORY_TURNS = 10
TITLE_MAX_CHARS = 80


class ConversationNotFoundError(Exception):
    """Missing and not-owned conversations are indistinguishable to the caller (404)."""


@dataclass(frozen=True)
class ConversationView:
    id: uuid.UUID
    assistant_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MessageView:
    id: uuid.UUID
    sequence_no: int
    role: str
    content: str
    citations: list[dict[str, Any]]
    grounded: bool
    created_at: datetime


@dataclass(frozen=True)
class SavedTurn:
    conversation_id: uuid.UUID
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID


def parse_conversation_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError, TypeError):
        raise ConversationNotFoundError from None


def _owned(principal: Principal, conversation_id: uuid.UUID, assistant_id: str | None):
    # Ownership is part of the WHERE clause, never checked after loading the row.
    query = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.tenant_id == principal.tenant_id,
        Conversation.principal_id == principal.principal_id,
    )
    if assistant_id is not None:
        query = query.where(Conversation.assistant_id == assistant_id)
    return query


class ConversationStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(
        self, principal: Principal, assistant_id: str, title: str | None = None
    ) -> ConversationView:
        async with self._session_factory() as session, session.begin():
            record = Conversation(
                tenant_id=principal.tenant_id,
                assistant_id=assistant_id,
                principal_id=principal.principal_id,
                title=title,
            )
            session.add(record)
            await session.flush()
            await session.refresh(record)
            return _conversation_view(record)

    async def get(
        self, principal: Principal, conversation_id: uuid.UUID, assistant_id: str | None = None
    ) -> ConversationView:
        async with self._session_factory() as session:
            record = await session.scalar(_owned(principal, conversation_id, assistant_id))
            if record is None:
                raise ConversationNotFoundError
            return _conversation_view(record)

    async def list_messages(
        self, principal: Principal, conversation_id: uuid.UUID
    ) -> list[MessageView]:
        async with self._session_factory() as session:
            if await session.scalar(_owned(principal, conversation_id, None)) is None:
                raise ConversationNotFoundError
            rows = await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_no)
            )
            return [_message_view(row) for row in rows]

    async def recent_history(
        self, principal: Principal, conversation_id: uuid.UUID, assistant_id: str
    ) -> list[MessageView]:
        async with self._session_factory() as session:
            if await session.scalar(_owned(principal, conversation_id, assistant_id)) is None:
                raise ConversationNotFoundError
            rows = await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.sequence_no.desc())
                .limit(HISTORY_TURNS * 2)
            )
            return [_message_view(row) for row in reversed(list(rows))]

    async def save_turn(
        self,
        principal: Principal,
        assistant_id: str,
        conversation_id: uuid.UUID | None,
        user_message: str,
        assistant_message: str,
        *,
        citations: Sequence[dict[str, Any]] = (),
        grounded: bool = False,
    ) -> SavedTurn:
        """Persist one question/answer pair atomically, creating the conversation if needed."""
        async with self._session_factory() as session, session.begin():
            if conversation_id is None:
                conversation = Conversation(
                    tenant_id=principal.tenant_id,
                    assistant_id=assistant_id,
                    principal_id=principal.principal_id,
                    title=_title(user_message),
                )
                session.add(conversation)
                await session.flush()
            else:
                conversation = await session.scalar(
                    _owned(principal, conversation_id, assistant_id).with_for_update()
                )
                if conversation is None:
                    raise ConversationNotFoundError
                if conversation.title is None:
                    conversation.title = _title(user_message)
            next_seq = await session.scalar(
                select(func.coalesce(func.max(Message.sequence_no), -1) + 1).where(
                    Message.conversation_id == conversation.id
                )
            )
            user = Message(
                conversation_id=conversation.id,
                sequence_no=next_seq,
                role="user",
                content=user_message,
            )
            reply = Message(
                conversation_id=conversation.id,
                sequence_no=next_seq + 1,
                role="assistant",
                content=assistant_message,
                citations=list(citations),
                grounded=grounded,
            )
            session.add_all([user, reply])
            conversation.updated_at = func.now()
            await session.flush()
            return SavedTurn(conversation.id, user.id, reply.id)


def _title(message: str) -> str:
    title = " ".join(message.split())
    return title if len(title) <= TITLE_MAX_CHARS else title[: TITLE_MAX_CHARS - 1] + "…"


def _conversation_view(record: Conversation) -> ConversationView:
    return ConversationView(
        id=record.id,
        assistant_id=record.assistant_id,
        title=record.title,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _message_view(record: Message) -> MessageView:
    return MessageView(
        id=record.id,
        sequence_no=record.sequence_no,
        role=record.role,
        content=record.content,
        citations=list(record.citations or []),
        grounded=record.grounded,
        created_at=record.created_at,
    )
