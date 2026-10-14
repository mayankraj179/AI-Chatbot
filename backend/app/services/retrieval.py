"""Permission-filtered vector search. Access control is part of the SQL WHERE clause."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

SNIPPET_CHARS = 300

# The query vector is passed as text and cast in SQL, so no driver codec is needed.
SEARCH_SQL = text(
    """
    WITH q AS (SELECT CAST(CAST(:embedding AS text) AS vector) AS v)
    SELECT c.id, c.document_id, d.title, d.source_uri, c.heading_path, c.page, c.text,
           1 - (c.embedding <=> q.v) AS score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN q
    WHERE c.tenant_id = :tenant_id
      AND c.assistant_id = :assistant_id
      AND c.embedding_model = :embedding_model
      AND c.access_labels && CAST(:labels AS text[])
      AND 1 - (c.embedding <=> q.v) >= :min_similarity
    ORDER BY c.embedding <=> q.v
    LIMIT :top_k
    """
)


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    source_uri: str
    heading_path: tuple[str, ...]
    page: int | None
    text: str
    score: float

    def citation(self, number: int) -> dict[str, Any]:
        return {
            "number": number,
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "title": self.title,
            "source_uri": self.source_uri,
            "heading_path": list(self.heading_path),
            "page": self.page,
            "score": round(self.score, 4),
            "snippet": self.text[:SNIPPET_CHARS],
        }


class Retriever:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search(
        self,
        *,
        tenant_id: str,
        assistant_id: str,
        labels: Sequence[str],
        embedding_model: str,
        query_vector: Sequence[float],
        top_k: int,
        min_similarity: float,
    ) -> list[RetrievedChunk]:
        if not labels:
            return []
        params = {
            "embedding": vector_literal(query_vector),
            "tenant_id": tenant_id,
            "assistant_id": assistant_id,
            "embedding_model": embedding_model,
            "labels": sorted(labels),
            "min_similarity": min_similarity,
            "top_k": top_k,
        }
        async with self._session_factory() as session:
            rows = (await session.execute(SEARCH_SQL, params)).all()
        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                title=row.title,
                source_uri=row.source_uri,
                heading_path=tuple(row.heading_path),
                page=row.page,
                text=row.text,
                score=float(row.score),
            )
            for row in rows
        ]
