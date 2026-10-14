"""Idempotent ingestion: files under each knowledge source are parsed, chunked,
embedded and stored. Unchanged files are skipped; files that disappeared are
removed from the index (the source itself is never written to)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.assistant_config import AssistantConfig, KnowledgeSourceConfig
from app.ingestion.chunking import chunk_document
from app.ingestion.parsers import ParseError, is_supported, parse_file
from app.providers.base import EmbeddingProvider
from app.services.retrieval import vector_literal

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    assistant_id: str
    embedding_model: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_written: int = 0


def resolve_source_dir(content_root: Path, source: KnowledgeSourceConfig) -> Path:
    root = content_root.resolve()
    directory = (root / source.path).resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"knowledge source path '{source.path}' escapes the content root")
    if not directory.is_dir():
        raise ValueError(f"knowledge source path '{source.path}' is not a directory")
    return directory


def _discover(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and is_supported(p)
        and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )


def _fingerprint(data: bytes, labels: list[str], chunk_settings: str) -> str:
    digest = hashlib.sha256(data)
    digest.update(("\0" + ",".join(sorted(labels)) + "\0" + chunk_settings).encode())
    return digest.hexdigest()


async def ingest_assistant(
    config: AssistantConfig,
    embedder: EmbeddingProvider,
    session_factory: async_sessionmaker[AsyncSession],
    content_root: Path,
) -> IngestReport:
    report = IngestReport(config.assistant_id, embedder.model_id)
    scope = {"tenant_id": config.tenant_id, "assistant_id": config.assistant_id}
    async with session_factory() as session:
        rows = await session.execute(
            text(
                "SELECT source_uri, content_hash, embedding_model FROM documents "
                "WHERE tenant_id = :tenant_id AND assistant_id = :assistant_id"
            ),
            scope,
        )
        existing = {r.source_uri: (r.content_hash, r.embedding_model) for r in rows}

    retrieval = config.retrieval
    chunk_settings = f"{retrieval.chunk_max_chars}/{retrieval.chunk_overlap_chars}"
    seen: set[str] = set()
    for source in config.knowledge_sources:
        directory = resolve_source_dir(content_root, source)
        for path in _discover(directory):
            uri = f"{source.path.rstrip('/')}/{path.relative_to(directory).as_posix()}"
            seen.add(uri)
            fingerprint = _fingerprint(path.read_bytes(), source.access_labels, chunk_settings)
            if existing.get(uri) == (fingerprint, embedder.model_id):
                report.unchanged.append(uri)
                continue
            try:
                document = parse_file(path)
            except ParseError as exc:
                # Keep any previously indexed version rather than dropping it.
                report.failed[uri] = exc.reason
                continue
            chunks = chunk_document(
                document,
                max_chars=retrieval.chunk_max_chars,
                overlap_chars=retrieval.chunk_overlap_chars,
            )
            vectors = await embedder.embed_documents(
                [c.embedding_text(document.title) for c in chunks]
            )
            async with session_factory() as session, session.begin():
                document_id = (
                    await session.execute(
                        text(
                            """
                            INSERT INTO documents (tenant_id, assistant_id, source_uri, title,
                                media_type, content_hash, embedding_model, access_labels)
                            VALUES (:tenant_id, :assistant_id, :uri, :title, :media_type,
                                :hash, :model, :labels)
                            ON CONFLICT (tenant_id, assistant_id, source_uri) DO UPDATE SET
                                title = EXCLUDED.title, media_type = EXCLUDED.media_type,
                                content_hash = EXCLUDED.content_hash,
                                embedding_model = EXCLUDED.embedding_model,
                                access_labels = EXCLUDED.access_labels, updated_at = now()
                            RETURNING id
                            """
                        ),
                        {
                            **scope,
                            "uri": uri,
                            "title": document.title,
                            "media_type": document.media_type,
                            "hash": fingerprint,
                            "model": embedder.model_id,
                            "labels": source.access_labels,
                        },
                    )
                ).scalar_one()
                await session.execute(
                    text("DELETE FROM chunks WHERE document_id = :id"), {"id": document_id}
                )
                if chunks:
                    await session.execute(
                        text(
                            """
                            INSERT INTO chunks (document_id, tenant_id, assistant_id,
                                chunk_index, text, heading_path, page, access_labels,
                                embedding_model, embedding)
                            VALUES (:document_id, :tenant_id, :assistant_id, :index, :text,
                                :heading_path, :page, :labels, :model,
                                CAST(CAST(:embedding AS text) AS vector))
                            """
                        ),
                        [
                            {
                                **scope,
                                "document_id": document_id,
                                "index": chunk.index,
                                "text": chunk.text,
                                "heading_path": list(chunk.heading_path),
                                "page": chunk.page,
                                "labels": source.access_labels,
                                "model": embedder.model_id,
                                "embedding": vector_literal(vector),
                            }
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ],
                    )
            (report.updated if uri in existing else report.added).append(uri)
            report.chunks_written += len(chunks)

    stale = sorted(set(existing) - seen)
    if stale:
        async with session_factory() as session, session.begin():
            await session.execute(
                text(
                    "DELETE FROM documents WHERE tenant_id = :tenant_id "
                    "AND assistant_id = :assistant_id AND source_uri = ANY(:uris)"
                ),
                {**scope, "uris": stale},
            )
        report.removed = stale
    logger.info(
        "ingested %s: added=%d updated=%d unchanged=%d removed=%d failed=%d",
        config.assistant_id,
        len(report.added),
        len(report.updated),
        len(report.unchanged),
        len(report.removed),
        len(report.failed),
    )
    return report
