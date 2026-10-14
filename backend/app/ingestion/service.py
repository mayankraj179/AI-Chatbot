"""Idempotent, versioned ingestion.

Each file is parsed and fingerprinted by its parsed text, access labels and chunk
settings. Unchanged files (same fingerprint and embedding model) are skipped. A
changed file is embedded first, then swapped in with one transaction: new chunks
become active and the previous version's chunks are flagged retired. Files that
disappear are flagged retired too. Nothing is deleted, and sources are only read.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.assistant_config import AssistantConfig, KnowledgeSourceConfig
from app.ingestion.chunking import Chunk, chunk_document
from app.ingestion.parsers import ParsedDocument, ParseError, is_supported, parse_file
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
    retired: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_written: int = 0


@dataclass(frozen=True)
class _Existing:
    id: uuid.UUID
    fingerprint: str
    embedding_model: str
    version: int
    retired: bool


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


def fingerprint(document: ParsedDocument, labels: list[str], chunk_settings: str) -> str:
    digest = hashlib.sha256()
    for section in document.sections:
        digest.update(repr((section.heading_path, section.page, section.text)).encode())
    digest.update(("\0" + document.title + "\0" + ",".join(sorted(labels))).encode())
    digest.update(("\0" + chunk_settings).encode())
    return digest.hexdigest()


_UPSERT_DOCUMENT = text(
    """
    INSERT INTO documents (tenant_id, assistant_id, source_uri, title, media_type,
        content_hash, embedding_model, access_labels, version)
    VALUES (:tenant_id, :assistant_id, :uri, :title, :media_type, :hash, :model, :labels,
        :version)
    ON CONFLICT (tenant_id, assistant_id, source_uri) DO UPDATE SET
        title = EXCLUDED.title, media_type = EXCLUDED.media_type,
        content_hash = EXCLUDED.content_hash, embedding_model = EXCLUDED.embedding_model,
        access_labels = EXCLUDED.access_labels, version = EXCLUDED.version,
        retired_at = NULL, updated_at = now()
    RETURNING id
    """
)

_INSERT_CHUNK = text(
    """
    INSERT INTO chunks (document_id, tenant_id, assistant_id, version, chunk_index, text,
        heading_path, page, access_labels, embedding_model, embedding)
    VALUES (:document_id, :tenant_id, :assistant_id, :version, :index, :text, :heading_path,
        :page, :labels, :model, CAST(CAST(:embedding AS text) AS vector))
    """
)

_RETIRE_OLD_CHUNKS = text(
    "UPDATE chunks SET retired_at = now() "
    "WHERE document_id = :document_id AND version < :version AND retired_at IS NULL"
)


async def _retire_previous(session: AsyncSession, document_id: uuid.UUID, version: int) -> None:
    await session.execute(_RETIRE_OLD_CHUNKS, {"document_id": document_id, "version": version})


async def _swap_in(
    session_factory: async_sessionmaker[AsyncSession],
    scope: dict[str, str],
    uri: str,
    document: ParsedDocument,
    source: KnowledgeSourceConfig,
    digest: str,
    model_id: str,
    version: int,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> None:
    async with session_factory() as session, session.begin():
        document_id = (
            await session.execute(
                _UPSERT_DOCUMENT,
                {
                    **scope,
                    "uri": uri,
                    "title": document.title,
                    "media_type": document.media_type,
                    "hash": digest,
                    "model": model_id,
                    "labels": source.access_labels,
                    "version": version,
                },
            )
        ).scalar_one()
        await session.execute(
            _INSERT_CHUNK,
            [
                {
                    **scope,
                    "document_id": document_id,
                    "version": version,
                    "index": chunk.index,
                    "text": chunk.text,
                    "heading_path": list(chunk.heading_path),
                    "page": chunk.page,
                    "labels": source.access_labels,
                    "model": model_id,
                    "embedding": vector_literal(vector),
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        await _retire_previous(session, document_id, version)


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
                "SELECT id, source_uri, content_hash, embedding_model, version, "
                "retired_at IS NOT NULL AS retired FROM documents "
                "WHERE tenant_id = :tenant_id AND assistant_id = :assistant_id"
            ),
            scope,
        )
        existing = {
            r.source_uri: _Existing(r.id, r.content_hash, r.embedding_model, r.version, r.retired)
            for r in rows
        }

    retrieval = config.retrieval
    chunk_settings = f"{retrieval.chunk_max_chars}/{retrieval.chunk_overlap_chars}"
    seen: set[str] = set()
    for source in config.knowledge_sources:
        directory = resolve_source_dir(content_root, source)
        for path in _discover(directory):
            uri = f"{source.path.rstrip('/')}/{path.relative_to(directory).as_posix()}"
            seen.add(uri)
            try:
                document = parse_file(path)
            except ParseError as exc:
                # Keep any previously indexed version active rather than dropping it.
                report.failed[uri] = exc.reason
                continue
            digest = fingerprint(document, source.access_labels, chunk_settings)
            previous = existing.get(uri)
            if (
                previous is not None
                and not previous.retired
                and (previous.fingerprint, previous.embedding_model) == (digest, embedder.model_id)
            ):
                report.unchanged.append(uri)
                continue
            chunks = chunk_document(
                document,
                max_chars=retrieval.chunk_max_chars,
                overlap_chars=retrieval.chunk_overlap_chars,
            )
            # Embed before opening the transaction, so a provider failure changes nothing.
            vectors = await embedder.embed_documents(
                [c.embedding_text(document.title) for c in chunks]
            )
            version = previous.version + 1 if previous else 1
            try:
                await _swap_in(
                    session_factory,
                    scope,
                    uri,
                    document,
                    source,
                    digest,
                    embedder.model_id,
                    version,
                    chunks,
                    vectors,
                )
            except Exception as exc:  # noqa: BLE001 - one failed swap must not stop the run
                logger.exception("swap failed for %s", uri)
                report.failed[uri] = f"could not store new version ({type(exc).__name__})"
                continue
            (report.updated if previous else report.added).append(uri)
            report.chunks_written += len(chunks)

    gone = sorted(uri for uri, doc in existing.items() if uri not in seen and not doc.retired)
    if gone:
        async with session_factory() as session, session.begin():
            ids = [existing[uri].id for uri in gone]
            await session.execute(
                text("UPDATE documents SET retired_at = now() WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
            await session.execute(
                text(
                    "UPDATE chunks SET retired_at = now() "
                    "WHERE document_id = ANY(:ids) AND retired_at IS NULL"
                ),
                {"ids": ids},
            )
        report.retired = gone
    logger.info(
        "ingested %s: added=%d updated=%d unchanged=%d retired=%d failed=%d",
        config.assistant_id,
        len(report.added),
        len(report.updated),
        len(report.unchanged),
        len(report.retired),
        len(report.failed),
    )
    return report
