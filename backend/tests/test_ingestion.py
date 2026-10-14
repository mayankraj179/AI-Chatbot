from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config.loader import load_assistant_config
from app.ingestion import service
from app.ingestion.service import ingest_assistant, resolve_source_dir
from tests.conftest import FIXTURES, FakeEmbedder, _truncate


def _run(url: str, content: Path, embedder: FakeEmbedder, config_path: Path):
    async def go():
        engine = create_async_engine(url)
        try:
            sf = async_sessionmaker(engine, expire_on_commit=False)
            report = await ingest_assistant(
                load_assistant_config(config_path), embedder, sf, content
            )
            async with sf() as session:
                rows = (
                    await session.execute(
                        text(
                            "SELECT d.source_uri, count(c.id) AS n, "
                            "min(c.access_labels::text) AS l FROM documents d "
                            "LEFT JOIN chunks c ON c.document_id = d.id "
                            "GROUP BY d.source_uri ORDER BY 1"
                        )
                    )
                ).all()
            return report, [(r.source_uri, r.n, r.l) for r in rows]
        finally:
            await engine.dispose()

    return asyncio.run(go())


@pytest.fixture
def workspace(tmp_path: Path, migrated_db: str) -> Path:
    _truncate(migrated_db)
    shutil.copytree(FIXTURES / "content", tmp_path / "content")
    return tmp_path / "content"


def _active(url: str) -> list[tuple[str, int, int]]:
    async def go():
        engine = create_async_engine(url)
        try:
            async with engine.connect() as conn:
                rows = await conn.execute(
                    text(
                        "SELECT d.source_uri, c.version, count(*) FROM chunks c "
                        "JOIN documents d ON d.id = c.document_id "
                        "WHERE c.retired_at IS NULL AND d.retired_at IS NULL "
                        "GROUP BY 1, 2 ORDER BY 1"
                    )
                )
                return [tuple(r) for r in rows]
        finally:
            await engine.dispose()

    return asyncio.run(go())


def test_ingest_is_idempotent_versions_and_never_deletes(workspace: Path, migrated_db: str) -> None:
    config = FIXTURES / "configs" / "leave_helper.yaml"
    embedder = FakeEmbedder()
    first, rows = _run(migrated_db, workspace, embedder, config)
    assert sorted(first.added) == ["managers/approvals.md", "public/leave.md"]
    assert {uri: labels for uri, _, labels in rows} == {
        "managers/approvals.md": "{role:manager}",
        "public/leave.md": "{tenant:local-development}",
    }
    total_chunks = sum(n for _, n, _ in rows)

    calls = embedder.calls
    # Touching a file without changing its parsed text is not a change.
    leave = workspace / "public" / "leave.md"
    leave.write_text(leave.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    second, rows_again = _run(migrated_db, workspace, embedder, config)
    assert sorted(second.unchanged) == ["managers/approvals.md", "public/leave.md"]
    assert second.added == second.updated == [] and embedder.calls == calls
    assert rows_again == rows

    leave.write_text("# Leave\nNew text only.\n", encoding="utf-8")
    (workspace / "managers" / "approvals.md").unlink()
    (workspace / "public" / "broken.pdf").write_bytes(b"not a pdf")
    (workspace / "public" / ".hidden.md").write_text("# secret draft", encoding="utf-8")
    third, rows = _run(migrated_db, workspace, embedder, config)
    assert third.updated == ["public/leave.md"]
    assert third.retired == ["managers/approvals.md"]
    assert list(third.failed) == ["public/broken.pdf"]
    assert _active(migrated_db) == [("public/leave.md", 2, 1)]
    # Old versions and retired documents are kept, only flagged.
    assert sum(n for _, n, _ in rows) == total_chunks + 1

    (workspace / "managers").mkdir(exist_ok=True)
    (workspace / "managers" / "approvals.md").write_text("# Approvals\nBack again.", "utf-8")
    fourth, _ = _run(migrated_db, workspace, embedder, config)
    assert fourth.updated == ["managers/approvals.md"]
    assert ("managers/approvals.md", 2, 1) in _active(migrated_db)


def test_failed_swap_keeps_previous_version_active(
    workspace: Path, migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FIXTURES / "configs" / "leave_helper.yaml"
    _run(migrated_db, workspace, FakeEmbedder(), config)
    before = _active(migrated_db)

    async def fail_commit(*args, **kwargs):
        raise RuntimeError("simulated failure before commit")

    monkeypatch.setattr(service, "_retire_previous", fail_commit)
    (workspace / "public" / "leave.md").write_text("# Leave\nChanged.\n", encoding="utf-8")
    report, _ = _run(migrated_db, workspace, FakeEmbedder(), config)
    assert report.failed == {"public/leave.md": "could not store new version (RuntimeError)"}
    assert _active(migrated_db) == before
    monkeypatch.undo()
    retry, _ = _run(migrated_db, workspace, FakeEmbedder(), config)
    assert retry.updated == ["public/leave.md"]


def test_source_path_cannot_escape_content_root(tmp_path: Path) -> None:
    config = load_assistant_config(FIXTURES / "configs" / "leave_helper.yaml")
    source = config.knowledge_sources[0].model_copy(update={"path": "../outside"})
    (tmp_path / "root").mkdir()
    (tmp_path / "outside").mkdir()
    with pytest.raises(ValueError, match="escapes the content root"):
        resolve_source_dir(tmp_path / "root", source)
