from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config.loader import load_assistant_config
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


def test_ingest_is_idempotent_and_tracks_changes(workspace: Path, migrated_db: str) -> None:
    config = FIXTURES / "configs" / "leave_helper.yaml"
    embedder = FakeEmbedder()
    first, rows = _run(migrated_db, workspace, embedder, config)
    assert sorted(first.added) == ["managers/approvals.md", "public/leave.md"]
    assert {uri: labels for uri, _, labels in rows} == {
        "managers/approvals.md": "{role:manager}",
        "public/leave.md": "{tenant:local-development}",
    }

    calls = embedder.calls
    second, rows_again = _run(migrated_db, workspace, embedder, config)
    assert sorted(second.unchanged) == ["managers/approvals.md", "public/leave.md"]
    assert second.added == second.updated == [] and embedder.calls == calls
    assert rows_again == rows

    (workspace / "public" / "leave.md").write_text("# Leave\nNew text only.\n", encoding="utf-8")
    (workspace / "managers" / "approvals.md").unlink()
    (workspace / "public" / "broken.pdf").write_bytes(b"not a pdf")
    (workspace / "public" / ".hidden.md").write_text("# secret draft", encoding="utf-8")
    third, rows = _run(migrated_db, workspace, embedder, config)
    assert third.updated == ["public/leave.md"]
    assert third.removed == ["managers/approvals.md"]
    assert list(third.failed) == ["public/broken.pdf"]
    assert rows == [("public/leave.md", 1, "{tenant:local-development}")]


def test_source_path_cannot_escape_content_root(tmp_path: Path) -> None:
    config = load_assistant_config(FIXTURES / "configs" / "leave_helper.yaml")
    source = config.knowledge_sources[0].model_copy(update={"path": "../outside"})
    (tmp_path / "root").mkdir()
    (tmp_path / "outside").mkdir()
    with pytest.raises(ValueError, match="escapes the content root"):
        resolve_source_dir(tmp_path / "root", source)
