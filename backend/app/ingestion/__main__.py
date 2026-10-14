"""CLI: ``python -m app.ingestion [--assistant ID]`` (run from backend/)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.config.loader import load_all_assistant_configs
from app.config.settings import get_settings
from app.db.session import create_engine, create_session_factory
from app.ingestion.service import ingest_assistant
from app.providers.factory import build_embedding_provider


async def main(assistant_id: str | None) -> int:
    settings = get_settings()
    configs = load_all_assistant_configs(settings.assistant_configs_dir)
    if assistant_id and assistant_id not in configs:
        print(f"unknown assistant '{assistant_id}'", file=sys.stderr)
        return 2
    targets = [configs[assistant_id]] if assistant_id else list(configs.values())
    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        failed = False
        for config in targets:
            if not config.retrieval.enabled:
                continue
            embedder = build_embedding_provider(config, settings)
            report = await ingest_assistant(
                config, embedder, session_factory, Path(settings.content_root)
            )
            print(json.dumps(asdict(report), indent=2))
            failed = failed or bool(report.failed)
        return 1 if failed else 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assistant", help="only ingest this assistant")
    sys.exit(asyncio.run(main(parser.parse_args().assistant)))
