"""Measure retrieval scores to choose min_similarity for one assistant and embedding model.

``python -m app.ingestion.calibrate --queries ../eval/calibration/leave_helper.yaml``

For every query the best chunk score is computed over all chunks of the assistant
(as if the user held every access label of its sources). The threshold should sit
between the highest unrelated score and the lowest relevant score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from app.config.loader import load_all_assistant_configs
from app.config.settings import get_settings
from app.db.session import create_engine, create_session_factory
from app.providers.factory import build_embedding_provider
from app.services.retrieval import Retriever


async def top_scores(queries: list[str], config, embedder, retriever: Retriever) -> list[float]:
    labels = sorted({label for s in config.knowledge_sources for label in s.access_labels})
    scores = []
    for query in queries:
        hits = await retriever.search(
            tenant_id=config.tenant_id,
            assistant_id=config.assistant_id,
            labels=labels,
            embedding_model=embedder.model_id,
            query_vector=await embedder.embed_query(query),
            top_k=1,
            min_similarity=-1.0,
        )
        scores.append(hits[0].score if hits else float("nan"))
    return scores


async def main(queries_file: Path) -> int:
    spec = yaml.safe_load(queries_file.read_text(encoding="utf-8"))
    settings = get_settings()
    config = load_all_assistant_configs(settings.assistant_configs_dir)[spec["assistant_id"]]
    embedder = build_embedding_provider(config, settings)
    engine = create_engine(settings.database_url)
    try:
        retriever = Retriever(create_session_factory(engine))
        relevant = await top_scores(spec["relevant"], config, embedder, retriever)
        unrelated = await top_scores(spec["unrelated"], config, embedder, retriever)
    finally:
        await engine.dispose()
    for label, queries, scores in (
        ("relevant", spec["relevant"], relevant),
        ("unrelated", spec["unrelated"], unrelated),
    ):
        for query, score in zip(queries, scores, strict=True):
            print(f"{label:9} {score:.4f}  {query}")
    low_relevant, high_unrelated = min(relevant), max(unrelated)
    summary = {
        "assistant_id": config.assistant_id,
        "embedding_model": embedder.model_id,
        "relevant_range": [round(low_relevant, 4), round(max(relevant), 4)],
        "unrelated_range": [round(min(unrelated), 4), round(high_unrelated, 4)],
        "separable": low_relevant > high_unrelated,
        "suggested_min_similarity": round((low_relevant + high_unrelated) / 2, 3),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["separable"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    sys.exit(asyncio.run(main(parser.parse_args().queries)))
