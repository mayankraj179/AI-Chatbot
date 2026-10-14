"""Embedding providers. Documents and queries are embedded differently where the
model supports it (Gemini task types, BGE query instruction)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.providers.base import ProviderError
from app.providers.gemini import classify_gemini_error

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class FastEmbedProvider:
    """Local ONNX model via fastembed; no API key and no quota."""

    name = "fastembed"

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.model_id = f"fastembed:{model_name}"
        self._cache_dir = cache_dir
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _load(self) -> Any:
        async with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = await asyncio.to_thread(
                    TextEmbedding, self.model_name, cache_dir=self._cache_dir
                )
        return self._model

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = await self._load()
        vectors = await asyncio.to_thread(lambda: list(model.embed(list(texts))))
        return [v.tolist() for v in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts) if texts else []

    async def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return (await self._embed([prefix + text]))[0]


class GeminiEmbeddingProvider:
    name = "gemini"
    _BATCH = 100

    def __init__(self, model_name: str, api_key: str | None) -> None:
        if not api_key:
            raise ProviderError("INVALID_KEY", "GEMINI_API_KEY is not set", provider="gemini")
        self.model_name = model_name
        self.model_id = f"gemini:{model_name}"
        self._client = genai.Client(api_key=api_key)

    async def _embed(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH):
            batch = list(texts[start : start + self._BATCH])
            try:
                response = await self._client.aio.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=genai_types.EmbedContentConfig(task_type=task_type),
                )
            except genai_errors.APIError as exc:
                raise classify_gemini_error(exc) from exc
            vectors.extend(list(e.values or []) for e in response.embeddings or [])
        if len(vectors) != len(texts):
            raise ProviderError("PROVIDER_ERROR", "embedding count mismatch", provider="gemini")
        return vectors

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]
