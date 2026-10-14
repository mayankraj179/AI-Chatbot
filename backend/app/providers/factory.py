from __future__ import annotations

from app.config.assistant_config import AssistantConfig
from app.config.settings import Settings
from app.providers.base import EmbeddingProvider, ModelProvider
from app.providers.embeddings import FastEmbedProvider, GeminiEmbeddingProvider
from app.providers.gemini import GeminiModelProvider


def build_model_provider(config: AssistantConfig, settings: Settings) -> ModelProvider:
    if config.model.provider == "gemini":
        return GeminiModelProvider(
            model_name=config.model.model_name, api_key=settings.gemini_api_key
        )
    raise ValueError(f"unsupported model provider '{config.model.provider}'")


def build_embedding_provider(config: AssistantConfig, settings: Settings) -> EmbeddingProvider:
    retrieval = config.retrieval
    if retrieval.embedding_provider == "fastembed":
        return FastEmbedProvider(retrieval.embedding_model, cache_dir=settings.embedding_cache_dir)
    if retrieval.embedding_provider == "gemini":
        return GeminiEmbeddingProvider(retrieval.embedding_model, settings.gemini_api_key)
    raise ValueError(f"unsupported embedding provider '{retrieval.embedding_provider}'")
