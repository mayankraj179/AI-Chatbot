from __future__ import annotations

from app.config.assistant_config import AssistantConfig
from app.config.settings import Settings
from app.providers.base import ModelProvider
from app.providers.gemini import GeminiModelProvider


def build_model_provider(config: AssistantConfig, settings: Settings) -> ModelProvider:
    if config.model.provider == "gemini":
        return GeminiModelProvider(
            model_name=config.model.model_name, api_key=settings.gemini_api_key
        )
    raise ValueError(f"unsupported model provider '{config.model.provider}'")
