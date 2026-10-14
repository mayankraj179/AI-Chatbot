from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.auth.jwt_validator import JwtValidator
from app.chat.service import ChatService, ProviderFactory
from app.config.loader import load_all_assistant_configs
from app.config.settings import Settings, get_settings
from app.db.session import create_engine, create_session_factory
from app.providers.factory import build_model_provider
from app.services.conversations import ConversationStore


def create_app(
    settings: Settings | None = None,
    *,
    jwt_validator: JwtValidator | None = None,
    provider_factory: ProviderFactory | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    # Fail at startup, with the file name and field, if any assistant config is invalid.
    assistants = load_all_assistant_configs(settings.assistant_configs_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings.database_url)
        store = ConversationStore(create_session_factory(engine))
        app.state.conversation_store = store
        app.state.jwt_validator = jwt_validator or JwtValidator(settings)
        app.state.chat_service = ChatService(
            assistants,
            store,
            provider_factory or (lambda config: build_model_provider(config, settings)),
        )
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(title="AI-Chatbot", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


def __getattr__(name: str) -> FastAPI:
    # `uvicorn app.main:app` builds the app on first access, so importing this module
    # (e.g. in tests) has no side effects.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
