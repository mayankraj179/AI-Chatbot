from __future__ import annotations

import httpx
from google.genai import errors

from app.providers.gemini import classify_gemini_error

# Body captured from a real call with an invalid key (HTTP 400).
INVALID_KEY_BODY = {
    "error": {
        "code": 400,
        "message": "API key not valid. Please pass a valid API key.",
        "status": "INVALID_ARGUMENT",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
                "domain": "googleapis.com",
            }
        ],
    }
}


def _error(code: int, body: dict) -> errors.APIError:
    return errors.APIError(code, body, httpx.Response(code))


def test_invalid_key_classified() -> None:
    assert classify_gemini_error(_error(400, INVALID_KEY_BODY)).kind == "INVALID_KEY"


def test_status_codes_classified() -> None:
    def body(code: int, status: str, message: str) -> dict:
        return {"error": {"code": code, "status": status, "message": message}}

    cases = {
        (429, "RESOURCE_EXHAUSTED", "You exceeded your current quota"): "QUOTA_EXCEEDED",
        (429, "RESOURCE_EXHAUSTED", "Too many requests"): "RATE_LIMITED",
        (503, "UNAVAILABLE", "The model is overloaded."): "OVERLOADED",
        (403, "PERMISSION_DENIED", "denied"): "INVALID_KEY",
        (400, "INVALID_ARGUMENT", "bad"): "BAD_REQUEST",
        (500, "INTERNAL", "oops"): "PROVIDER_ERROR",
    }
    for (code, status, message), kind in cases.items():
        assert classify_gemini_error(_error(code, body(code, status, message))).kind == kind


def test_gemini_embeddings_require_key() -> None:
    import pytest

    from app.providers.base import ProviderError
    from app.providers.embeddings import GeminiEmbeddingProvider

    with pytest.raises(ProviderError) as info:
        GeminiEmbeddingProvider("gemini-embedding-001", api_key=None)
    assert info.value.kind == "INVALID_KEY"


def test_gemini_embeddings_use_task_types() -> None:
    import asyncio
    from types import SimpleNamespace

    from app.providers.embeddings import GeminiEmbeddingProvider

    calls = []

    async def embed_content(*, model, contents, config):
        calls.append((model, list(contents), config.task_type))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents])

    provider = GeminiEmbeddingProvider("gemini-embedding-001", api_key="test-key")
    provider._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    )
    assert asyncio.run(provider.embed_documents(["a", "b"])) == [[0.1, 0.2], [0.1, 0.2]]
    assert asyncio.run(provider.embed_query("q")) == [0.1, 0.2]
    assert [c[2] for c in calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
