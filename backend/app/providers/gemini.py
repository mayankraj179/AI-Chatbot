from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors, types

from app.providers.base import (
    ModelReply,
    ModelRequest,
    ProviderError,
    StreamDelta,
    StreamEnd,
    StreamEvent,
    TokenUsage,
)

PROVIDER = "gemini"


def classify_gemini_error(exc: errors.APIError) -> ProviderError:
    """Map a google-genai APIError to one ProviderError kind.

    Shapes taken from real responses, e.g. an invalid key returns HTTP 400,
    status INVALID_ARGUMENT, details[].reason API_KEY_INVALID.
    """
    code = exc.code or 0
    status = (exc.status or "").upper()
    message = (exc.message or "").strip() or status or "unknown error"
    reasons = _error_reasons(exc.details)
    if "API_KEY_INVALID" in reasons or code in (401, 403) or status == "PERMISSION_DENIED":
        kind = "INVALID_KEY"
    elif code == 429 or status == "RESOURCE_EXHAUSTED":
        kind = "QUOTA_EXCEEDED" if "quota" in message.lower() else "RATE_LIMITED"
    elif code in (503, 504) or status in ("UNAVAILABLE", "DEADLINE_EXCEEDED"):
        kind = "OVERLOADED"
    elif 400 <= code < 500:
        kind = "BAD_REQUEST"
    else:
        kind = "PROVIDER_ERROR"
    return ProviderError(kind, f"HTTP {code} {status}: {message}"[:500], provider=PROVIDER)


def _error_reasons(details: Any) -> set[str]:
    reasons: set[str] = set()
    error = details.get("error") if isinstance(details, dict) else None
    for item in (error or {}).get("details", []) if isinstance(error, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("reason"), str):
            reasons.add(item["reason"])
    return reasons


def _usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> TokenUsage:
    if metadata is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
        reasoning_tokens=metadata.thoughts_token_count or 0,
    )


class GeminiModelProvider:
    name = PROVIDER

    def __init__(self, *, model_name: str, api_key: str | None, client: Any = None) -> None:
        if client is None and not api_key:
            raise ProviderError("INVALID_KEY", "GEMINI_API_KEY is not set", provider=PROVIDER)
        self.model_name = model_name
        self._client = client or genai.Client(api_key=api_key)

    def _contents(self, request: ModelRequest) -> list[types.Content]:
        contents = [
            types.Content(
                role="model" if turn.role == "assistant" else "user",
                parts=[types.Part(text=turn.content)],
            )
            for turn in request.history
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=request.user_message)]))
        return contents

    def _config(self, request: ModelRequest) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def generate(self, request: ModelRequest) -> ModelReply:
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=self._contents(request),
                config=self._config(request),
            )
        except errors.APIError as exc:
            raise classify_gemini_error(exc) from None
        return ModelReply(text=response.text or "", usage=_usage(response.usage_metadata))

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        usage = TokenUsage()
        try:
            chunks = await self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=self._contents(request),
                config=self._config(request),
            )
            async for chunk in chunks:
                if chunk.usage_metadata is not None:
                    usage = _usage(chunk.usage_metadata)
                if chunk.text:
                    yield StreamDelta(chunk.text)
        except errors.APIError as exc:
            raise classify_gemini_error(exc) from None
        yield StreamEnd(usage)
