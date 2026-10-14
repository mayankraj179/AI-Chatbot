from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Provider and tool names the platform implements. A config naming anything else is
# rejected at load time. Tools are implemented in Phase 3; their names are reserved now.
KNOWN_MODEL_PROVIDERS = frozenset({"gemini"})
KNOWN_EMBEDDING_PROVIDERS = frozenset({"gemini"})
KNOWN_TOOLS = frozenset({"current_datetime", "calculate"})
KNOWN_SOURCE_TYPES = frozenset({"filesystem"})

DEFAULT_CHART_TRIGGER_PATTERNS: list[str] = [
    r"\bchart\b",
    r"\bgraph\b",
    r"\bplot\b",
    r"visuali[sz]e",
    r"\btrend\b",
    r"\bbreakdown\b",
]

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_LABEL_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*:[^\s:][^\s]*$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ModelConfig(_Strict):
    provider: str
    model_name: str = Field(min_length=1)

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        if v not in KNOWN_MODEL_PROVIDERS:
            raise ValueError(
                f"unknown model provider '{v}'; known: {sorted(KNOWN_MODEL_PROVIDERS)}"
            )
        return v


class RetrievalConfig(_Strict):
    enabled: bool = True
    top_k: int = Field(default=8, ge=1, le=50)
    # Measured per assistant and embedding model (Phase 2); required when enabled.
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    embedding_provider: str = "gemini"

    @field_validator("embedding_provider")
    @classmethod
    def _known_embedding_provider(cls, v: str) -> str:
        if v not in KNOWN_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"unknown embedding provider '{v}'; known: {sorted(KNOWN_EMBEDDING_PROVIDERS)}"
            )
        return v

    @model_validator(mode="after")
    def _threshold_when_enabled(self) -> RetrievalConfig:
        if self.enabled and self.min_similarity is None:
            raise ValueError("retrieval.min_similarity is required when retrieval is enabled")
        return self


class GuardrailsConfig(_Strict):
    prompt_injection_screening: bool = True
    input_pii_block: bool = False
    output_pii_block: bool = False
    max_input_chars: int = Field(default=4000, ge=1, le=100_000)


class KnowledgeSourceConfig(_Strict):
    type: str
    path: str = Field(min_length=1)
    access_labels: list[str] = Field(min_length=1)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in KNOWN_SOURCE_TYPES:
            raise ValueError(
                f"unknown knowledge source type '{v}'; known: {sorted(KNOWN_SOURCE_TYPES)}"
            )
        return v

    @field_validator("access_labels")
    @classmethod
    def _namespaced(cls, v: list[str]) -> list[str]:
        for label in v:
            if not _LABEL_PATTERN.match(label):
                raise ValueError(f"access label '{label}' must be namespaced, e.g. 'role:employee'")
        return v


class AssistantConfig(_Strict):
    assistant_id: str
    display_name: str = Field(min_length=1, max_length=100)
    description: str = ""
    tenant_id: str = Field(min_length=1)
    model: ModelConfig
    retrieval: RetrievalConfig = Field(default_factory=lambda: RetrievalConfig(enabled=False))
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    enabled_tools: list[str] = Field(default_factory=list)
    chart_trigger_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_CHART_TRIGGER_PATTERNS)
    )
    system_prompt: str = Field(min_length=1)
    knowledge_sources: list[KnowledgeSourceConfig] = Field(default_factory=list)

    @field_validator("assistant_id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"assistant_id '{v}' must be lowercase letters, digits, '_' or '-' (max 64)"
            )
        return v

    @field_validator("enabled_tools")
    @classmethod
    def _known_tools(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("enabled_tools contains duplicates")
        unknown = sorted(set(v) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tool(s) {unknown}; known: {sorted(KNOWN_TOOLS)}")
        return v

    @field_validator("chart_trigger_patterns")
    @classmethod
    def _valid_regex(cls, v: list[str]) -> list[str]:
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"chart_trigger_patterns: invalid regex {pattern!r}: {exc}"
                ) from exc
        return v
