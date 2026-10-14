from __future__ import annotations

from pathlib import Path

import pytest

from app.config.loader import ConfigLoadError, load_all_assistant_configs, load_assistant_config

BASE = """
assistant_id: leave_helper
display_name: Leave Helper
tenant_id: local-development
model: {provider: gemini, model_name: some-model}
system_prompt: Answer from documents only.
"""


def write(tmp_path: Path, text: str, name: str = "a.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_minimal_config_loads_with_defaults(tmp_path: Path) -> None:
    config = load_assistant_config(write(tmp_path, BASE))
    assert config.assistant_id == "leave_helper"
    assert config.retrieval.enabled is False
    assert config.guardrails.max_input_chars == 4000
    assert config.chart_trigger_patterns


def test_shipped_configs_are_valid() -> None:
    configs = load_all_assistant_configs(Path(__file__).resolve().parents[1] / "configs")
    assert "leave_helper" in configs


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ("unknown_field: 1\n", "unknown_field: Extra inputs are not permitted"),
        ("enabled_tools: [send_email]\n", "unknown tool(s) ['send_email']"),
        ("retrieval: {enabled: true}\n", "min_similarity is required"),
        ("chart_trigger_patterns: ['(']\n", "invalid regex"),
        (
            "knowledge_sources: [{type: filesystem, path: x, access_labels: [employee]}]\n",
            "must be namespaced",
        ),
        ("knowledge_sources: [{type: filesystem, path: x, access_labels: []}]\n", "at least 1"),
    ],
)
def test_invalid_configs_give_clear_errors(tmp_path: Path, extra: str, expected: str) -> None:
    with pytest.raises(ConfigLoadError) as info:
        load_assistant_config(write(tmp_path, BASE + extra))
    assert expected in info.value.reason
    assert "a.yaml" in str(info.value)


def test_unknown_provider_rejected(tmp_path: Path) -> None:
    text = BASE.replace("provider: gemini", "provider: acme")
    with pytest.raises(ConfigLoadError, match="unknown model provider 'acme'"):
        load_assistant_config(write(tmp_path, text))


@pytest.mark.parametrize(
    "extra",
    [
        "model_api_key: not-a-real-value-123\n",
        "knowledge_sources: [{type: filesystem, path: x, access_labels: [r:a], password: p}]\n",
        "description: AIzaSyD-fake-but-shaped-like-a-key-0000000\n",
    ],
)
def test_literal_secrets_rejected_without_echoing_value(tmp_path: Path, extra: str) -> None:
    with pytest.raises(ConfigLoadError) as info:
        load_assistant_config(write(tmp_path, BASE + extra))
    message = str(info.value)
    assert "secret" in message or "credential" in message
    for value in ("not-a-real-value-123", "AIzaSyD-fake"):
        assert value not in message


def test_env_reference_is_allowed_by_secret_scan(tmp_path: Path) -> None:
    # The secret scan passes; the schema then rejects the unknown field by name.
    with pytest.raises(ConfigLoadError, match="api_key_env: Extra inputs"):
        load_assistant_config(write(tmp_path, BASE + "api_key_env: GEMINI_API_KEY\n"))


def test_duplicate_assistant_ids_rejected(tmp_path: Path) -> None:
    write(tmp_path, BASE, "a.yaml")
    write(tmp_path, BASE, "b.yaml")
    with pytest.raises(ConfigLoadError, match="duplicate assistant_id"):
        load_all_assistant_configs(tmp_path)


@pytest.mark.parametrize("text", ["", "[1, 2]", "a: [unclosed"])
def test_empty_or_malformed_yaml(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigLoadError):
        load_assistant_config(write(tmp_path, text))
