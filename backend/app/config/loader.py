from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.config.assistant_config import AssistantConfig

# Keys that would hold a credential. In YAML they may only name an environment
# variable via a key ending in "_env" (e.g. api_key_env: GEMINI_API_KEY).
_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|password|passwd|token|credential|private[_-]?key)", re.IGNORECASE
)
# Values that look like real credentials wherever they appear.
_SECRET_VALUE_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API key
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),  # OpenAI/OpenRouter style
    re.compile(r"\bxai-[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
]
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ConfigLoadError(Exception):
    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"assistant config {path}: {reason}")


def _walk(node: Any, location: str = "") -> Iterator[tuple[str, str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{location}.{key}" if location else str(key)
            yield child, str(key), value
            yield from _walk(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{location}[{index}]")


def find_literal_secrets(raw: Any) -> list[str]:
    """Locations (never values) of literal secrets in a parsed YAML document."""
    problems: list[str] = []
    for location, key, value in _walk(raw):
        if isinstance(value, str) and _SECRET_KEY_PATTERN.search(key):
            if key.lower().endswith("_env") and _ENV_NAME.match(value):
                continue
            problems.append(
                f"{location}: literal secret not allowed; reference an environment "
                f"variable with '{key}_env: VAR_NAME' instead"
            )
        elif isinstance(value, str) and any(p.search(value) for p in _SECRET_VALUE_PATTERNS):
            problems.append(f"{location}: value looks like a credential; use an env var")
    return problems


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        where = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"{where}: {error['msg']}")
    return "; ".join(lines)


def load_assistant_config(path: str | Path) -> AssistantConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigLoadError(path, "file does not exist")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigLoadError(path, f"invalid YAML: {exc}") from exc
    if raw is None:
        raise ConfigLoadError(path, "file is empty")
    if not isinstance(raw, dict):
        raise ConfigLoadError(path, "top level must be a mapping")

    secrets = find_literal_secrets(raw)
    if secrets:
        raise ConfigLoadError(path, "; ".join(secrets))

    try:
        return AssistantConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(path, _format_validation_error(exc)) from None


def load_all_assistant_configs(directory: str | Path) -> dict[str, AssistantConfig]:
    """Every *.yaml / *.yml file in ``directory``; fails on the first invalid file."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigLoadError(directory, "config directory does not exist")
    configs: dict[str, AssistantConfig] = {}
    for file_path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        config = load_assistant_config(file_path)
        if config.assistant_id in configs:
            raise ConfigLoadError(file_path, f"duplicate assistant_id '{config.assistant_id}'")
        configs[config.assistant_id] = config
    return configs
