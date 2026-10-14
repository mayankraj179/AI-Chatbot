"""Output guardrails: citation-marker stripping (also across stream chunks),
refusal detection and output PII checks, run on the full answer text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config.assistant_config import GuardrailsConfig
from app.guardrails.pii import find_pii
from app.guardrails.text import normalise

# 【4:0†source】, [1], [1, 2], [source: x], (Source 3) style markers.
_MARKERS = re.compile(
    r"\s?【[^】]{0,80}】|\s?\[(?:\d+(?:\s*[,;-]\s*\d+)*|source[^\]]{0,60}|doc[^\]]{0,60})\]"
    r"|\s?\((?:source|sources|doc) ?\d+(?:\s*,\s*\d+)*\)",
    re.IGNORECASE,
)
_OPENERS = ("【", "[", "(")
_HOLD_CHARS = 90

_REFUSAL_PATTERNS = [
    r"\bi (do not|don't|dont) know\b",
    r"\bi (could not|couldn't|cannot|can't|was unable to|am unable to) (find|locate|answer)\b",
    r"\b(not|isn't|is not) (mentioned|covered|included|specified|stated|available|provided) in "
    r"the (provided |available )?(context|documents?|sources?)\b",
    r"\b(the )?(provided |available )?(context|documents?) (does not|doesn't|do not|don't) "
    r"(contain|mention|include|cover|say|provide|specify)\b",
    r"\bno (relevant )?information (about|on|regarding)\b",
    r"\bi('m| am) (sorry|afraid),? (but )?i (can't|cannot)\b",
    r"\boutside (of )?(the )?(scope|documents)\b",
]
_REFUSAL = [re.compile(p) for p in _REFUSAL_PATTERNS]


def strip_markers(text: str) -> str:
    return _MARKERS.sub("", text)


class MarkerStripper:
    """Streaming version of strip_markers: holds back a possible unfinished marker."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> str:
        text = strip_markers(self._pending + chunk)
        cut = max(text.rfind(opener) for opener in _OPENERS)
        if cut != -1 and len(text) - cut <= _HOLD_CHARS and not _closed(text[cut:]):
            # Keep a leading space with the held text so stripping it later leaves no gap.
            if cut > 0 and text[cut - 1] == " ":
                cut -= 1
            self._pending = text[cut:]
            return text[:cut]
        # Trailing whitespace waits too, in case the next chunk starts with a marker.
        body = text.rstrip()
        self._pending = text[len(body) :]
        return body

    def flush(self) -> str:
        text, self._pending = strip_markers(self._pending), ""
        return text


def _closed(fragment: str) -> bool:
    closer = {"【": "】", "[": "]", "(": ")"}[fragment[0]]
    return closer in fragment


def is_refusal(text: str) -> bool:
    lowered = normalise(text)
    return any(pattern.search(lowered) for pattern in _REFUSAL)


@dataclass(frozen=True)
class OutputVerdict:
    reason: str
    message: str


def screen_output(text: str, config: GuardrailsConfig) -> OutputVerdict | None:
    if config.output_pii_block:
        kinds = find_pii(text)
        if kinds:
            return OutputVerdict(
                "output_pii",
                "I can't share that answer because it contains personal data ("
                + ", ".join(sorted(kinds))
                + ").",
            )
    return None
