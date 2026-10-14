"""Pattern-based personal-data detection (best effort, no external service)."""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b"),
    "phone number": re.compile(r"(?<![\w.])\+?\d[\d ()-]{8,}\d(?![\w.])"),
    "national id": re.compile(
        r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b|\b\d{3}-\d{2}-\d{4}\b|\b[A-Z]{5}\d{4}[A-Z]\b"
    ),
}
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def find_pii(text: str) -> set[str]:
    found = {kind for kind, pattern in _PATTERNS.items() if pattern.search(text)}
    for match in _CARD.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.add("card number")
            found.discard("phone number")
            found.discard("national id")
    return found


def redact_pii(text: str) -> str:
    for kind, pattern in _PATTERNS.items():
        text = pattern.sub(f"[{kind} removed]", text)
    return text
