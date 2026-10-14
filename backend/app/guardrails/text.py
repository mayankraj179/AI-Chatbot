"""Shared text normalisation for guardrail matching."""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text).lower()
    return re.sub(r"\s+", " ", text).strip()
