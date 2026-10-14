"""Input guardrails, run before retrieval or any model call."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config.assistant_config import GuardrailsConfig
from app.guardrails.pii import find_pii
from app.guardrails.text import normalise

_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|"
    r"your|the|system)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|directions?)\b",
    r"\b(reveal|show|print|repeat|leak|output|tell me)\b.{0,40}\b(system|hidden|initial|"
    r"developer)\b.{0,15}\b(prompt|instructions?|message)\b",
    r"\byou are (now|no longer)\b",
    r"\b(act|behave|respond) as (if you were |an? )?(unrestricted|unfiltered|jailbroken|dan)\b",
    r"\b(developer|god|dan|jailbreak) mode\b",
    r"\bjailbreak\b",
    r"\bpretend (that )?(you|there) (are|is|have) no (rules|restrictions|guidelines|limits)\b",
    r"</?\s*(system|assistant|instructions?)\s*>",
    r"\[/?(system|inst)\]",
    r"\bnew (system )?instructions?\s*:",
    r"\banswer from (your )?(general|own|world) knowledge\b",
]
_INJECTION = [re.compile(p) for p in _INJECTION_PATTERNS]


@dataclass(frozen=True)
class InputVerdict:
    reason: str
    message: str


def screen_input(message: str, config: GuardrailsConfig) -> InputVerdict | None:
    """Returns why the message is refused, or None when it may proceed."""
    if config.prompt_injection_screening:
        text = normalise(message)
        if any(pattern.search(text) for pattern in _INJECTION):
            return InputVerdict(
                "prompt_injection",
                "This message looks like an attempt to change my instructions, so I can't "
                "process it. Please ask a question about the documents.",
            )
    if config.input_pii_block:
        kinds = find_pii(message)
        if kinds:
            return InputVerdict(
                "input_pii",
                "Please remove personal data ("
                + ", ".join(sorted(kinds))
                + ") from your message and ask again.",
            )
    return None
