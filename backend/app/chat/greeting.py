"""Greeting handler: a whole-message plain greeting gets a fixed reply, with no
retrieval or provider call."""

from __future__ import annotations

import re

from app.guardrails.text import normalise

_GREETINGS = {
    "hi",
    "hii",
    "hiii",
    "hello",
    "helo",
    "hey",
    "heya",
    "hiya",
    "howdy",
    "namaste",
    "namaskar",
    "greetings",
    "yo",
    "hola",
}
_TIMED = {"morning", "afternoon", "evening", "day"}
_FILLER = {
    "there",
    "all",
    "everyone",
    "team",
    "folks",
    "friend",
    "buddy",
    "bot",
    "assistant",
    "again",
    "ji",
    "sir",
    "madam",
    "good",
    "and",
    "a",
    "very",
    "to",
    "you",
}


def is_greeting(message: str, display_name: str = "") -> bool:
    words = re.findall(r"[a-z]+", normalise(message))
    if not words or len(words) > 8:
        return False
    if re.search(r"[?]|\d", message):
        return False
    allowed = _GREETINGS | _TIMED | _FILLER | set(re.findall(r"[a-z]+", display_name.lower()))
    if any(word not in allowed for word in words):
        return False
    has_timed = any(a == "good" and b in _TIMED for a, b in zip(words, words[1:], strict=False))
    return has_timed or any(word in _GREETINGS for word in words)


def greeting_reply(display_name: str) -> str:
    return (
        f"Hello! I'm {display_name}. Ask me a question and I'll answer from the documents "
        "you have access to."
    )
