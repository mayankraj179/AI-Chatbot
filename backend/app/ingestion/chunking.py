"""Heading-aware chunking: chunks never cross a section boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.parsers import ParsedDocument, Section

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 150


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    heading_path: tuple[str, ...]
    page: int | None

    def embedding_text(self, title: str) -> str:
        """Text sent to the embedder: title and headings give a chunk its context."""
        context = " > ".join((title, *self.heading_path))
        return f"{context}\n\n{self.text}"


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _is_table(paragraph: str) -> bool:
    lines = paragraph.splitlines()
    return len(lines) >= 2 and "|" in lines[0] and bool(_TABLE_SEPARATOR.match(lines[1]))


def _table_units(paragraph: str, max_chars: int) -> list[str]:
    """Splits a Markdown table by rows; every piece repeats the header and separator."""
    lines = paragraph.splitlines()
    header = "\n".join(lines[:2])
    units: list[str] = []
    current = header
    for row in lines[2:]:
        if current != header and len(current) + 1 + len(row) > max_chars:
            units.append(current)
            current = header
        current += "\n" + row
    units.append(current)
    return units


def _units(text: str, max_chars: int) -> list[str]:
    """Tables by rows, then paragraphs, sentences and hard slices, each at most max_chars."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _is_table(paragraph):
            units.extend(_table_units(paragraph, max_chars))
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > max_chars:
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence.strip():
                units.append(sentence.strip())
    return units


def _split_section(section: Section, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for unit in _units(section.text, max_chars):
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        if _is_table(unit):
            # A table piece must start with its header, so it gets no overlap prefix.
            current = unit
            continue
        tail = current[-overlap:] if overlap else ""
        tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
        current = f"{tail} {unit}".strip() if len(tail) + len(unit) < max_chars else unit
    if current:
        pieces.append(current)
    return pieces


def chunk_document(
    document: ParsedDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    chunks: list[Chunk] = []
    for section in document.sections:
        for text in _split_section(section, max_chars, overlap_chars):
            chunks.append(Chunk(len(chunks), text, section.heading_path, section.page))
    return chunks
