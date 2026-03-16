"""Chunking utilities for the Grounded Conversation RAG.

This module converts ingested ``Document`` objects into smaller ``Chunk``
objects suitable for retrieval. The implementation keeps metadata intact and
generates deterministic chunk identifiers.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Sequence, Tuple

from .ingestion import normalize_text
from .schemas import Chunk, Document


_CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "grounded-conversation-rag/chunk")


def _deterministic_chunk_id(doc_id: str, text: str, offset: int) -> str:
    key = f"{doc_id}|{offset}|{normalize_text(text)}"
    return f"chunk_{uuid.uuid5(_CHUNK_NAMESPACE, key).hex[:12]}"


def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _split_sentences(text: str) -> List[Tuple[int, str]]:
    """Lightweight sentence splitter that preserves character offsets."""

    import re

    boundary = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

    spans: List[Tuple[int, str]] = []
    start = 0

    for match in boundary.finditer(text):
        end = match.start()
        raw = text[start:end]
        stripped = raw.strip()
        if stripped:
            leading_ws = len(raw) - len(raw.lstrip())
            sentence_start = start + leading_ws
            spans.append((sentence_start, stripped))
        start = match.end()

    tail = text[start:]
    if tail.strip():
        leading_ws = len(tail) - len(tail.lstrip())
        spans.append((start + leading_ws, tail.strip()))

    return spans


def _split_long_sentence(sentence: str, offset: int, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, str]]:
    """Fallback splitter for sentences longer than the allowed chunk size."""

    units: List[Tuple[int, str]] = []
    start = 0
    while start < len(sentence):
        end = min(start + chunk_size, len(sentence))
        units.append((offset + start, sentence[start:end]))
        if end == len(sentence):
            break
        start = end - chunk_overlap
    return units


def _joined_length(units: List[Tuple[int, str]]) -> int:
    if not units:
        return 0
    total = sum(len(text) for _, text in units)
    return total + (len(units) - 1)


def _carry_overlap_text(units: List[Tuple[int, str]], chunk_start_offset: int, overlap: int) -> List[Tuple[int, str]]:
    if overlap <= 0 or not units:
        return []

    chunk_text = " ".join(text for _, text in units)
    if not chunk_text:
        return []

    overlap_text = chunk_text[-overlap:]
    overlap_offset = chunk_start_offset + len(chunk_text) - len(overlap_text)
    return [(overlap_offset, overlap_text)]


def _prepare_units(text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[int, str]]:
    """Return sentence-aligned units, further split if any exceed the size."""

    units: List[Tuple[int, str]] = []
    for offset, sentence in _split_sentences(text):
        if len(sentence) <= chunk_size:
            units.append((offset, sentence))
            continue
        units.extend(_split_long_sentence(sentence, offset, chunk_size, chunk_overlap))
    return units


def chunk_document(doc: Document, chunk_size: int = 512, chunk_overlap: int = 64) -> List[Chunk]:
    """Chunk a single document into retrieval-ready pieces."""

    _validate_params(chunk_size, chunk_overlap)

    text = normalize_text(doc.text)
    units = _prepare_units(text, chunk_size, chunk_overlap)
    if not units:
        return []

    chunks: List[Chunk] = []

    current: List[Tuple[int, str]] = []
    current_length = 0
    chunk_start_offset = units[0][0]

    for offset, unit_text in units:
        additional = len(unit_text) if not current else len(unit_text) + 1

        if current and current_length + additional > chunk_size:
            chunk_text = " ".join(text for _, text in current)
            chunk_id = _deterministic_chunk_id(doc.doc_id, chunk_text, chunk_start_offset)
            chunk_metadata = dict(doc.metadata)
            chunk_metadata.update({"offset": str(chunk_start_offset)})

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=doc.source,
                    title=doc.title,
                    section=doc.section or "",
                    doc_id=doc.doc_id,
                metadata=chunk_metadata,
                )
            )

            # Only carry overlap when the chunk combined multiple units (i.e., at sentence boundaries).
            current = _carry_overlap_text(current, chunk_start_offset, chunk_overlap) if len(current) > 1 else []
            current_length = _joined_length(current)
            chunk_start_offset = current[0][0] if current else offset
            additional = len(unit_text) if not current else len(unit_text) + 1

        if not current:
            chunk_start_offset = offset

        current.append((offset, unit_text))
        current_length += additional

    if current:
        chunk_text = " ".join(text for _, text in current)
        chunk_id = _deterministic_chunk_id(doc.doc_id, chunk_text, chunk_start_offset)
        chunk_metadata = dict(doc.metadata)
        chunk_metadata.update({"offset": str(chunk_start_offset)})

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=doc.source,
                title=doc.title,
                section=doc.section or "",
                doc_id=doc.doc_id,
                metadata=chunk_metadata,
            )
        )

    return chunks


def chunk_documents(documents: Sequence[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks


__all__ = ["Chunk", "chunk_document", "chunk_documents"]
