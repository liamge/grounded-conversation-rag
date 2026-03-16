"""Document ingestion utilities for the Grounded Conversation RAG.

This module provides lightweight, dependency-free loaders for plain text,
markdown, and simple JSON sources. Each file is normalized into a common
``Document`` object with deterministic identifiers so downstream components
can rely on stable metadata.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .schemas import Document


# Supported extensions for ingestion.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".jsonl"}

# Namespace used to derive deterministic document IDs.
_DOC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "grounded-conversation-rag/document")


_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize raw document text.

    Operations:
    * Convert Windows/Mac newlines to ``\n``
    * Collapse repeated whitespace into single spaces
    * Trim leading/trailing whitespace
    """

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _whitespace_re.sub(" ", cleaned)
    return cleaned.strip()


def _derive_title_from_markdown(text: str, fallback: str) -> str:
    """Use the first markdown heading as title, otherwise the fallback."""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            # Remove leading #'s and trailing hash decorations.
            heading = stripped.lstrip("#").strip(" #")
            if heading:
                return heading
    return fallback


def _deterministic_doc_id(text: str, source: str, section: Optional[str] = None) -> str:
    """Create a stable document id from content, source, and section."""

    normalized = normalize_text(text)
    key = "|".join([source, section or "", normalized])
    return f"doc_{uuid.uuid5(_DOC_NAMESPACE, key).hex[:12]}"


def _load_text_like(path: Path, kind: str) -> Document:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    title_fallback = path.stem
    title = _derive_title_from_markdown(raw, title_fallback) if kind == "md" else title_fallback
    text = normalize_text(raw)
    doc_id = _deterministic_doc_id(text, str(path))
    return Document(doc_id=doc_id, text=text, source=str(path), title=title)


def _load_json(path: Path) -> List[Document]:
    """Load simple JSON documents.

    Supported shapes:
    1) List[dict] where each dict contains at least a ``text`` field.
    2) Dict with a top-level ``text`` field (treated as a single document).
    """

    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    def _build(doc_obj: Dict[str, str], idx: int | None = None) -> Document:
        if "text" not in doc_obj:
            raise ValueError(f"JSON item missing 'text' field in {path}")

        text = normalize_text(str(doc_obj.get("text", "")))
        title = str(doc_obj.get("title") or path.stem)
        section = doc_obj.get("section")
        source = str(doc_obj.get("source") or path)

        metadata = {k: str(v) for k, v in doc_obj.items() if k not in {"text", "title", "section", "source"}}
        if idx is not None:
            metadata.setdefault("index", str(idx))

        doc_id = _deterministic_doc_id(text, source, section)
        return Document(doc_id=doc_id, text=text, source=source, title=title, section=section, metadata=metadata)

    if isinstance(payload, list):
        return [_build(item, idx=i) for i, item in enumerate(payload)]

    if isinstance(payload, dict):
        return [_build(payload)]

    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_jsonl(path: Path) -> List[Document]:
    """Load JSON Lines where each line is a dict with a ``text`` field."""

    records: List[Document] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            records.extend(_load_json_obj(obj, path, idx))
    return records


def _load_json_obj(obj: object, path: Path, idx: int) -> List[Document]:
    """Helper to reuse JSON handling for jsonl lines."""

    def _build(doc_obj: Dict[str, str], i: int | None = None) -> Document:
        if "text" not in doc_obj:
            raise ValueError(f"JSON item missing 'text' field in {path}")
        text = normalize_text(str(doc_obj.get("text", "")))
        title = str(doc_obj.get("title") or path.stem)
        section = doc_obj.get("section")
        source = str(doc_obj.get("source") or path)

        metadata = {k: str(v) for k, v in doc_obj.items() if k not in {"text", "title", "section", "source"}}
        if i is not None:
            metadata.setdefault("index", str(i))

        doc_id = _deterministic_doc_id(text, source, section)
        return Document(doc_id=doc_id, text=text, source=source, title=title, section=section, metadata=metadata)

    if isinstance(obj, list):
        return [_build(item, i=idx) for i, item in enumerate(obj)]
    if isinstance(obj, dict):
        return [_build(obj, i=idx)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_document(path: Union[str, Path]) -> List[Document]:
    """Load a single file into one or more ``Document`` objects."""

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")

    if suffix == ".json":
        return _load_json(p)
    if suffix == ".jsonl":
        return _load_jsonl(p)
    if suffix == ".md":
        return [_load_text_like(p, kind="md")]
    # Default to plain text for .txt and other registered extensions
    return [_load_text_like(p, kind="txt")]


def iter_supported_files(paths: Sequence[Union[str, Path]]) -> Iterable[Path]:
    """Yield supported files from a sequence of files or directories."""

    for entry in paths:
        p = Path(entry)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield child
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def ingest_documents(paths: Sequence[Union[str, Path]]) -> List[Document]:
    """Ingest all supported documents under the provided paths.

    Parameters
    ----------
    paths:
        Sequence of file or directory paths to scan.

    Returns
    -------
    List[Document]
        Flattened list of normalized ``Document`` objects.
    """

    documents: List[Document] = []
    for file_path in iter_supported_files(paths):
        documents.extend(load_document(file_path))
    return documents


__all__ = ["Document", "ingest_documents", "load_document", "normalize_text", "SUPPORTED_EXTENSIONS"]
