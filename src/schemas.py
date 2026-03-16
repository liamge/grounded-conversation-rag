"""Shared data models for the Grounded Conversation RAG system.

These dataclasses centralize the core entities used across ingestion,
chunking, retrieval, generation, evaluation, and telemetry. Keeping them
in one module avoids duplication and ensures type consistency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Corpus objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Document:
    """Normalized document representation used throughout the pipeline."""

    doc_id: str
    text: str
    source: str
    title: str
    section: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Union[str, Dict[str, str]]]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "section": self.section or "",
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Chunk:
    """Retrieval-ready chunk with source metadata."""

    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    doc_id: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        payload = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "section": self.section,
            "doc_id": self.doc_id,
        }
        payload.update(self.metadata)
        return payload


# ---------------------------------------------------------------------------
# Retrieval + generation outputs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RetrievalResult:
    """Structured retrieval output for a single chunk."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "retriever": self.retriever,
            "chunk": self.chunk.to_dict(),
        }


@dataclass(slots=True)
class GeneratedAnswer:
    """Grounded answer produced by a generator."""

    answer: str
    citations: List[str]
    prompt: str
    context: str
    provider: str
    model: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "prompt": self.prompt,
            "context": self.context,
            "provider": self.provider,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Evaluation records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvalExample:
    """Single evaluation item loaded from JSONL."""

    example_id: str
    query: str
    relevant_doc_ids: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    expected_facts: List[str] = field(default_factory=list)
    should_abstain: Optional[bool] = None

    @classmethod
    def from_record(cls, raw: Dict[str, object], idx: int) -> "EvalExample":
        """Build from a dictionary record with flexible key names."""

        def _get(keys: Sequence[str], default):
            for key in keys:
                if key in raw:
                    return raw[key]
            return default

        example_id = str(_get(["id", "example_id"], f"ex_{idx}"))
        query = str(_get(["query", "question"], "")).strip()
        rel_docs = list(_get(["relevant_doc_ids", "relevant_docs", "gold_doc_ids"], []))
        rel_chunks = list(_get(["relevant_chunk_ids", "gold_chunk_ids"], []))
        reference_answer = _get(["reference_answer", "ideal_answer", "answer"], None)
        expected_facts = list(_get(["expected_facts", "required_facts"], []))
        should_abstain = _get(["should_abstain", "must_abstain"], None)

        return cls(
            example_id=example_id,
            query=query,
            relevant_doc_ids=[str(d) for d in rel_docs],
            relevant_chunk_ids=[str(c) for c in rel_chunks],
            reference_answer=str(reference_answer) if reference_answer is not None else None,
            expected_facts=[str(f) for f in expected_facts],
            should_abstain=bool(should_abstain) if should_abstain is not None else None,
        )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class QueryTrace:
    """Per-query telemetry record."""

    query: str
    latency_ms: float
    retriever: str
    top_scores: List[float] = field(default_factory=list)
    num_citations: int = 0
    abstained: bool = False
    prompt_chars: int = 0
    context_chars: int = 0
    answer_chars: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_row(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "latency_ms": round(self.latency_ms, 2),
            "retriever": self.retriever,
            "top_scores": ";".join(f"{s:.4f}" for s in self.top_scores),
            "num_citations": self.num_citations,
            "abstained": self.abstained,
            "prompt_chars": self.prompt_chars,
            "context_chars": self.context_chars,
            "answer_chars": self.answer_chars,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_row(), ensure_ascii=False)


__all__ = [
    "Document",
    "Chunk",
    "RetrievalResult",
    "GeneratedAnswer",
    "EvalExample",
    "QueryTrace",
]
