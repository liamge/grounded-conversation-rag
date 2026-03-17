"""Reranking utilities for the Grounded Conversation RAG.

This module keeps reranking optional while preferring a stronger
cross-encoder when the dependency is available. A lightweight keyword
overlap fallback is provided to avoid pulling in heavy models in
resource-constrained environments.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

from .config import Settings
from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

# Allow disabling sentence-transformers entirely (e.g., envs where libtorch crashes)
_DISABLE_DENSE = os.getenv("RAG_DISABLE_DENSE", "").lower() in {"1", "true", "yes"}

if _DISABLE_DENSE:
    CrossEncoder = None  # type: ignore
    _HAS_CROSS_ENCODER = False
else:
    try:  # Optional heavy dependency
        from sentence_transformers import CrossEncoder  # type: ignore

        _HAS_CROSS_ENCODER = True
    except Exception:  # pragma: no cover - optional path
        CrossEncoder = None  # type: ignore
        _HAS_CROSS_ENCODER = False


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseReranker(Protocol):
    """Minimal reranker interface used by the pipeline."""

    name: str

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_n: int
    ) -> List[RetrievalResult]:
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class KeywordOverlapReranker:
    """Heuristic reranker based on token overlap with the query."""

    name = "keyword-overlap"

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None) -> None:
        self.tokenizer = tokenizer or self._default_tokenizer

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        import re

        return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_n: int
    ) -> List[RetrievalResult]:
        if not candidates:
            return []

        query_tokens = set(self.tokenizer(query))
        scored: List[Tuple[float, RetrievalResult]] = []

        for res in candidates:
            chunk_tokens = set(self.tokenizer(res.chunk.text))
            overlap = len(query_tokens & chunk_tokens)
            coverage = overlap / max(len(query_tokens), 1)
            combined = 0.7 * coverage + 0.3 * float(res.score)
            scored.append((combined, res))

        scored.sort(key=lambda x: x[0], reverse=True)
        limited = scored[: max(top_n, 0)] if top_n else scored

        reranked: List[RetrievalResult] = []
        for new_rank, (score, res) in enumerate(limited, start=1):
            reranked.append(
                RetrievalResult(
                    chunk=res.chunk,
                    score=float(score),
                    rank=new_rank,
                    retriever=res.retriever,
                )
            )
        return reranked


class CrossEncoderReranker:
    """Cross-encoder reranker using Sentence-Transformers."""

    name = "cross-encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        model: Optional["CrossEncoder"] = None,
    ) -> None:
        if not _HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers CrossEncoder is not available. "
                "Install the optional dependency to enable the cross-encoder reranker."
            )
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Optional["CrossEncoder"] = model

    def _get_model(self) -> "CrossEncoder":
        if self._model is None:
            assert CrossEncoder is not None  # for type checkers
            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[RetrievalResult], top_n: int
    ) -> List[RetrievalResult]:
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, res.chunk.text) for res in candidates]
        scores = model.predict(pairs, batch_size=self.batch_size)

        scored: List[Tuple[float, RetrievalResult]] = []
        for score, res in zip(scores, candidates):
            scored.append((float(score), res))

        scored.sort(key=lambda x: x[0], reverse=True)
        limited = scored[: max(top_n, 0)] if top_n else scored

        reranked: List[RetrievalResult] = []
        for new_rank, (score, res) in enumerate(limited, start=1):
            reranked.append(
                RetrievalResult(
                    chunk=res.chunk,
                    score=float(score),
                    rank=new_rank,
                    retriever=res.retriever,
                )
            )
        return reranked


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_reranker_from_config(settings: Settings) -> BaseReranker:
    """Prefer a cross-encoder reranker; fallback to keyword overlap."""

    model_name = settings.models.reranker_model

    if _HAS_CROSS_ENCODER:
        try:
            return CrossEncoderReranker(model_name=model_name)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(
                "Failed to initialize cross-encoder reranker (%s); falling back. %s",
                model_name,
                exc,
            )

    logger.info("Using keyword overlap reranker (cross-encoder unavailable).")
    return KeywordOverlapReranker()


__all__ = [
    "BaseReranker",
    "KeywordOverlapReranker",
    "CrossEncoderReranker",
    "build_reranker_from_config",
]
