"""Post-retrieval diversity utilities.

This module provides a deterministic filter that removes near-duplicate
chunks, optionally limits the number of chunks per source document, and
renumbers ranks while preserving the original retrieval scores and metadata.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Sequence

from .schemas import RetrievalResult


def _text_similarity(a: str, b: str) -> float:
    """Return a stable similarity ratio between two strings.

    ``SequenceMatcher`` is deterministic and available in the standard
    library, making it a reasonable choice for near-duplicate detection
    without introducing extra dependencies.
    """

    return SequenceMatcher(None, a, b).ratio()


def apply_diversity_filter(
    results: Sequence[RetrievalResult],
    *,
    top_k: int,
    enable: bool = True,
    max_chunks_per_document: int = 0,
    duplicate_similarity_threshold: float = 0.9,
) -> List[RetrievalResult]:
    """Select a diverse subset of retrieval results.

    The filter walks the ranked list in order, keeping the first instance of
    any near-duplicate chunk (based on ``duplicate_similarity_threshold``)
    and enforcing an optional per-document quota. Returned results are
    re-ranked starting at 1 while preserving the original score and chunk
    metadata. When disabled, the input list is truncated to ``top_k``.
    """

    budget = max(top_k, 0)
    if budget == 0:
        return []

    if not enable:
        return [
            RetrievalResult(
                chunk=res.chunk,
                score=res.score,
                rank=idx,
                retriever=res.retriever,
            )
            for idx, res in enumerate(results[:budget], start=1)
        ]

    selected: List[RetrievalResult] = []
    per_doc = defaultdict(int)
    threshold = max(0.0, float(duplicate_similarity_threshold))

    for res in results:
        if max_chunks_per_document > 0 and per_doc[res.chunk.doc_id] >= max_chunks_per_document:
            continue

        is_duplicate = False
        if threshold > 0.0:
            for kept in selected:
                if _text_similarity(res.chunk.text, kept.chunk.text) >= threshold:
                    is_duplicate = True
                    break
        if is_duplicate:
            continue

        per_doc[res.chunk.doc_id] += 1
        selected.append(res)

        if len(selected) >= budget:
            break

    diversified: List[RetrievalResult] = []
    for idx, res in enumerate(selected, start=1):
        diversified.append(
            RetrievalResult(
                chunk=res.chunk,
                score=res.score,
                rank=idx,
                retriever=res.retriever,
            )
        )

    return diversified


__all__ = ["apply_diversity_filter"]
