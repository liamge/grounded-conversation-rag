import pytest

from src.schemas import Chunk, RetrievalResult
from src.diversity import apply_diversity_filter


def _make_result(chunk_id, text, doc_id, score):
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        source="src",
        title="title",
        section="",
        doc_id=doc_id,
    )
    return RetrievalResult(chunk=chunk, score=score, rank=0, retriever="stub")


def test_removes_near_duplicates():
    results = [
        _make_result("c1", "alpha beta gamma", "d1", 0.9),
        _make_result("c2", "alpha beta gamma", "d2", 0.8),
        _make_result("c3", "completely different", "d3", 0.7),
    ]

    filtered = apply_diversity_filter(results, top_k=3, duplicate_similarity_threshold=0.8)

    assert [r.chunk.chunk_id for r in filtered] == ["c1", "c3"]
    assert [r.rank for r in filtered] == [1, 2]


def test_limits_chunks_per_document():
    results = [
        _make_result("c1", "first", "d1", 0.9),
        _make_result("c2", "second", "d1", 0.8),
        _make_result("c3", "third", "d2", 0.7),
    ]

    filtered = apply_diversity_filter(
        results,
        top_k=3,
        max_chunks_per_document=1,
        duplicate_similarity_threshold=0.0,
    )

    assert [r.chunk.chunk_id for r in filtered] == ["c1", "c3"]
    assert all(r.chunk.doc_id in {"d1", "d2"} for r in filtered)


def test_disabled_filter_truncates_only():
    results = [
        _make_result("c1", "one", "d1", 0.9),
        _make_result("c2", "two", "d1", 0.8),
        _make_result("c3", "three", "d2", 0.7),
    ]

    filtered = apply_diversity_filter(results, top_k=2, enable=False)

    assert [r.chunk.chunk_id for r in filtered] == ["c1", "c2"]
    assert [r.rank for r in filtered] == [1, 2]
