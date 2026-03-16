import pytest

from src.pipeline import PipelineResult
from src.schemas import Chunk, GeneratedAnswer, QueryTrace, RetrievalResult, StageTimings


def _dummy_retrieval(rank: int = 1) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"c{rank}",
        text="example chunk text",
        source="doc.txt",
        title="Doc Title",
        section="s1",
        doc_id="d1",
    )
    return RetrievalResult(chunk=chunk, score=0.5, rank=rank, retriever="bm25")


def test_pipeline_result_instantiation_and_defaults():
    answer = GeneratedAnswer(
        answer="",
        citations=[],
        prompt="p",
        context="ctx",
        provider="prov",
        model="m",
    )
    trace = QueryTrace(query="q", latency_ms=1.0, retriever="bm25")

    result = PipelineResult(
        query="q",
        retriever="bm25",
        reranker=None,
        retrieval_results=[_dummy_retrieval()],
        answer=answer,
        trace=trace,
        prompt="p",
        context="ctx",
    )

    assert result.reranked_results == []
    assert result.reranked_candidates == []
    assert result.timings is None


def test_pipeline_result_to_dict_includes_new_fields():
    answer = GeneratedAnswer(
        answer="response",
        citations=["c1"],
        prompt="prompt",
        context="context",
        provider="prov",
        model="m",
    )
    trace = QueryTrace(query="q", latency_ms=2.0, retriever="bm25")
    base = _dummy_retrieval(rank=1)
    reranked = _dummy_retrieval(rank=2)
    timings = StageTimings(retrieval_ms=1.0, rerank_ms=0.5, diversity_ms=0.2, generation_ms=3.0, total_ms=4.7)

    result = PipelineResult(
        query="q",
        retriever="bm25",
        reranker="keyword-overlap",
        retrieval_results=[base],
        answer=answer,
        trace=trace,
        prompt="prompt",
        context="context",
        reranked_results=[reranked],
        reranked_candidates=[base, reranked],
        timings=timings,
    )

    payload = result.to_dict()

    assert payload["retrieval_results"] and payload["reranked_results"]
    assert payload["reranked_candidates"]
    assert payload["grounding"]["citations"] == ["c1"]
    assert payload["timings"]["total_ms"] == pytest.approx(4.7)
