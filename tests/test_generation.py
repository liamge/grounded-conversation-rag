from src.generation import assemble_context_with_budget
from src.schemas import Chunk, RetrievalResult


def _res(chunk_id: str, text: str, rank: int = 1) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        text=text,
        source="src",
        title="Title",
        section="sec",
        doc_id="doc1",
        metadata={},
    )
    return RetrievalResult(chunk=chunk, score=1.0 / rank, rank=rank, retriever="bm25")


def test_truncates_oversized_chunk_and_keeps_id() -> None:
    results = [_res("chunk_big", "word " * 50)]

    context, ids, meta = assemble_context_with_budget(results, max_tokens=6)

    assert ids == ["chunk_big"]
    assert context  # not empty even though budget is small
    assert meta["truncated_chunks"]  # chunk was trimmed
    assert meta["truncated_chunks"][0]["chunk_id"] == "chunk_big"
    assert meta["truncated_chunks"][0]["included_tokens"] <= 6


def test_preserves_rank_order_and_drops_excess() -> None:
    first = _res("chunk_1", "a b c d e f", rank=1)
    second = _res("chunk_2", "g h i j k l", rank=2)

    context, ids, meta = assemble_context_with_budget([first, second], max_tokens=5)

    assert ids == ["chunk_1"]  # only top chunk fits
    assert "chunk_2" in meta["dropped_chunks"]
    assert context.startswith("[chunk_1]")


def test_budget_zero_still_emits_first_chunk() -> None:
    only = _res("chunk_zero", "lone chunk text", rank=1)

    context, ids, meta = assemble_context_with_budget([only], max_tokens=0)

    assert ids == ["chunk_zero"]
    assert context  # enforced minimum budget keeps something
    assert meta["effective_budget_tokens"] == 1
    assert meta["used_tokens"] == 1


def test_empty_input_returns_empty_context() -> None:
    context, ids, meta = assemble_context_with_budget([], max_tokens=10)

    assert context == ""
    assert ids == []
    assert meta["used_tokens"] == 0
    assert not meta["truncated_chunks"]
    assert not meta["dropped_chunks"]
