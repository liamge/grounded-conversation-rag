from src.config import Settings
from src.generation import (
    LightweightExtractiveGenerator,
    _split_sentences,
    assemble_context_with_budget,
    build_generator,
    generate_answer,
)
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


def test_generate_uses_extractive_demo_when_no_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings()
    settings.models.llm_provider = "openai"

    generator = build_generator(settings)

    assert isinstance(generator, LightweightExtractiveGenerator)


def test_sentence_split_basic() -> None:
    text = "This is one. And another? Last!"
    sentences = _split_sentences(text)

    assert sentences == ["This is one.", "And another?", "Last!"]


def test_lightweight_generator_prefers_relevant_sentences(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings()
    retrieved = [
        _res("chunk_a", "The pipeline uses hybrid retrieval. It blends dense and lexical signals."),
        _res("chunk_b", "Unrelated filler text with no useful info."),
    ]

    answer = generate_answer("How does the pipeline combine signals?", retrieved, settings=settings)

    assert answer.citations == ["chunk_a"]
    assert "hybrid retrieval" in answer.answer.lower()


def test_lightweight_generator_deduplicates_similar_sentences(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    retrieved = [
        _res(
            "chunk_a",
            "Hybrid retrieval combines lexical and dense scores. "
            "Hybrid retrieval combines lexical and dense scores.",
        ),
        _res("chunk_b", "Lexical and dense scores are merged in the hybrid approach."),
    ]

    generator = LightweightExtractiveGenerator(max_sentences=3)
    answer = generator.generate("What is hybrid retrieval?", retrieved)

    # Only one variant should remain after redundancy filtering
    assert answer.answer.count("chunk_a") + answer.answer.count("chunk_b") <= 2
    assert len(answer.citations) == len(set(answer.citations))


def test_fallback_routing_ignores_openai_when_not_requested(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    settings = Settings()
    settings.models.llm_provider = "fallback"

    generator = build_generator(settings)

    assert isinstance(generator, LightweightExtractiveGenerator)
