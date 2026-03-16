import numpy as np
import pytest

from src.chunking import Chunk
from src.config import Settings
from src.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
    TfidfRetriever,
    build_retriever_from_config,
    BaseRetriever,
)


def _make_chunks(texts):
    chunks = []
    for i, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_id=f"c{i}",
                text=text,
                source="src",
                title="t",
                section="",
                doc_id=f"d{i}",
            )
        )
    return chunks


class DummyModel:
    def __init__(self, vectors):
        self.vectors = np.array(vectors, dtype=float)

    def encode(self, texts, batch_size=1, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False):
        # Return stored vectors for corpus; for query, use first vector
        if len(texts) == len(self.vectors):
            return self.vectors
        # single query
        return np.array([self.vectors[0]])


def test_tfidf_retriever_orders_by_score():
    chunks = _make_chunks(["cat on mat", "dog at park"])
    retriever = TfidfRetriever(top_k=2)
    retriever.index(chunks)
    results = retriever.search("cat")
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c0"
    assert results[0].score >= results[1].score


def test_bm25_retriever_basic():
    chunks = _make_chunks(["apple pie recipe", "soccer match report"])
    retriever = BM25Retriever(top_k=1)
    retriever.index(chunks)
    results = retriever.search("apple")
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c0"


def test_dense_retriever_with_dummy_model():
    chunks = _make_chunks(["first", "second"])
    dummy_vectors = [[1, 0], [0, 1]]
    retriever = DenseRetriever(model_name="dummy", model=DummyModel(dummy_vectors), top_k=2, normalize_embeddings=False)
    retriever.index(chunks)
    results = retriever.search("query")
    assert [r.chunk.chunk_id for r in results] == ["c0", "c1"]


class _StubRetriever(BaseRetriever):
    name = "stub"

    def __init__(self, scores):
        super().__init__(top_k=2)
        self.scores = scores
        self._chunks = []

    def index(self, chunks):
        self._chunks = list(chunks)
        self._indexed = True

    def search(self, query, top_k=None):
        self._ensure_indexed()
        results = []
        for idx, chunk in enumerate(self._chunks):
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=self.scores[idx],
                    rank=idx + 1,
                    retriever=self.name,
                )
            )
        return sorted(results, key=lambda r: r.score, reverse=True)[: top_k or self.top_k]


def test_hybrid_fusion_prefers_weighted_scores():
    chunks = _make_chunks(["a", "b"])
    lexical = _StubRetriever(scores=[0.9, 0.1])
    dense = _StubRetriever(scores=[0.2, 0.8])
    hybrid = HybridRetriever(lexical=lexical, dense=dense, weight_lexical=0.3, weight_dense=0.7, top_k=2)
    hybrid.index(chunks)
    results = hybrid.search("anything", top_k=2)
    # Dense score dominates, so chunk1 should be ranked first
    assert results[0].chunk.chunk_id == "c1"


def test_factory_builds_default_hybrid():
    settings = Settings()
    retriever = build_retriever_from_config(settings)
    assert isinstance(retriever, HybridRetriever)


def test_factory_accepts_name_override():
    settings = Settings()
    retriever = build_retriever_from_config(settings, retriever_name="bm25")
    assert isinstance(retriever, BM25Retriever)
