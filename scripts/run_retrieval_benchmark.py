"""Quick-and-dirty retrieval benchmark on toy data.

Runs TF-IDF, BM25, Dense (if available), and Hybrid over a few sample
queries, computing Recall@k and MRR. This is meant for sanity checks and
comparisons, not rigorous evaluation.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.chunking import Chunk, chunk_document
from src.config import Settings
from src.ingestion import Document, normalize_text
from src.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    TfidfRetriever,
    build_retriever_from_config,
)

# ---------------------------------------------------------------------------
# Toy corpus and queries
# ---------------------------------------------------------------------------


def _toy_corpus() -> List[Document]:
    data = [
        {
            "text": "Cats are small domesticated carnivores. They love naps and chasing lasers.",
            "title": "About Cats",
            "source": "toy",
        },
        {
            "text": "Dogs are loyal companions. They enjoy walks, fetching balls, and treats.",
            "title": "About Dogs",
            "source": "toy",
        },
        {
            "text": "Parrots are intelligent birds that can mimic human speech and need enrichment.",
            "title": "About Parrots",
            "source": "toy",
        },
    ]
    docs: List[Document] = []
    for item in data:
        text = normalize_text(item["text"])
        docs.append(
            Document(
                doc_id=f"doc_{item['title'].lower().replace(' ', '_')}",
                text=text,
                source=item["source"],
                title=item["title"],
                section=None,
            )
        )
    return docs


def _toy_queries() -> List[Dict[str, object]]:
    return [
        {"query": "What pet chases lasers?", "relevant_doc_ids": {"doc_about_cats"}},
        {"query": "Which animal enjoys walks and fetching balls?", "relevant_doc_ids": {"doc_about_dogs"}},
        {"query": "Bird that mimics human speech?", "relevant_doc_ids": {"doc_about_parrots"}},
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[Chunk], relevant: set[str], k: int) -> float:
    top = list(retrieved)[:k]
    hits = sum(1 for c in top if c.doc_id in relevant)
    return hits / max(len(relevant), 1)


def reciprocal_rank(retrieved: Sequence[Chunk], relevant: set[str]) -> float:
    for idx, chunk in enumerate(retrieved, start=1):
        if chunk.doc_id in relevant:
            return 1.0 / idx
    return 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(include_dense: bool = True) -> None:
    settings = Settings()
    docs = _toy_corpus()
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size=settings.retrieval.chunk_size, chunk_overlap=settings.retrieval.chunk_overlap))

    queries = _toy_queries()

    retrievers = {
        "tfidf": TfidfRetriever(top_k=settings.retrieval.top_k),
        "bm25": BM25Retriever(top_k=settings.retrieval.top_k, k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b),
    }

    if include_dense:
        try:
            retrievers["dense"] = DenseRetriever(model_name=settings.models.embedding_model, top_k=settings.retrieval.top_k)
            retrievers["hybrid"] = HybridRetriever(
                lexical=retrievers["bm25"],
                dense=retrievers["dense"],
                weight_lexical=settings.retrieval.hybrid_weight_lexical,
                weight_dense=settings.retrieval.hybrid_weight_dense,
                top_k=settings.retrieval.top_k,
            )
        except Exception as exc:  # pragma: no cover - best-effort for environments without weights
            print(f"[warning] Dense model unavailable ({exc}); skipping dense/hybrid retrievers.")

    ready: Dict[str, object] = {}
    for name, retriever in retrievers.items():
        try:
            retriever.index(chunks)
            ready[name] = retriever
        except Exception as exc:
            print(f"[warning] Skipping retriever '{name}' due to initialization error: {exc}")

    print("Retriever,Recall@K,MRR")
    for name, retriever in ready.items():
        recall_scores: List[float] = []
        mrr_scores: List[float] = []
        for item in queries:
            results = retriever.search(item["query"], top_k=settings.retrieval.top_k)
            retrieved_chunks = [r.chunk for r in results]
            relevant = item["relevant_doc_ids"]
            recall_scores.append(recall_at_k(retrieved_chunks, relevant, settings.retrieval.top_k))
            mrr_scores.append(reciprocal_rank(retrieved_chunks, relevant))
        print(f"{name},{np.mean(recall_scores):.2f},{np.mean(mrr_scores):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a toy retrieval benchmark.")
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Skip dense + hybrid retrievers (default True to avoid heavy deps)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(include_dense=not args.no_dense)
