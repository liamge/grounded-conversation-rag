"""Precompute and persist dense embeddings + FAISS index.

Usage:
    python scripts/build_embeddings.py \
        --config config/settings.yaml \
        --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.config import Settings
from src.ingestion import ingest_documents
from src.chunking import chunk_document
from src.retrieval import DenseRetriever


def build_embeddings(settings: Settings, model_name: str | None = None) -> None:
    settings.ensure_output_dirs()
    model_name = model_name or settings.models.embedding_model

    # Ingest and chunk
    documents = ingest_documents([settings.data.raw_dir])
    chunks: List = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                doc,
                chunk_size=settings.retrieval.chunk_size,
                chunk_overlap=settings.retrieval.chunk_overlap,
            )
        )

    retriever = DenseRetriever(
        model_name=model_name,
        top_k=settings.retrieval.top_k,
        min_score=settings.retrieval.min_score,
        embeddings_dir=settings.data.embeddings_dir,
        use_faiss=True,
        prefer_cache=False,
        force_recompute=True,
    )
    retriever.index(chunks)
    print(
        f"Saved embeddings for {len(chunks)} chunks to {settings.data.embeddings_dir} "
        f"using model {model_name}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and persist dense embeddings.")
    parser.add_argument("--config", type=str, default=None, help="Path to settings YAML.")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    settings = Settings.load(args.config) if args.config else Settings.load()
    build_embeddings(settings, model_name=args.model)
