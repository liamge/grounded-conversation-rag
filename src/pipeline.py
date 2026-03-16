"""End-to-end orchestration for the Grounded Conversation RAG system.

The pipeline coordinates the main stages in a single, reusable object:

1. Load settings (YAML + env overrides)
2. Ingest source documents
3. Create retrieval-ready chunks
4. Build and index retrievers
5. Run retrieval (hybrid/lexical/dense), optionally rerank results
6. Generate a grounded answer with citations
7. Emit a structured ``PipelineResult`` that includes a ``QueryTrace`` for
   telemetry as well as the prompt/context used for generation.

The goal is to give the Streamlit app and evaluation code a single entry
point that hides wiring details while keeping components testable on their
own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .chunking import chunk_documents
from .config import Settings
from .evaluation import is_abstention
from .generation import generate_answer
from .ingestion import ingest_documents
from .retrieval import BaseRetriever, RetrievalResult, build_retriever_from_config
from .schemas import Chunk, Document, GeneratedAnswer, QueryTrace
from .reranking import BaseReranker, KeywordOverlapReranker, build_reranker_from_config


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineResult:
    """Structured output produced by ``RAGPipeline.run``."""

    query: str
    retriever: str
    reranker: Optional[str]
    retrieval_results: List[RetrievalResult]
    reranked_results: List[RetrievalResult]
    answer: GeneratedAnswer
    trace: QueryTrace
    prompt: str
    context: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retriever": self.retriever,
            "reranker": self.reranker,
            "retrieval_results": [r.to_dict() for r in self.retrieval_results],
            "reranked_results": [r.to_dict() for r in self.reranked_results],
            "answer": self.answer.to_dict(),
            "trace": self.trace.to_row(),
            "prompt": self.prompt,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class RAGPipeline:
    """High-level orchestration for ingestion → retrieval → generation."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        config_path: Optional[Path | str] = None,
        retriever: Optional[BaseRetriever] = None,
        reranker: Optional[BaseReranker] = None,
    ) -> None:
        self.settings = settings or Settings.load(path=config_path)
        self.settings.ensure_output_dirs()

        self._documents: List[Document] = []
        self._chunks: List[Chunk] = []
        self._retriever: BaseRetriever = retriever or build_retriever_from_config(self.settings)
        self._reranker: Optional[BaseReranker] = reranker
        self._retriever_ready = False

    # Public helpers ----------------------------------------------------

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    @property
    def documents(self) -> List[Document]:
        return self._documents

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def reranker(self) -> Optional[BaseReranker]:
        return self._reranker

    # Corpus + index setup ---------------------------------------------

    def load_corpus(self, sources: Optional[Sequence[str | Path]] = None, *, force: bool = False) -> None:
        """Ingest and chunk documents, caching results for reuse."""

        if self._chunks and not force:
            return

        raw_sources = sources or [self.settings.data.raw_dir]
        self._documents = ingest_documents(raw_sources)
        if not self._documents:
            raise ValueError(
                "No supported documents were found to ingest. Add source files under"
                f" {raw_sources} or update settings.data.raw_dir."
            )
        self._chunks = chunk_documents(
            self._documents,
            chunk_size=self.settings.retrieval.chunk_size,
            chunk_overlap=self.settings.retrieval.chunk_overlap,
        )
        if not self._chunks:
            raise ValueError("Chunking produced no results; check chunking parameters or input texts.")
        self._retriever_ready = False

    def index(self, *, retriever_name: Optional[str] = None, force: bool = False) -> None:
        """Build or rebuild the retriever index over current chunks."""

        if retriever_name and retriever_name != self._retriever.name:
            self._retriever = build_retriever_from_config(self.settings, retriever_name=retriever_name)
            self._retriever_ready = False

        if self._retriever_ready and not force:
            return

        if not self._chunks:
            self.load_corpus()

        self._retriever.index(self._chunks)
        self._retriever_ready = True

    # Query execution ---------------------------------------------------

    def run(
        self,
        query: str,
        *,
        retriever_name: Optional[str] = None,
        top_k: Optional[int] = None,
        max_context_chars: Optional[int] = None,
        system_instruction: Optional[str] = None,
        use_reranker: Optional[bool] = None,
    ) -> PipelineResult:
        """Execute the full RAG pipeline for a single query."""

        self.load_corpus()
        self.index(retriever_name=retriever_name)

        k = top_k or self.settings.retrieval.top_k
        rerank_enabled = use_reranker if use_reranker is not None else self.settings.retrieval.use_reranker

        if rerank_enabled and self._reranker is None:
            self._reranker = build_reranker_from_config(self.settings)

        # Fetch enough candidates to allow reranking when requested.
        candidate_k = max(k, self.settings.retrieval.reranker_top_n) if rerank_enabled else k

        start = time.perf_counter()
        retrieval_results = self._retriever.search(query, top_k=candidate_k)

        reranked_results = self._apply_reranker(query, retrieval_results, top_n=k) if rerank_enabled else []
        final_results = reranked_results or retrieval_results[:k]

        answer = generate_answer(
            query,
            final_results,
            settings=self.settings,
            max_context_chars=max_context_chars or self.settings.retrieval.chunk_size * k,
            system_instruction=system_instruction,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        trace = self._build_trace(
            query=query,
            results=final_results,
            generation=answer,
            latency_ms=elapsed_ms,
            rerank_used=bool(reranked_results),
        )

        return PipelineResult(
            query=query,
            retriever=self._retriever.name,
            reranker=self._reranker.name if reranked_results else None,
            retrieval_results=retrieval_results,
            reranked_results=final_results,
            answer=answer,
            trace=trace,
            prompt=answer.prompt,
            context=answer.context,
        )

    # Internal helpers --------------------------------------------------

    def _apply_reranker(
        self, query: str, results: Sequence[RetrievalResult], top_n: int
    ) -> List[RetrievalResult]:
        if not self._reranker:
            return []
        if not results:
            return []
        return self._reranker.rerank(query, results, top_n)

    def _build_trace(
        self,
        *,
        query: str,
        results: Sequence[RetrievalResult],
        generation: GeneratedAnswer,
        latency_ms: float,
        rerank_used: bool,
    ) -> QueryTrace:
        retriever_label = self._retriever.name
        if rerank_used and self._reranker:
            retriever_label = f"{retriever_label}+{self._reranker.name}"

        top_scores = [float(res.score) for res in results[: self.settings.retrieval.top_k]]

        return QueryTrace(
            query=query.strip(),
            latency_ms=float(latency_ms),
            retriever=retriever_label,
            top_scores=top_scores,
            num_citations=len(generation.citations),
            abstained=is_abstention(generation.answer),
            prompt_chars=len(generation.prompt),
            context_chars=len(generation.context),
            answer_chars=len(generation.answer),
        )


__all__ = ["RAGPipeline", "PipelineResult", "BaseReranker", "KeywordOverlapReranker"]
