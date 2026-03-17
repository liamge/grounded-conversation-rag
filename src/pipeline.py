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

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .chunking import chunk_documents
from .config import Settings
from .diversity import apply_diversity_filter
from .evaluation import is_abstention
from .generation import generate_answer
from .index_artifacts import IndexContext, compute_corpus_fingerprint
from .ingestion import ingest_documents
from .logging_utils import log_event
from .reranking import BaseReranker, KeywordOverlapReranker, build_reranker_from_config
from .retrieval import BaseRetriever, HybridRetriever, RetrievalResult, build_retriever_from_config
from .schemas import Chunk, Document, GeneratedAnswer, QueryTrace, StageTimings

_LOGGER = logging.getLogger(__name__)


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
    answer: GeneratedAnswer
    trace: QueryTrace
    prompt: str
    context: str
    reranked_results: List[RetrievalResult] = field(default_factory=list)
    reranked_candidates: List[RetrievalResult] = field(default_factory=list)
    timings: Optional[StageTimings] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "retriever": self.retriever,
            "reranker": self.reranker,
            "retrieval_results": [r.to_dict() for r in self.retrieval_results],
            "reranked_candidates": [r.to_dict() for r in self.reranked_candidates],
            "reranked_results": [r.to_dict() for r in self.reranked_results],
            "answer": self.answer.to_dict(),
            "grounding": {
                "answer": self.answer.answer,
                "citations": list(self.answer.citations),
                "evidence_chunks": [c.to_dict() for c in self.answer.evidence_chunks],
                "supported": self.answer.supported,
            },
            "trace": self.trace.to_row(),
            "prompt": self.prompt,
            "context": self.context,
            "timings": self.timings.to_row() if self.timings else None,
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
        self._corpus_fingerprint: Optional[str] = None
        self._index_context: Optional[IndexContext] = None

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

    def load_corpus(
        self, sources: Optional[Sequence[str | Path]] = None, *, force: bool = False
    ) -> None:
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
        self._corpus_fingerprint = compute_corpus_fingerprint(self._documents)
        self._index_context = IndexContext(
            artifacts_root=self.settings.output.artifacts_dir,
            corpus_fingerprint=self._corpus_fingerprint,
            embedding_model=self.settings.models.embedding_model,
            chunk_size=self.settings.retrieval.chunk_size,
            chunk_overlap=self.settings.retrieval.chunk_overlap,
        )
        self._chunks = chunk_documents(
            self._documents,
            chunk_size=self.settings.retrieval.chunk_size,
            chunk_overlap=self.settings.retrieval.chunk_overlap,
        )
        if not self._chunks:
            raise ValueError(
                "Chunking produced no results; check chunking parameters or input texts."
            )
        self._retriever_ready = False
        self._apply_index_context()

    def index(self, *, retriever_name: Optional[str] = None, force: bool = False) -> None:
        """Build or rebuild the retriever index over current chunks."""

        if retriever_name and retriever_name != self._retriever.name:
            self._retriever = build_retriever_from_config(
                self.settings, retriever_name=retriever_name
            )
            self._retriever_ready = False
            self._apply_index_context()

        if self._retriever_ready and not force:
            return

        if not self._chunks:
            self.load_corpus()

        self._apply_index_context()
        self._retriever.index(self._chunks)
        self._retriever_ready = True
        log_event(
            _LOGGER,
            "index.build",
            message=f"Indexed corpus for retriever {self._retriever.name}",
            retriever=self._retriever.name,
            chunk_count=len(self._chunks),
        )

    def build_all_indices(self, *, force: bool = False) -> None:
        """Materialize indexes for all available retrievers without running a query."""

        self.load_corpus()

        names = ["tfidf", "bm25"]
        disable_dense = os.getenv("RAG_DISABLE_DENSE", "").lower() in {"1", "true", "yes"}
        if not disable_dense:
            names.extend(["dense", "hybrid"])

        for name in names:
            retriever = build_retriever_from_config(self.settings, retriever_name=name)
            self._apply_index_context_to(retriever)
            if force:
                try:
                    target = (
                        self._index_context.retriever_dir(retriever.name)
                        if self._index_context
                        else None
                    )
                    if target and target.exists():
                        shutil.rmtree(target)
                except Exception as exc:  # pragma: no cover - defensive
                    _LOGGER.warning("Failed to clear artifacts for %s: %s", name, exc)

            # Dense retriever can be forced via flag; lexical ones rebuild after artifact purge.
            if force and hasattr(retriever, "force_recompute"):
                try:
                    setattr(retriever, "force_recompute", True)
                except Exception:
                    pass

            try:
                retriever.index(self._chunks)
                log_event(
                    _LOGGER,
                    "index.build",
                    message=f"Indexed corpus for retriever {retriever.name}",
                    retriever=retriever.name,
                    chunk_count=len(self._chunks),
                )
            except ImportError as exc:
                _LOGGER.warning("Skipping retriever %s due to missing optional deps: %s", name, exc)
            except Exception as exc:  # pragma: no cover - defensive
                _LOGGER.warning("Failed to build index for %s: %s", name, exc)

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
        rerank_enabled = (
            use_reranker if use_reranker is not None else self.settings.retrieval.use_reranker
        )
        diversity_enabled = self.settings.retrieval.enable_diversity_filter

        if rerank_enabled and self._reranker is None:
            self._reranker = build_reranker_from_config(self.settings)

        # Fetch enough candidates to allow reranking and diversity filtering when requested.
        candidate_k = (
            max(k, self.settings.retrieval.reranker_top_n)
            if (rerank_enabled or diversity_enabled)
            else k
        )

        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieval_results = self._retriever.search(query, top_k=candidate_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        log_event(
            _LOGGER,
            "retrieval.execute",
            message="Retriever completed",
            retriever=self._retriever.name,
            query=query,
            latency=round(retrieval_ms, 2),
            chunk_count=len(retrieval_results),
        )

        rerank_start = time.perf_counter()
        reranked_candidates = (
            self._apply_reranker(query, retrieval_results, top_n=candidate_k)
            if rerank_enabled
            else retrieval_results
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0 if rerank_enabled else 0.0
        if rerank_enabled:
            log_event(
                _LOGGER,
                "rerank.execute",
                message="Reranker completed",
                retriever=self._retriever.name,
                query=query,
                latency=round(rerank_ms, 2),
                chunk_count=len(reranked_candidates),
            )

        diversity_start = time.perf_counter()
        final_results = apply_diversity_filter(
            reranked_candidates,
            top_k=k,
            enable=diversity_enabled,
            max_chunks_per_document=self.settings.retrieval.max_chunks_per_document,
            duplicate_similarity_threshold=self.settings.retrieval.duplicate_similarity_threshold,
        )
        diversity_ms = (time.perf_counter() - diversity_start) * 1000.0

        generation_start = time.perf_counter()
        answer = generate_answer(
            query,
            final_results,
            settings=self.settings,
            max_context_chars=max_context_chars or self.settings.retrieval.chunk_size * k,
            system_instruction=system_instruction,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
        log_event(
            _LOGGER,
            "generation.complete",
            message="Generation completed",
            retriever=self._retriever.name,
            query=query,
            latency=round(generation_ms, 2),
            chunk_count=len(final_results),
        )

        elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        timings = StageTimings(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            diversity_ms=diversity_ms,
            generation_ms=generation_ms,
            total_ms=elapsed_ms,
        )
        log_event(
            _LOGGER,
            "pipeline.complete",
            message="Pipeline finished",
            retriever=self._retriever.name,
            query=query,
            latency=round(elapsed_ms, 2),
            chunk_count=len(final_results),
        )
        trace = self._build_trace(
            query=query,
            results=final_results,
            generation=answer,
            latency_ms=elapsed_ms,
            rerank_used=bool(rerank_enabled),
        )

        return PipelineResult(
            query=query,
            retriever=self._retriever.name,
            reranker=self._reranker.name if rerank_enabled and self._reranker else None,
            retrieval_results=retrieval_results,
            reranked_results=final_results,
            reranked_candidates=reranked_candidates,
            answer=answer,
            trace=trace,
            prompt=answer.prompt,
            context=answer.context,
            timings=timings,
        )

    # Internal helpers --------------------------------------------------

    def _apply_index_context(self) -> None:
        if not self._index_context:
            return
        self._retriever.set_index_context(self._index_context)
        if isinstance(self._retriever, HybridRetriever):
            self._retriever.lexical.set_index_context(self._index_context)
            self._retriever.dense.set_index_context(self._index_context)

    def _apply_index_context_to(self, retriever: BaseRetriever) -> None:
        if not self._index_context:
            return
        retriever.set_index_context(self._index_context)
        if isinstance(retriever, HybridRetriever):
            retriever.lexical.set_index_context(self._index_context)
            retriever.dense.set_index_context(self._index_context)

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
