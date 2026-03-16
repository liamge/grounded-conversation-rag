"""Retrieval components for the Grounded Conversation RAG.

This module defines a small, benchmarking-friendly retrieval stack:

* ``BaseRetriever``: common interface for all retrievers
* ``TfidfRetriever``: lightweight lexical baseline
* ``BM25Retriever``: stronger lexical scorer using Okapi BM25
* ``DenseRetriever``: embedding similarity via sentence-transformers
* ``HybridRetriever``: fusion of lexical + dense scores

Each retriever returns ``RetrievalResult`` objects that carry the score,
rank, retriever name, and the full ``Chunk`` metadata so downstream steps
can assemble citations or debug retrieval quality.
"""

from __future__ import annotations

import os
from pathlib import Path
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from .config import ModelConfig, RetrievalConfig, Settings
from .schemas import Chunk, RetrievalResult

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

try:  # Optional FAISS for persistent vector store
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:  # pragma: no cover - optional dep
    _HAS_FAISS = False


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseRetriever:
    """Abstract retriever interface.

    Subclasses implement ``index`` and ``search``; they should set a
    human-readable ``name`` attribute for logging/benchmarking.
    """

    name: str = "base"

    def __init__(self, top_k: int = 5, min_score: float = 0.0) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self._indexed = False

    def index(self, chunks: Sequence[Chunk]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(  # pragma: no cover
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        raise NotImplementedError

    # Utility used by subclasses
    def _take_top_k(self, scores: np.ndarray, top_k: Optional[int]) -> np.ndarray:
        k = top_k or self.top_k
        # Clamp to available results to avoid out-of-bounds when corpus < k.
        k = min(max(k, 0), scores.shape[0])
        if k == 0:
            return np.array([], dtype=int)
        # Stable descending sort; ties keep original corpus order (important for deterministic tests).
        order = np.argsort(-scores, kind="stable")
        return order[:k]

    def _ensure_indexed(self) -> None:
        if not self._indexed:
            raise RuntimeError("Call index(chunks) before search().")


# ---------------------------------------------------------------------------
# Lexical retrievers
# ---------------------------------------------------------------------------


class TfidfRetriever(BaseRetriever):
    """Simple TF-IDF cosine similarity retriever."""

    name = "tfidf"

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.0,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        super().__init__(top_k=top_k, min_score=min_score)
        self.ngram_range = ngram_range
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._chunks: List[Chunk] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=self.ngram_range)
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        self._indexed = True

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        self._ensure_indexed()
        assert self._vectorizer is not None and self._tfidf_matrix is not None

        query_vec = self._vectorizer.transform([query])
        # Matrix product returns a sparse vector; densify for ranking.
        scores = (self._tfidf_matrix @ query_vec.T).toarray().ravel()
        top_indices = self._take_top_k(scores, top_k)

        results: List[RetrievalResult] = []
        for rank_idx, chunk_idx in enumerate(top_indices, start=1):
            score = float(scores[chunk_idx])
            if score < self.min_score:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[chunk_idx],
                    score=score,
                    rank=rank_idx,
                    retriever=self.name,
                )
            )
        return results


class BM25Retriever(BaseRetriever):
    """Okapi BM25 retriever (lexical)."""

    name = "bm25"

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = float("-inf"),
        k1: float = 1.6,
        b: float = 0.75,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        super().__init__(top_k=top_k, min_score=min_score)
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or self._default_tokenizer
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: List[Chunk] = []
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        import re

        return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if t]

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized_corpus = [self.tokenizer(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        self._indexed = True

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        self._ensure_indexed()
        assert self._bm25 is not None

        query_tokens = self.tokenizer(query)
        scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=float)
        top_indices = self._take_top_k(scores, top_k)

        results: List[RetrievalResult] = []
        for rank_idx, chunk_idx in enumerate(top_indices, start=1):
            score = float(scores[chunk_idx])
            if score < self.min_score:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[chunk_idx],
                    score=score,
                    rank=rank_idx,
                    retriever=self.name,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------


class DenseRetriever(BaseRetriever):
    """Embedding-based retriever using sentence-transformers."""

    name = "dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        min_score: float = 0.0,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
        model: Optional["SentenceTransformer"] = None,
        embeddings_dir: Optional[Path] = None,
        use_faiss: bool = True,
        prefer_cache: bool = True,
        force_recompute: bool = False,
    ) -> None:
        super().__init__(top_k=top_k, min_score=min_score)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self._model: Optional[SentenceTransformer] = model
        self._embeddings: Optional[np.ndarray] = None
        self._chunks: List[Chunk] = []
        self.embeddings_dir = embeddings_dir
        self.use_faiss = use_faiss and _HAS_FAISS
        self.prefer_cache = prefer_cache
        self.force_recompute = force_recompute
        self._faiss_index: Optional["faiss.Index"] = None

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "sentence-transformers is required for DenseRetriever. "
                    "Install with `pip install sentence-transformers`."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    # -------------------------- caching helpers ---------------------------

    def _embedding_prefix(self) -> Optional[Path]:
        if not self.embeddings_dir:
            return None
        safe_model = self.model_name.replace("/", "_")
        return Path(self.embeddings_dir) / safe_model

    def _load_cached_embeddings(self, chunks: Sequence[Chunk]) -> bool:
        prefix = self._embedding_prefix()
        if not prefix:
            return False

        emb_path = prefix.with_suffix(".npy")
        meta_path = prefix.with_suffix(".meta.jsonl")
        faiss_path = prefix.with_suffix(".faiss")

        if not emb_path.exists() or not meta_path.exists():
            return False

        chunk_ids = [c.chunk_id for c in chunks]
        saved_ids: List[str] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                saved_ids.append(str(obj.get("chunk_id")))

        if saved_ids != chunk_ids:
            return False

        embeddings = np.load(emb_path)
        if embeddings.shape[0] != len(chunks):
            return False

        self._embeddings = embeddings.astype(np.float32)
        self._chunks = list(chunks)
        if self.use_faiss and faiss_path.exists():
            try:
                self._faiss_index = faiss.read_index(str(faiss_path))
            except Exception:
                self._faiss_index = None
        self._indexed = True
        return True

    def _save_embeddings(self) -> None:
        prefix = self._embedding_prefix()
        if not prefix or self._embeddings is None:
            return

        prefix.parent.mkdir(parents=True, exist_ok=True)
        emb_path = prefix.with_suffix(".npy")
        meta_path = prefix.with_suffix(".meta.jsonl")
        faiss_path = prefix.with_suffix(".faiss")

        np.save(emb_path, self._embeddings)
        with meta_path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id}) + "\n")

        if self.use_faiss:
            dim = self._embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self._embeddings)
            faiss.write_index(index, str(faiss_path))
            self._faiss_index = index

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)

        if self.prefer_cache and not self.force_recompute:
            if self._load_cached_embeddings(chunks):
                return

        model = self._get_model()
        embeddings = model.encode(
            [c.text for c in self._chunks],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        self._embeddings = embeddings.astype(np.float32)
        if self.normalize_embeddings and self._embeddings is not None:
            self._embeddings = l2_normalize(self._embeddings)

        self._indexed = True
        self._save_embeddings()

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        self._ensure_indexed()
        assert self._embeddings is not None
        model = self._get_model()

        query_vec = model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        if self.normalize_embeddings:
            query_vec = l2_normalize(query_vec.reshape(1, -1)).ravel()

        k = top_k or self.top_k
        k = min(max(k, 0), len(self._chunks))
        if k == 0:
            return []

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(np.array([query_vec]).astype(np.float32), k)
            scores = scores.ravel()
            top_indices = indices.ravel()
        else:
            scores = self._embeddings @ query_vec
            top_indices = self._take_top_k(scores, top_k)

        results: List[RetrievalResult] = []
        rank_idx = 1
        for idx_pos, chunk_idx in enumerate(top_indices):
            if chunk_idx < 0 or chunk_idx >= len(self._chunks):
                continue  # skip FAISS padding or bad indices defensively
            score = float(scores[idx_pos] if idx_pos < len(scores) else scores[chunk_idx])
            if score < self.min_score:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[chunk_idx],
                    score=score,
                    rank=rank_idx,
                    retriever=self.name,
                )
            )
            rank_idx += 1
            if rank_idx > k:
                break
        return results


# ---------------------------------------------------------------------------
# Hybrid fusion retriever
# ---------------------------------------------------------------------------


def _normalize_result_scores(results: Sequence[RetrievalResult]) -> Dict[str, float]:
    if not results:
        return {}
    scores = np.array([r.score for r in results], dtype=float)
    min_s, max_s = scores.min(), scores.max()
    if np.isclose(max_s, min_s):
        return {r.chunk.chunk_id: 1.0 for r in results}
    return {r.chunk.chunk_id: float((r.score - min_s) / (max_s - min_s)) for r in results}


class HybridRetriever(BaseRetriever):
    """Weighted fusion of a lexical retriever and a dense retriever."""

    name = "hybrid"

    def __init__(
        self,
        lexical: BaseRetriever,
        dense: BaseRetriever,
        weight_lexical: float = 0.5,
        weight_dense: float = 0.5,
        top_k: int = 5,
    ) -> None:
        super().__init__(top_k=top_k, min_score=0.0)
        self.lexical = lexical
        self.dense = dense
        self.weight_lexical = weight_lexical
        self.weight_dense = weight_dense

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.lexical.index(chunks)
        self.dense.index(chunks)
        self._indexed = True

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        self._ensure_indexed()

        candidate_k = max(self.top_k, top_k or self.top_k) * 2
        lexical_results = self.lexical.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)

        lex_norm = _normalize_result_scores(lexical_results)
        dense_norm = _normalize_result_scores(dense_results)

        combined_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Chunk] = {}

        for res in lexical_results:
            chunk_id = res.chunk.chunk_id
            chunk_lookup[chunk_id] = res.chunk
            combined_scores[chunk_id] = (
                combined_scores.get(chunk_id, 0.0)
                + self.weight_lexical * lex_norm.get(chunk_id, 0.0)
            )

        for res in dense_results:
            chunk_id = res.chunk.chunk_id
            chunk_lookup[chunk_id] = res.chunk
            combined_scores[chunk_id] = (
                combined_scores.get(chunk_id, 0.0)
                + self.weight_dense * dense_norm.get(chunk_id, 0.0)
            )

        if not combined_scores:
            return []

        sorted_items = sorted(combined_scores.items(), key=lambda kv: kv[1], reverse=True)
        k = top_k or self.top_k
        top_items = sorted_items[:k]

        results: List[RetrievalResult] = []
        for rank_idx, (chunk_id, score) in enumerate(top_items, start=1):
            chunk = chunk_lookup[chunk_id]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank_idx,
                    retriever=self.name,
                )
            )
        return results


def build_retriever_from_config(
    settings: Settings, retriever_name: Optional[str] = None
) -> BaseRetriever:
    """Factory to build a retriever based on Settings.

    Parameters
    ----------
    settings:
        Full application settings object.
    retriever_name:
        Overrides ``settings.app.default_retriever`` when provided. Accepted values:
        ``tfidf``, ``bm25``, ``dense``, ``hybrid``.
    """

    r_cfg: RetrievalConfig = settings.retrieval
    m_cfg: ModelConfig = settings.models
    name = (retriever_name or settings.app.default_retriever).lower()
    disable_dense = os.getenv("RAG_DISABLE_DENSE", "").lower() in {"1", "true", "yes"}

    if disable_dense and name in {"dense", "hybrid"}:
        name = "bm25"

    lexical = BM25Retriever(top_k=r_cfg.top_k, min_score=r_cfg.min_score, k1=r_cfg.bm25_k1, b=r_cfg.bm25_b)

    if name == "tfidf":
        return TfidfRetriever(top_k=r_cfg.top_k, min_score=r_cfg.min_score)
    if name == "bm25":
        return lexical
    if name == "dense":
        return DenseRetriever(
            model_name=m_cfg.embedding_model,
            top_k=r_cfg.top_k,
            min_score=r_cfg.min_score,
            embeddings_dir=settings.data.embeddings_dir,
        )
    if name == "hybrid":
        dense = DenseRetriever(
            model_name=m_cfg.embedding_model,
            top_k=r_cfg.top_k,
            min_score=r_cfg.min_score,
            embeddings_dir=settings.data.embeddings_dir,
        )
        return HybridRetriever(
            lexical=lexical,
            dense=dense,
            weight_lexical=r_cfg.hybrid_weight_lexical,
            weight_dense=r_cfg.hybrid_weight_dense,
            top_k=r_cfg.top_k,
        )

    raise ValueError(f"Unknown retriever name: {name}")


__all__ = [
    "RetrievalResult",
    "BaseRetriever",
    "TfidfRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "build_retriever_from_config",
]
