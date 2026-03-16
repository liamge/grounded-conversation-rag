"""Streamlit demo app for the Grounded Conversation RAG project.

The UI is designed for portfolio-ready screenshots:
* query box with retrieval mode + top-k controls
* grounded answer with inline citations
* expandable retrieved chunks (score + metadata)
* latency / telemetry in the sidebar
* compact architecture + live benchmark snapshot section
"""

from __future__ import annotations

import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import httpx
import streamlit as st

# Support running via `streamlit run src/app.py` (script) or as a package module.
try:  # pragma: no cover - import guard for Streamlit execution style
    from .pipeline import PipelineResult, RAGPipeline
    from .schemas import Chunk, GeneratedAnswer, QueryTrace, RetrievalResult
except ImportError:  # pragma: no cover
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.pipeline import PipelineResult, RAGPipeline  # type: ignore
    from src.schemas import Chunk, GeneratedAnswer, QueryTrace, RetrievalResult  # type: ignore


# ---------------------------------------------------------------------------
# Page styling
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Grounded Conversation RAG",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

_ACCENT = "#2563eb"
st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at 10% 20%, #f3f6ff 0, #ffffff 35%, #f7f9fc 100%);
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: #0f172a;
    }}
    .pill {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid {_ACCENT};
        background: rgba(37, 99, 235, 0.08);
        color: #0f172a;
        font-size: 12px;
        margin-right: 6px;
        margin-bottom: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Caching + helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_pipeline(config_path: Optional[str | Path] = None) -> RAGPipeline:
    """Create a single pipeline instance reused across reruns."""

    return RAGPipeline(config_path=config_path)


def _default_queries() -> List[str]:
    return [
        "How does the pipeline chunk documents before retrieval?",
        "What datasets does this repo ship with?",
        "Explain how hybrid retrieval is implemented here.",
    ]


_DISABLE_DENSE = os.getenv("RAG_DISABLE_DENSE", "").lower() in {"1", "true", "yes"}
_ALL_RETRIEVAL_LABELS: Dict[str, str] = {
    "Hybrid (lexical + dense)": "hybrid",
    "BM25 (lexical)": "bm25",
    "TF-IDF (lexical)": "tfidf",
    "Dense (MiniLM embeddings)": "dense",
}
RETRIEVAL_LABELS: Dict[str, str] = (
    {label: name for label, name in _ALL_RETRIEVAL_LABELS.items() if name in {"bm25", "tfidf"}}
    if _DISABLE_DENSE
    else _ALL_RETRIEVAL_LABELS
)
_API_URL = os.getenv("RAG_API_URL", "").strip()
_USE_REMOTE_API = bool(_API_URL)
_API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "30"))


def _format_score(score: float) -> str:
    return f"{score:.3f}"


def _append_trace(result: PipelineResult) -> None:
    history: List[Dict[str, object]] = st.session_state.setdefault("query_history", [])
    history.append(result.trace.to_row())


def _history_frame() -> pd.DataFrame:
    history: List[Dict[str, object]] = st.session_state.get("query_history", [])
    if not history:
        return pd.DataFrame(columns=["query", "latency_ms", "retriever", "num_citations"])
    return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# API helpers (used when RAG_API_URL is provided)
# ---------------------------------------------------------------------------


def _chunk_from_payload(raw: Dict[str, object]) -> Chunk:
    metadata = raw.get("metadata", {}) or {}
    return Chunk(
        chunk_id=str(raw.get("chunk_id", "")),
        text=str(raw.get("text", "")),
        source=str(raw.get("source", "")),
        title=str(raw.get("title", "")),
        section=str(raw.get("section", "")),
        doc_id=str(raw.get("doc_id", "")),
        metadata={str(k): str(v) for k, v in metadata.items()},
    )


def _retrieval_from_payload(raw: Dict[str, object]) -> RetrievalResult:
    return RetrievalResult(
        chunk=_chunk_from_payload(raw.get("chunk", {})),
        score=float(raw.get("score", 0.0)),
        rank=int(raw.get("rank", 0)),
        retriever=str(raw.get("retriever", "")),
    )


def _trace_from_payload(raw: Dict[str, object]) -> QueryTrace:
    ts_raw = raw.get("timestamp")
    timestamp = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) and ts_raw else datetime.utcnow()
    return QueryTrace(
        query=str(raw.get("query", "")),
        latency_ms=float(raw.get("latency_ms", 0.0)),
        retriever=str(raw.get("retriever", "")),
        top_scores=[float(v) for v in raw.get("top_scores", [])],
        num_citations=int(raw.get("num_citations", 0)),
        abstained=bool(raw.get("abstained", False)),
        prompt_chars=int(raw.get("prompt_chars", 0)),
        context_chars=int(raw.get("context_chars", 0)),
        answer_chars=int(raw.get("answer_chars", 0)),
        timestamp=timestamp,
    )


def _answer_from_payload(raw: Dict[str, object]) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer=str(raw.get("answer", "")),
        citations=[str(c) for c in raw.get("citations", [])],
        prompt=str(raw.get("prompt", "")),
        context=str(raw.get("context", "")),
        provider=str(raw.get("provider", "unknown")),
        model=str(raw.get("model", "unknown")),
    )


def _pipeline_result_from_api(payload: Dict[str, object]) -> PipelineResult:
    retrieval_results = [_retrieval_from_payload(rec) for rec in payload.get("retrieval_results", [])]
    reranked_results = [_retrieval_from_payload(rec) for rec in payload.get("retrieved_chunks", [])]
    trace = _trace_from_payload(payload.get("trace", {}))
    answer = _answer_from_payload(
        {
            "answer": payload.get("answer", ""),
            "citations": payload.get("citations", []),
            "prompt": payload.get("prompt", ""),
            "context": payload.get("context", ""),
            "provider": payload.get("provider", "unknown"),
            "model": payload.get("model", "unknown"),
        }
    )

    return PipelineResult(
        query=trace.query or payload.get("query", ""),
        retriever=str(payload.get("retriever", trace.retriever)),
        reranker=payload.get("reranker"),
        retrieval_results=retrieval_results,
        reranked_results=reranked_results or retrieval_results,
        answer=answer,
        trace=trace,
        prompt=answer.prompt,
        context=answer.context,
    )


def _run_query_via_api(query: str, retriever_name: str, top_k: int, use_reranker: bool) -> PipelineResult:
    if not _API_URL:
        raise RuntimeError("RAG_API_URL is not configured.")

    url = f"{_API_URL.rstrip('/')}/query"
    payload = {
        "query": query,
        "retriever_name": retriever_name,
        "top_k": top_k,
        "use_reranker": use_reranker,
    }
    try:
        response = httpx.post(url, json=payload, timeout=_API_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:  # pragma: no cover - network path
        detail = exc.response.json().get("detail") if exc.response is not None else str(exc)
        raise RuntimeError(
            f"API error ({exc.response.status_code if exc.response else 'unknown'}): {detail}"
        ) from exc
    except httpx.RequestError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"Failed to reach RAG API: {exc}") from exc

    return _pipeline_result_from_api(response.json())


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_answer(result: PipelineResult) -> None:
    st.subheader("Generated Answer")
    answer = result.answer.answer

    # Render citations as badges under the answer.
    st.markdown(answer)
    if result.answer.citations:
        st.caption("Citations (chunk ids):")
        st.markdown(" ".join(f"<span class='pill'>{cid}</span>" for cid in result.answer.citations), unsafe_allow_html=True)

    with st.expander("View prompt + context", expanded=False):
        st.code(result.prompt, language="markdown")
        st.caption(f"Context length: {len(result.context)} chars")


def render_chunks(results: Sequence[RetrievalResult], title: str = "Retrieved Chunks") -> None:
    st.subheader(title)
    if not results:
        st.info("No chunks retrieved. Try another query or retriever.")
        return

    for res in results:
        chunk = res.chunk
        header = f"#{res.rank} · {chunk.title or 'Untitled'}"
        score_str = _format_score(res.score)
        with st.expander(f"{header} — score {score_str} — {chunk.source}"):
            st.markdown(f"**Chunk ID:** `{chunk.chunk_id}` · **Retriever:** `{res.retriever}` · **Section:** {chunk.section or 'n/a'}")
            st.write(chunk.text)

            metadata = {**chunk.metadata, "doc_id": chunk.doc_id, "source": chunk.source}
            meta_rows = [{"key": k, "value": v} for k, v in metadata.items()]
            st.caption("Metadata")
            st.table(meta_rows)


def render_sidebar(result: Optional[PipelineResult]) -> None:
    st.sidebar.header("Query Metrics")
    if not result:
        st.sidebar.info("Run a query to see latency, scores, and citation counts.")
        return

    trace = result.trace
    st.sidebar.metric("Latency (ms)", f"{trace.latency_ms:.1f}")
    st.sidebar.metric("Citations", trace.num_citations)
    st.sidebar.metric("Retriever", trace.retriever)

    st.sidebar.caption("Top scores")
    if trace.top_scores:
        st.sidebar.bar_chart(trace.top_scores)
    else:
        st.sidebar.write("n/a")

    st.sidebar.caption("Lengths (chars)")
    st.sidebar.write(
        {
            "prompt": trace.prompt_chars,
            "context": trace.context_chars,
            "answer": trace.answer_chars,
        }
    )

    history_df = _history_frame()
    st.sidebar.divider()
    st.sidebar.caption("Session benchmark (live)")
    if history_df.empty:
        st.sidebar.write("No history yet.")
    else:
        st.sidebar.metric("Avg latency", f"{history_df.latency_ms.mean():.1f} ms")
        st.sidebar.metric("P95 latency", f"{history_df.latency_ms.quantile(0.95):.1f} ms")
        top_retriever = history_df.retriever.mode()[0]
        st.sidebar.caption(f"Most-used retriever: {top_retriever}")


def render_architecture_section(result: Optional[PipelineResult]) -> None:
    st.markdown("---")
    st.subheader("Architecture & Benchmarks")
    col1, col2, col3 = st.columns([1.2, 1.2, 1])

    with col1:
        st.markdown("**Pipeline**")
        st.markdown(
            "- Ingestion: txt / md / json(l) → normalized `Document`\n"
            "- Chunking: sentence-aware, deterministic IDs\n"
            "- Retrieval: BM25, TF-IDF, MiniLM dense, hybrid fusion\n"
            "- Rerank: optional keyword-overlap heuristic\n"
            "- Generation: grounded prompt with abstention guard"
        )

    with col2:
        st.markdown("**Current run settings**")
        if result:
            st.write(
                {
                    "retriever": result.retriever,
                    "reranker": result.reranker or "disabled",
                    "top_k": len(result.reranked_results),
                    "model": result.answer.model,
                }
            )
        else:
            st.info("Run your first query to populate this panel.")

    with col3:
        st.markdown("**Live benchmarks (session)**")
        df = _history_frame()
        if df.empty:
            st.write("No data yet.")
        else:
            stats = {
                "queries": len(df),
                "median_ms": f"{statistics.median(df.latency_ms):.1f}",
                "best_ms": f"{df.latency_ms.min():.1f}",
            }
            st.write(stats)
            st.caption("Computed from this browser session; great for quick screenshots.")


# ---------------------------------------------------------------------------
# Main app logic
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Grounded Conversation RAG")
    st.caption("Hybrid retrieval + grounded answering with portfolio-friendly UI.")

    if _USE_REMOTE_API:
        st.info(f"Using remote RAG API at: `{_API_URL}`")
        pipeline: Optional[RAGPipeline] = None
    else:
        pipeline = get_pipeline()
    result: Optional[PipelineResult] = None

    with st.form(key="query_form"):
        query = st.text_area(
            "Ask a question about the corpus",
            value=_default_queries()[0],
            placeholder="e.g., How does hybrid retrieval combine dense and lexical scores?",
            height=100,
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            retriever_label = st.selectbox("Retrieval mode", list(RETRIEVAL_LABELS.keys()))
        with col_b:
            top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)
        with col_c:
            use_reranker = st.toggle("Rerank (keyword overlap)", value=False, help="Lightweight heuristic reranker.")

        submitted = st.form_submit_button("Run search", type="primary", use_container_width=True)

    if submitted:
        if not query.strip():
            st.warning("Enter a question to run the pipeline.")
        else:
            with st.spinner("Retrieving + generating..."):
                try:
                    if _USE_REMOTE_API:
                        result = _run_query_via_api(
                            query=query,
                            retriever_name=RETRIEVAL_LABELS[retriever_label],
                            top_k=top_k,
                            use_reranker=use_reranker,
                        )
                    else:
                        assert pipeline is not None
                        result = pipeline.run(
                            query=query,
                            retriever_name=RETRIEVAL_LABELS[retriever_label],
                            top_k=top_k,
                            use_reranker=use_reranker,
                        )
                    _append_trace(result)
                except Exception as exc:  # pragma: no cover - UI surfacing
                    st.error(f"Pipeline error: {exc}")

    if result:
        render_answer(result)
        render_chunks(result.reranked_results, title="Retrieved Chunks (used for generation)")
    else:
        st.info("Run a query to see grounded answers and retrieved evidence.")

    render_architecture_section(result)
    render_sidebar(result)


if __name__ == "__main__":
    main()
