"""Streamlit demo app for the Grounded Conversation RAG project.

The UI is designed for portfolio-ready screenshots:
* query box with retrieval mode + top-k controls
* grounded answer with inline citations
* expandable retrieved chunks (score + metadata)
* latency / telemetry in the sidebar
* compact architecture + live benchmark snapshot section
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

# Support running via `streamlit run src/app.py` (script) or as a package module.
try:  # pragma: no cover - import guard for Streamlit execution style
    from .config import Settings, is_truthy_env
    from .pipeline import PipelineResult, RAGPipeline
    from .schemas import Chunk, GeneratedAnswer, QueryTrace, RetrievalResult, StageTimings
except ImportError:  # pragma: no cover
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.config import Settings, is_truthy_env  # type: ignore
    from src.pipeline import PipelineResult, RAGPipeline  # type: ignore
    from src.schemas import Chunk, GeneratedAnswer, QueryTrace, RetrievalResult, StageTimings  # type: ignore


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


@st.cache_resource(show_spinner=False)
def get_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Load Settings once for the session (used for locating artifacts)."""

    return Settings.load(path=config_path)


@st.cache_data(show_spinner=False)
def load_eval_runs(
    reports_dir: Path,
) -> list[tuple[str, pd.DataFrame, Dict[str, float]]]:
    """Load all eval runs found in reports_dir, keyed by prefix."""

    runs: list[tuple[str, pd.DataFrame, Dict[str, float]]] = []
    for summary_path in sorted(reports_dir.glob("*_summary.json")):
        prefix = summary_path.stem.replace("_summary", "")
        results_path = summary_path.with_name(f"{prefix}_results.csv")
        if not results_path.exists():
            continue
        try:
            summary: Dict[str, float] = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        try:
            df = pd.read_csv(results_path)
        except Exception:
            continue
        runs.append((prefix, df, summary))
    return runs


def _default_queries() -> List[str]:
    return [
        "How does the pipeline chunk documents before retrieval?",
        "What datasets does this repo ship with?",
        "Explain how hybrid retrieval is implemented here.",
    ]


_DEMO_MODE = is_truthy_env("RAG_DEMO_MODE")


def _dense_disabled() -> bool:
    return _DEMO_MODE or is_truthy_env("RAG_DISABLE_DENSE")
_ALL_RETRIEVAL_LABELS: Dict[str, str] = {
    "Hybrid (lexical + dense)": "hybrid",
    "BM25 (lexical)": "bm25",
    "TF-IDF (lexical)": "tfidf",
    "Dense (MiniLM embeddings)": "dense",
}


def _retrieval_labels() -> Dict[str, str]:
    if _dense_disabled():
        return {label: name for label, name in _ALL_RETRIEVAL_LABELS.items() if name in {"bm25", "tfidf"}}
    return _ALL_RETRIEVAL_LABELS
_API_URL = os.getenv("RAG_API_URL", "").strip()
_USE_REMOTE_API = bool(_API_URL)
_API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "30"))


def _format_score(score: float) -> str:
    return f"{score:.3f}"


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _append_trace(result: PipelineResult) -> None:
    history: List[Dict[str, object]] = st.session_state.setdefault("query_history", [])
    history.append(result.trace.to_row())
    # Cache the most recent full result for the Diagnostics tab.
    st.session_state["last_result"] = result


def _history_frame() -> pd.DataFrame:
    history: List[Dict[str, object]] = st.session_state.get("query_history", [])
    if not history:
        return pd.DataFrame(columns=["query", "latency_ms", "retriever", "num_citations"])
    return pd.DataFrame(history)


def _mode_badges(settings: Settings) -> List[str]:
    badges: List[str] = []
    provider = (settings.models.llm_provider or "fallback").lower()
    api_key = bool(os.getenv("OPENAI_API_KEY"))
    demo_flag = settings.demo_mode or _DEMO_MODE
    prefer_openai = api_key and not demo_flag and provider in {"openai", "fallback", "demo", "extractive"}

    if demo_flag:
        badges.append("Demo mode (lightweight)")

    badges.append("Lexical retrieval" if _dense_disabled() else "Hybrid retrieval")

    if prefer_openai:
        badges.append("OpenAI generator")
    elif provider in {"local", "transformers", "hf"}:
        badges.append("Local tiny model (opt-in)")
    else:
        badges.append("Lightweight summarizer")

    if not prefer_openai:
        badges.append("No external API required")

    return badges


def render_mode_banner(settings: Settings) -> None:
    badges = _mode_badges(settings)
    if not badges:
        return

    badge_html = " ".join(f"<span class='pill'>{label}</span>" for label in badges)
    st.markdown(
        f"""
        <div style='margin-top:6px; margin-bottom:8px;'>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _timings_from_payload(raw: Dict[str, object]) -> Optional[StageTimings]:
    if not isinstance(raw, dict):
        return None
    return StageTimings(
        retrieval_ms=float(raw.get("retrieval_ms", 0.0)),
        rerank_ms=float(raw.get("rerank_ms", 0.0)),
        diversity_ms=float(raw.get("diversity_ms", 0.0)),
        generation_ms=float(raw.get("generation_ms", 0.0)),
        total_ms=float(raw.get("total_ms", raw.get("latency_ms", 0.0))),
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
    evidence_raw = raw.get("evidence_chunks", []) or []
    evidence_chunks = [_chunk_from_payload(chunk_raw) for chunk_raw in evidence_raw]
    return GeneratedAnswer(
        answer=str(raw.get("answer", "")),
        citations=[str(c) for c in raw.get("citations", [])],
        evidence_chunks=evidence_chunks,
        supported=bool(raw.get("supported", bool(raw.get("citations")))),
        prompt=str(raw.get("prompt", "")),
        context=str(raw.get("context", "")),
        provider=str(raw.get("provider", "unknown")),
        model=str(raw.get("model", "unknown")),
    )


def _pipeline_result_from_api(payload: Dict[str, object]) -> PipelineResult:
    retrieval_results = [_retrieval_from_payload(rec) for rec in payload.get("retrieval_results", [])]
    reranked_candidates = [_retrieval_from_payload(rec) for rec in payload.get("reranked_candidates", [])]
    reranked_results = [_retrieval_from_payload(rec) for rec in payload.get("retrieved_chunks", [])]
    trace = _trace_from_payload(payload.get("trace", {}))
    answer = _answer_from_payload(
        {
            "answer": payload.get("answer", ""),
            "citations": payload.get("citations", []),
            "evidence_chunks": payload.get("evidence_chunks", []),
            "supported": payload.get("supported", False),
            "prompt": payload.get("prompt", ""),
            "context": payload.get("context", ""),
            "provider": payload.get("provider", "unknown"),
            "model": payload.get("model", "unknown"),
        }
    )

    timings = _timings_from_payload(payload.get("timings"))

    return PipelineResult(
        query=trace.query or payload.get("query", ""),
        retriever=str(payload.get("retriever", trace.retriever)),
        reranker=payload.get("reranker"),
        retrieval_results=retrieval_results,
        reranked_candidates=reranked_candidates or reranked_results or retrieval_results,
        reranked_results=reranked_results or reranked_candidates or retrieval_results,
        answer=answer,
        trace=trace,
        prompt=answer.prompt,
        context=answer.context,
        timings=timings,
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
        st.markdown(
            " ".join(f"<a class='pill' href='#{cid}'>{cid}</a>" for cid in result.answer.citations),
            unsafe_allow_html=True,
        )
    if not result.answer.supported:
        st.warning("Answer is unsupported because no valid citations were found in the provided context.")

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
        st.markdown(f"<div id='{chunk.chunk_id}'></div>", unsafe_allow_html=True)
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


def render_diagnostics_tab(pipeline: Optional[RAGPipeline], result: Optional[PipelineResult]) -> None:
    """Deep-dive diagnostics for retrieval quality and latency."""

    st.subheader("Retrieval Diagnostics")
    if result is None:
        result = st.session_state.get("last_result")
    if not result:
        st.info("Run a query on the Search tab to populate diagnostics.")
        return

    compare_toggle = st.toggle(
        "Compare retrievers",
        value=False,
        help="Show session-level latency and usage stats per retriever.",
        key="diag_compare_toggle",
    )
    diversity_toggle = st.toggle(
        "Inspect chunk diversity filtering",
        value=True,
        help="Highlight chunks removed after rerank/diversity limits.",
        key="diag_diversity_toggle",
    )
    context_toggle = st.toggle(
        "Inspect context assembly output",
        value=False,
        help="View the concatenated context sent to the generator.",
        key="diag_context_toggle",
    )

    st.markdown("### Latency breakdown")
    timings = result.timings
    if timings:
        latency_rows = [
            {"stage": "retrieval", "ms": timings.retrieval_ms},
            {"stage": "rerank", "ms": timings.rerank_ms},
            {"stage": "diversity", "ms": timings.diversity_ms},
            {"stage": "generation", "ms": timings.generation_ms},
            {"stage": "total", "ms": timings.total_ms},
        ]
    else:
        latency_rows = [{"stage": "total", "ms": result.trace.latency_ms}]
        st.caption("Stage-level timing not available for this run.")
    latency_df = pd.DataFrame(latency_rows)
    st.dataframe(latency_df, hide_index=True, use_container_width=True)
    st.bar_chart(latency_df.set_index("stage"), use_container_width=True)

    st.markdown("### Retrieval + reranker scores")
    raw_map = {res.chunk.chunk_id: res for res in result.retrieval_results}
    rerank_list = result.reranked_candidates or result.reranked_results
    final_ids = {res.chunk.chunk_id for res in result.reranked_results}

    score_rows: List[Dict[str, object]] = []
    for res in rerank_list:
        chunk_id = res.chunk.chunk_id
        raw_res = raw_map.get(chunk_id)
        score_rows.append(
            {
                "chunk_id": chunk_id,
                "source": res.chunk.source,
                "doc_id": res.chunk.doc_id,
                "title": res.chunk.title,
                "preview": (res.chunk.text[:180] + "...") if len(res.chunk.text) > 180 else res.chunk.text,
                "raw_rank": raw_res.rank if raw_res else None,
                "raw_score": raw_res.score if raw_res else None,
                "rerank_rank": res.rank,
                "rerank_score": res.score,
                "final_rank": next((f.rank for f in result.reranked_results if f.chunk.chunk_id == chunk_id), None),
                "kept_after_diversity": chunk_id in final_ids,
            }
        )
    score_df = pd.DataFrame(score_rows)
    if not score_df.empty:
        st.dataframe(score_df, hide_index=True, use_container_width=True)
    else:
        st.info("No retrieval rows to display.")

    st.markdown("### Chunk sources (final context)")
    if result.reranked_results:
        source_rows = [
            {
                "rank": res.rank,
                "chunk_id": res.chunk.chunk_id,
                "source": res.chunk.source,
                "doc_id": res.chunk.doc_id,
                "preview": (res.chunk.text[:120] + "...") if len(res.chunk.text) > 120 else res.chunk.text,
                "score": res.score,
            }
            for res in result.reranked_results
        ]
        st.dataframe(source_rows, hide_index=True, use_container_width=True)
        source_df = pd.DataFrame(source_rows)
        grouped = source_df.groupby("source").agg(chunks=("chunk_id", "count"), avg_score=("score", "mean"))
        grouped = grouped.reset_index().sort_values(by="chunks", ascending=False)
        st.bar_chart(grouped.set_index("source"), use_container_width=True)
    else:
        st.info("No chunks were kept after retrieval.")

    if diversity_toggle:
        st.markdown("### Diversity filter impact")
        removed = [res for res in rerank_list if res.chunk.chunk_id not in final_ids]
        st.caption(
            f"Removed {len(removed)} of {len(rerank_list)} reranked candidates "
            f"(final top-k: {len(result.reranked_results)})."
        )
        if removed:
            removed_rows = [
                {
                    "chunk_id": res.chunk.chunk_id,
                    "source": res.chunk.source,
                    "doc_id": res.chunk.doc_id,
                    "rerank_rank": res.rank,
                    "rerank_score": res.score,
                }
                for res in removed
            ]
            st.dataframe(removed_rows, hide_index=True, use_container_width=True)
        else:
            st.info("Diversity filter did not remove any chunks for this run.")

    if compare_toggle:
        st.markdown("### Retriever comparison (session)")
        history_df = _history_frame()
        if history_df.empty:
            st.info("Run a few queries to build comparison data.")
        else:
            grouped = (
                history_df.groupby("retriever")
                .agg(
                    queries=("query", "count"),
                    avg_latency_ms=("latency_ms", "mean"),
                    p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
                )
                .reset_index()
                .sort_values(by="queries", ascending=False)
            )
            st.dataframe(grouped, hide_index=True, use_container_width=True)
            fig = px.bar(grouped, x="retriever", y="avg_latency_ms", title="Avg latency by retriever")
            st.plotly_chart(fig, use_container_width=True)

    if context_toggle:
        st.markdown("### Context assembly output")
        st.caption(f"Context length: {len(result.context)} characters")
        st.text_area("Context sent to generator", result.context, height=260)


# ---------------------------------------------------------------------------
# Main app logic
# ---------------------------------------------------------------------------


def render_search_tab(pipeline: Optional[RAGPipeline]) -> Optional[PipelineResult]:
    """Main retrieval + generation experience (kept on its own tab)."""

    result: Optional[PipelineResult] = st.session_state.get("last_result")

    with st.form(key="query_form"):
        query = st.text_area(
            "Ask a question about the corpus",
            value=_default_queries()[0],
            placeholder="e.g., How does hybrid retrieval combine dense and lexical scores?",
            height=100,
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        retrieval_labels = _retrieval_labels()
        with col_a:
            retriever_label = st.selectbox("Retrieval mode", list(retrieval_labels.keys()))
        with col_b:
            top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)
        with col_c:
            use_reranker = st.toggle(
                "Rerank (keyword overlap)", value=False, help="Lightweight heuristic reranker."
            )

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
                            retriever_name=retrieval_labels[retriever_label],
                            top_k=top_k,
                            use_reranker=use_reranker,
                        )
                    else:
                        assert pipeline is not None
                        result = pipeline.run(
                            query=query,
                            retriever_name=retrieval_labels[retriever_label],
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
    return result


def render_evaluation_tab(pipeline: Optional[RAGPipeline]) -> None:
    """Visualization of offline evaluation artifacts."""

    st.subheader("Evaluation")
    settings: Optional[Settings] = None
    try:
        settings = pipeline.settings if pipeline else get_settings()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Unable to load settings to locate reports: {exc}")
        return

    reports_dir = Path(settings.output.reports_dir)
    runs = load_eval_runs(reports_dir)
    st.caption(f"Artifacts directory: `{reports_dir}`")

    if not runs:
        st.info("Evaluation artifacts not found. Generate them via the CLI:")
        st.code("python -m src.cli eval --report-prefix eval", language="bash")
        st.caption("Artifacts write to <prefix>_summary.json and <prefix>_results.csv under reports/.")
        return

    st.markdown("### Aggregate metrics (per run)")
    agg_rows: List[Dict[str, object]] = []
    retrieval_rows: List[Dict[str, object]] = []
    for prefix, df, summary in runs:
        label = prefix.replace("eval", "").lstrip("_") or "default"
        agg_rows.append(
            {
                "run": label,
                "mrr": summary.get("mrr"),
                "citation_coverage": summary.get("citation_coverage"),
                "abstention_accuracy": summary.get("abstention_accuracy"),
            }
        )
        for key, value in summary.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("recall@") or key.startswith("precision@"):
                try:
                    k_val = int(key.split("@", 1)[1])
                except ValueError:
                    continue
                metric_name = "Recall@k" if key.startswith("recall@") else "Precision@k"
                retrieval_rows.append({"run": label, "k": k_val, "score": value, "metric": metric_name})

    agg_df = pd.DataFrame(agg_rows)
    if not agg_df.empty:
        agg_fig = px.bar(
            agg_df.melt(id_vars=["run"], var_name="metric", value_name="score"),
            x="run",
            y="score",
            color="metric",
            barmode="group",
            title="Answer quality per run",
        )
        agg_fig.update_yaxes(range=[0, 1], tickformat=".2f")
        st.plotly_chart(agg_fig, use_container_width=True)
        st.dataframe(agg_df, hide_index=True, use_container_width=True)

    retrieval_df = pd.DataFrame(retrieval_rows)
    if not retrieval_df.empty:
        retr_fig = px.bar(
            retrieval_df,
            x="k",
            y="score",
            color="run",
            facet_col="metric",
            barmode="group",
            title="Retrieval quality per run",
        )
        retr_fig.update_yaxes(range=[0, 1], tickformat=".2f")
        st.plotly_chart(retr_fig, use_container_width=True)

    st.markdown("### Per-example results")
    merged_frames: List[pd.DataFrame] = []
    for prefix, df, _ in runs:
        label = prefix.replace("eval", "").lstrip("_") or "default"
        df = df.copy()
        df["run"] = label
        merged_frames.append(df)
    if merged_frames:
        combined = pd.concat(merged_frames, ignore_index=True)
        focus_cols = [
            "run",
            "example_id",
            "query",
            "retriever",
            "mrr",
            "citation_coverage",
            "abstention_correct",
            "num_citations",
        ]
        available_cols = [col for col in focus_cols if col in combined.columns]
        display_df = combined[available_cols] if available_cols else combined
        st.dataframe(display_df, use_container_width=True, height=360)


def main() -> None:
    st.title("Grounded Conversation RAG")
    st.caption("Hybrid retrieval + grounded answering with portfolio-friendly UI.")

    settings = get_settings()
    render_mode_banner(settings)

    if _USE_REMOTE_API:
        st.info(f"Using remote RAG API at: `{_API_URL}`")
        pipeline: Optional[RAGPipeline] = None
    else:
        pipeline = get_pipeline()

    result: Optional[PipelineResult] = None
    search_tab, diag_tab, eval_tab = st.tabs(["Search", "Retrieval Diagnostics", "Evaluation"])

    with search_tab:
        result = render_search_tab(pipeline)
    with diag_tab:
        render_diagnostics_tab(pipeline, result)
    with eval_tab:
        render_evaluation_tab(pipeline)

    render_sidebar(result)


if __name__ == "__main__":
    main()
