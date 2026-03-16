"""Lightweight telemetry collection and visualization for RAG queries.

The goal is to capture production-style signals for every user query and make
them easy to aggregate:
* latency (ms)
* retriever mode used
* top retrieval scores
* number of citations
* abstention flag
* prompt/context size
* answer length

The module provides a simple ``TelemetryRecorder`` that stores per-query
records, writes them to ``reports/``, computes aggregates, and renders a few
basic matplotlib charts for quick portfolio-friendly observability.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Settings
from .evaluation import is_abstention
from .retrieval import RetrievalResult
from .schemas import GeneratedAnswer, QueryTrace

# Backward compatibility for previous telemetry name
QueryTelemetry = QueryTrace


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _safe_percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(values, percentile))


def aggregate_records(records: Sequence[QueryTrace]) -> Dict[str, float]:
    """Compute macro-level telemetry summaries."""

    if not records:
        return {}

    latencies = [r.latency_ms for r in records]
    citations = [r.num_citations for r in records]
    abstentions = [1.0 if r.abstained else 0.0 for r in records]
    prompt_sizes = [r.prompt_chars for r in records]
    context_sizes = [r.context_chars for r in records]
    answer_sizes = [r.answer_chars for r in records]

    top1 = [r.top_scores[0] for r in records if r.top_scores]
    top3 = [float(np.mean(r.top_scores[:3])) for r in records if r.top_scores]

    summary: Dict[str, float] = {
        "count": float(len(records)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": _safe_percentile(latencies, 50) or 0.0,
        "latency_p95_ms": _safe_percentile(latencies, 95) or 0.0,
        "avg_citations": float(np.mean(citations)),
        "abstention_rate": float(np.mean(abstentions)),
        "avg_prompt_chars": float(np.mean(prompt_sizes)),
        "avg_context_chars": float(np.mean(context_sizes)),
        "avg_answer_chars": float(np.mean(answer_sizes)),
    }

    if top1:
        summary["avg_top1_score"] = float(np.mean(top1))
    if top3:
        summary["avg_top3_score"] = float(np.mean(top3))

    return summary


# ---------------------------------------------------------------------------
# Recording + persistence
# ---------------------------------------------------------------------------


class TelemetryRecorder:
    """Collect per-query telemetry and emit reports."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        reports_dir: Optional[Path] = None,
    ) -> None:
        self.settings = settings or Settings.load()
        self.reports_dir = Path(reports_dir or self.settings.output.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[QueryTrace] = []

    def record(
        self,
        query: str,
        latency_s: float,
        retrieval_results: Sequence[RetrievalResult],
        generation: Optional[GeneratedAnswer],
        *,
        retriever_name: Optional[str] = None,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> QueryTrace:
        retriever = (
            retriever_name
            or (retrieval_results[0].retriever if retrieval_results else "unknown")
        )
        top_scores = [
            float(res.score)
            for res in retrieval_results[: self.settings.retrieval.top_k]
        ]

        prompt_text = (
            prompt if prompt is not None else (generation.prompt if generation else "")
        )
        context_text = (
            context if context is not None else (generation.context if generation else "")
        )

        record = QueryTrace(
            query=query.strip(),
            latency_ms=float(latency_s * 1000.0),
            retriever=retriever,
            top_scores=top_scores,
            num_citations=len(generation.citations) if generation else 0,
            abstained=is_abstention(generation.answer if generation else ""),
            prompt_chars=len(prompt_text),
            context_chars=len(context_text),
            answer_chars=len(generation.answer) if generation else 0,
            timestamp=timestamp or datetime.utcnow(),
        )

        self.records.append(record)
        return record

    # Persistence ---------------------------------------------------------

    def save_csv(self, filename: str = "telemetry.csv") -> Path:
        if not self.records:
            raise ValueError("No telemetry records to write.")

        path = self.reports_dir / filename
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.records[0].to_row().keys()))
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec.to_row())
        return path

    def save_jsonl(self, filename: str = "telemetry.jsonl") -> Path:
        if not self.records:
            raise ValueError("No telemetry records to write.")

        path = self.reports_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(rec.to_json() + "\n")
        return path

    # Aggregation + plotting ---------------------------------------------

    def summarize(self) -> Dict[str, float]:
        return aggregate_records(self.records)

    def plot_charts(self, prefix: str = "telemetry") -> Dict[str, Path]:
        if not self.records:
            raise ValueError("No telemetry records to plot.")

        charts = {}
        charts["latency"] = _plot_latency(self.records, self.reports_dir / f"{prefix}_latency.png")
        charts["citations"] = _plot_citations(
            self.records, self.reports_dir / f"{prefix}_citations.png"
        )
        charts["scores"] = _plot_top_scores(
            self.records, self.reports_dir / f"{prefix}_scores.png"
        )
        return charts


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_latency(records: Sequence[QueryTrace], path: Path) -> Path:
    import matplotlib.pyplot as plt

    latencies = [r.latency_ms for r in records]
    med = _safe_percentile(latencies, 50) or 0.0
    p95 = _safe_percentile(latencies, 95) or 0.0

    plt.figure(figsize=(6, 4))
    plt.hist(latencies, bins=min(20, max(5, len(latencies))), color="#4c72b0", alpha=0.85)
    plt.axvline(med, color="green", linestyle="--", label=f"p50={med:.1f} ms")
    plt.axvline(p95, color="red", linestyle=":", label=f"p95={p95:.1f} ms")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.title("Query latency distribution")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_citations(records: Sequence[QueryTrace], path: Path) -> Path:
    import matplotlib.pyplot as plt

    counts: Dict[int, int] = {}
    for rec in records:
        counts[rec.num_citations] = counts.get(rec.num_citations, 0) + 1

    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    plt.figure(figsize=(5, 4))
    plt.bar(xs, ys, color="#55a868")
    plt.xlabel("Citations per answer")
    plt.ylabel("Count")
    plt.title("Citation usage")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_top_scores(records: Sequence[QueryTrace], path: Path) -> Path:
    import matplotlib.pyplot as plt

    # Align scores by rank; pad with NaN to keep shapes consistent.
    max_k = max((len(r.top_scores) for r in records), default=0)
    if max_k == 0:
        raise ValueError("No retrieval scores available to plot.")

    data = np.full((len(records), max_k), np.nan)
    for i, rec in enumerate(records):
        data[i, : len(rec.top_scores)] = rec.top_scores

    means = np.nanmean(data, axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_k + 1), means, marker="o", color="#c44e52")
    plt.xlabel("Rank")
    plt.ylabel("Mean score")
    plt.title("Average retrieval scores by rank")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


__all__ = [
    "QueryTrace",
    "QueryTelemetry",
    "TelemetryRecorder",
    "aggregate_records",
]
