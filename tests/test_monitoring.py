import json
from pathlib import Path

import numpy as np
import pytest

from src.config import Settings
from src.chunking import Chunk
from src.generation import GeneratedAnswer
from src.monitoring import (
    QueryTrace,
    TelemetryRecorder,
    aggregate_records,
)
from src.retrieval import RetrievalResult


def _fake_result(score: float, retriever: str = "bm25") -> RetrievalResult:
    chunk = Chunk(
        chunk_id=f"chunk_{int(score * 1000)}",
        text="some text",
        source="src",
        title="Title",
        section="sec",
        doc_id="doc_1",
        metadata={},
    )
    return RetrievalResult(chunk=chunk, score=score, rank=1, retriever=retriever)


def test_query_telemetry_to_row_rounds_values() -> None:
    telemetry = QueryTrace(
        query="What is RAG?",
        latency_ms=123.4567,
        retriever="bm25",
        top_scores=[0.9, 0.8],
        num_citations=2,
        abstained=False,
        prompt_chars=10,
        context_chars=20,
        answer_chars=30,
    )

    row = telemetry.to_row()

    assert row["latency_ms"] == 123.46  # rounded
    assert row["top_scores"] == "0.9000;0.8000"
    assert row["num_citations"] == 2
    assert row["prompt_chars"] == 10
    assert "timestamp" in row


def test_aggregate_records_computes_means_and_percentiles() -> None:
    records = [
        QueryTrace(
            query="q1",
            latency_ms=100,
            retriever="bm25",
            top_scores=[0.8],
            num_citations=1,
            abstained=False,
            prompt_chars=10,
            context_chars=15,
            answer_chars=20,
        ),
        QueryTrace(
            query="q2",
            latency_ms=300,
            retriever="bm25",
            top_scores=[0.6],
            num_citations=2,
            abstained=True,
            prompt_chars=20,
            context_chars=25,
            answer_chars=30,
        ),
    ]

    summary = aggregate_records(records)

    assert summary["count"] == 2.0
    assert summary["latency_mean_ms"] == pytest.approx(200.0)
    assert summary["latency_p50_ms"] == 200.0
    # numpy.percentile with linear interpolation on [100, 300] yields 290 for p95
    assert summary["latency_p95_ms"] == pytest.approx(290.0)
    assert summary["avg_citations"] == pytest.approx(1.5)
    assert summary["abstention_rate"] == pytest.approx(0.5)
    assert summary["avg_top1_score"] == pytest.approx(0.7)


def test_recorder_record_and_persistence(tmp_path: Path) -> None:
    settings = Settings()
    recorder = TelemetryRecorder(settings=settings, reports_dir=tmp_path)

    retrieval = [_fake_result(0.9)]
    generation = GeneratedAnswer(
        answer="Answer [chunk_900]",
        citations=["chunk_900"],
        prompt="system prompt",
        context="ctx",
        provider="fallback",
        model="deterministic",
    )

    rec = recorder.record(
        query="hello",
        latency_s=0.123,
        retrieval_results=retrieval,
        generation=generation,
    )

    assert rec.latency_ms == pytest.approx(123.0, rel=0.05)
    assert rec.num_citations == 1
    assert not rec.abstained

    csv_path = recorder.save_csv()
    jsonl_path = recorder.save_jsonl()

    assert csv_path.exists()
    assert jsonl_path.exists()

    csv_lines = csv_path.read_text().strip().splitlines()
    assert len(csv_lines) == 2  # header + one record

    json_lines = jsonl_path.read_text().strip().splitlines()
    assert len(json_lines) == 1
    loaded = json.loads(json_lines[0])
    assert loaded["query"] == "hello"


def test_plot_charts_creates_images(tmp_path: Path) -> None:
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")  # headless backend

    settings = Settings()
    recorder = TelemetryRecorder(settings=settings, reports_dir=tmp_path)

    retrieval = [_fake_result(0.75)]
    generation = GeneratedAnswer(
        answer="text [chunk_750]",
        citations=["chunk_750"],
        prompt="prompt",
        context="context",
        provider="fallback",
        model="deterministic",
    )
    recorder.record(
        query="q",
        latency_s=0.05,
        retrieval_results=retrieval,
        generation=generation,
    )

    charts = recorder.plot_charts(prefix="test")

    expected_keys = {"latency", "citations", "scores"}
    assert expected_keys.issubset(charts.keys())
    for path in charts.values():
        assert path.exists()
