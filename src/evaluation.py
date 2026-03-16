"""Evaluation utilities for retrieval and grounded answers.

The module provides:
* classic retrieval metrics (Recall@k, MRR, Precision@k)
* lightweight answer quality checks (citation coverage, abstention accuracy,
  evidence overlap against required facts)
* a benchmark runner that loads a JSONL eval set, runs retrieval/generation,
  and writes per‑example + aggregate reports under ``reports/``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz

from .chunking import chunk_document
from .config import Settings
from .generation import generate_answer
from .ingestion import ingest_documents
from .retrieval import BaseRetriever, build_retriever_from_config
from .schemas import Chunk, Document, EvalExample, GeneratedAnswer, RetrievalResult

# ---------------------------------------------------------------------------
# Core metric primitives
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Recall@k over an ordered list of retrieved ids."""

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for rid in top if rid in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Precision@k over an ordered list of retrieved ids."""

    if k <= 0:
        return 0.0
    relevant = set(relevant_ids)
    top = retrieved_ids[:k]
    hits = sum(1 for rid in top if rid in relevant)
    return hits / max(k, 1)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Mean reciprocal rank for a single query."""

    relevant = set(relevant_ids)
    for idx, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / idx
    return 0.0


def mrr(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Alias for ``mean_reciprocal_rank`` to match common shorthand."""

    return mean_reciprocal_rank(retrieved_ids, relevant_ids)


# ---------------------------------------------------------------------------
# Answer quality heuristics
# ---------------------------------------------------------------------------


ABSTENTION_PHRASE = "I don't have enough evidence to answer confidently."


def is_abstention(answer_text: str, phrase: str = ABSTENTION_PHRASE) -> bool:
    """Return True when the answer matches the canonical abstention phrase."""

    normalized = answer_text.strip().lower()
    return normalized == phrase.strip().lower()


def citation_coverage(
    found_citations: Sequence[str], expected_citations: Iterable[str]
) -> Optional[float]:
    """Portion of expected citations that were referenced.

    Returns ``None`` when no expected citations are provided.
    """

    expected = set(expected_citations)
    if not expected:
        return None
    if not found_citations:
        return 0.0
    hits = len(set(found_citations) & expected)
    return hits / len(expected)


def evidence_overlap(
    answer_text: str, expected_facts: Sequence[str], threshold: int = 80
) -> Optional[float]:
    """Heuristic fraction of required facts present in the answer.

    Uses RapidFuzz partial ratio to allow minor wording changes.
    Returns ``None`` when no expected facts are supplied.
    """

    if not expected_facts:
        return None

    text = answer_text.lower()
    scores: List[int] = []
    for fact in expected_facts:
        fact_clean = fact.lower()
        if fact_clean in text:
            scores.append(100)
            continue
        scores.append(fuzz.partial_ratio(fact_clean, text))

    hits = sum(1 for s in scores if s >= threshold)
    return hits / len(expected_facts)


def abstention_correct(answer_text: str, should_abstain: Optional[bool]) -> Optional[bool]:
    """Compare model abstention against expectation.

    Returns ``None`` when expectation is unspecified.
    """

    if should_abstain is None:
        return None
    return is_abstention(answer_text) == should_abstain


@dataclass(slots=True)
class ExampleMetrics:
    """Per-example metrics spanning retrieval and answer quality."""

    example_id: str
    query: str
    retriever: str
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    citation_coverage: Optional[float]
    evidence_overlap: Optional[float]
    abstained: bool
    abstention_correct: Optional[bool]
    num_citations: int
    top_retrieved_ids: List[str]

    def to_flat_dict(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "example_id": self.example_id,
            "query": self.query,
            "retriever": self.retriever,
            "mrr": round(self.mrr, 4),
            "citation_coverage": None
            if self.citation_coverage is None
            else round(self.citation_coverage, 4),
            "evidence_overlap": None
            if self.evidence_overlap is None
            else round(self.evidence_overlap, 4),
            "abstained": self.abstained,
            "abstention_correct": self.abstention_correct,
            "num_citations": self.num_citations,
            "top_retrieved_ids": ";".join(self.top_retrieved_ids),
        }
        for k, v in self.recall_at_k.items():
            row[f"recall@{k}"] = round(v, 4)
        for k, v in self.precision_at_k.items():
            row[f"precision@{k}"] = round(v, 4)
        return row


# ---------------------------------------------------------------------------
# Helpers for dataset + corpus loading
# ---------------------------------------------------------------------------


def _load_eval_set(path: Path) -> List[EvalExample]:
    if not path.exists():
        raise FileNotFoundError(f"Eval set not found at {path}")

    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(EvalExample.from_record(record, idx))
    return examples


def _load_corpus(raw_dir: Path, settings: Settings) -> List[Chunk]:
    """Ingest and chunk all supported documents under ``raw_dir`` or at a file path."""

    raw_path = Path(raw_dir)
    search_paths: List[Path] = []
    if raw_path.is_file():
        search_paths = [raw_path]
    elif raw_path.is_dir():
        files = list(raw_path.rglob("*"))
        if any(p.is_file() for p in files):
            search_paths = [raw_path]
    if not search_paths:
        raise FileNotFoundError(
            f"No source documents found under {raw_dir}. "
            "Add files to data/raw/ or point settings.data.raw_dir elsewhere."
        )

    documents: List[Document] = ingest_documents(search_paths)
    chunks: List[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                doc,
                chunk_size=settings.retrieval.chunk_size,
                chunk_overlap=settings.retrieval.chunk_overlap,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Per-example scoring
# ---------------------------------------------------------------------------


def _select_ids(results: Sequence[RetrievalResult], use_chunks: bool) -> List[str]:
    ids: List[str] = []
    for res in results:
        ids.append(res.chunk.chunk_id if use_chunks else res.chunk.doc_id)
    return ids


def score_example(
    example: EvalExample,
    retrieval_results: Sequence[RetrievalResult],
    answer: Optional[GeneratedAnswer],
    ks: Sequence[int],
) -> ExampleMetrics:
    """Compute retrieval + answer metrics for a single example."""

    use_chunk_ids = bool(example.relevant_chunk_ids)
    relevant_ids = example.relevant_chunk_ids if use_chunk_ids else example.relevant_doc_ids
    retrieved_ids = _select_ids(retrieval_results, use_chunks=use_chunk_ids)

    recall = {k: recall_at_k(retrieved_ids, relevant_ids, k) for k in ks}
    precision = {k: precision_at_k(retrieved_ids, relevant_ids, k) for k in ks}
    mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)

    citations = answer.citations if answer else []
    cit_cov = citation_coverage(citations, example.relevant_chunk_ids)
    ev_overlap = evidence_overlap(answer.answer if answer else "", example.expected_facts)
    abstained = is_abstention(answer.answer if answer else "")
    abst_correct = abstention_correct(answer.answer if answer else "", example.should_abstain)

    return ExampleMetrics(
        example_id=example.example_id,
        query=example.query,
        retriever=retrieval_results[0].retriever if retrieval_results else "unknown",
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=mrr,
        citation_coverage=cit_cov,
        evidence_overlap=ev_overlap,
        abstained=abstained,
        abstention_correct=abst_correct,
        num_citations=len(citations),
        top_retrieved_ids=retrieved_ids[: max(ks) if ks else len(retrieved_ids)],
    )


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def _aggregate(metrics: Sequence[ExampleMetrics]) -> Dict[str, float]:
    """Compute macro averages across examples."""

    if not metrics:
        return {}

    summary: Dict[str, List[float]] = {}

    def _collect(key: str, value: Optional[float]) -> None:
        if value is None:
            return
        summary.setdefault(key, []).append(value)

    for m in metrics:
        _collect("mrr", m.mrr)
        _collect("citation_coverage", m.citation_coverage)
        _collect("evidence_overlap", m.evidence_overlap)
        _collect(
            "abstention_accuracy",
            float(m.abstention_correct) if m.abstention_correct is not None else None,
        )
        for k, v in m.recall_at_k.items():
            _collect(f"recall@{k}", v)
        for k, v in m.precision_at_k.items():
            _collect(f"precision@{k}", v)

    return {name: float(np.mean(values)) for name, values in summary.items() if values}


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    settings: Optional[Settings] = None,
    eval_path: Optional[Path] = None,
    retriever: Optional[BaseRetriever] = None,
    generate_answers: bool = True,
    ks: Sequence[int] | None = None,
) -> Tuple[List[ExampleMetrics], Dict[str, float]]:
    """Execute the full evaluation pipeline.

    Returns per-example metrics and an aggregate summary.
    """

    settings = settings or Settings.load()
    settings.ensure_output_dirs()

    ks = tuple(ks or (settings.retrieval.top_k,))
    eval_file = eval_path or settings.data.eval_path
    examples = _load_eval_set(Path(eval_file))

    corpus_chunks = _load_corpus(settings.data.raw_dir, settings)
    retriever = retriever or build_retriever_from_config(settings)
    retriever.index(corpus_chunks)

    metrics: List[ExampleMetrics] = []
    for example in examples:
        results = retriever.search(example.query, top_k=max(ks))
        answer: Optional[GeneratedAnswer] = None
        if generate_answers:
            answer = generate_answer(example.query, results, settings)
        metrics.append(score_example(example, results, answer, ks))

    summary = _aggregate(metrics)

    reports_dir = Path(settings.output.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_csv([m.to_flat_dict() for m in metrics], reports_dir / "eval_results.csv")
    _write_json(summary, reports_dir / "eval_summary.json")

    return metrics, summary


def main() -> None:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(
        description="Run retrieval + answer evaluation over the eval set."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to settings YAML (optional)."
    )
    parser.add_argument(
        "--eval-path", type=str, default=None, help="Override path to eval JSONL file."
    )
    parser.add_argument(
        "--retriever", type=str, default=None, help="Force a specific retriever name."
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip answer generation, retrieval-only.",
    )
    args = parser.parse_args()

    settings = Settings.load(args.config) if args.config else Settings.load()
    retriever_obj = build_retriever_from_config(settings, retriever_name=args.retriever)

    metrics, summary = run_benchmark(
        settings=settings,
        eval_path=Path(args.eval_path) if args.eval_path else None,
        retriever=retriever_obj,
        generate_answers=not args.no_generation,
    )

    print(f"Evaluated {len(metrics)} examples. Aggregate metrics:")
    for name, value in summary.items():
        print(f"- {name}: {value:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "EvalExample",
    "ExampleMetrics",
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "mrr",
    "citation_coverage",
    "evidence_overlap",
    "abstention_correct",
    "run_benchmark",
    "score_example",
]
