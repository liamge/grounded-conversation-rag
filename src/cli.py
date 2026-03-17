"""Unified CLI for Grounded Conversation RAG.

Provides subcommands that wrap existing modules so the same interface works
for local development and CI:
* ``index`` – build retrieval indexes and cache artifacts
* ``eval`` – run the evaluation benchmark and write reports
* ``serve`` – launch the FastAPI backend
* ``query`` – execute a single query through the pipeline and print results

Run with ``python -m src.cli <command> [options]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import Settings
from .logging_utils import configure_logging, request_logging_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_error(message: str) -> None:
    print(f"[error] {message}", file=sys.stderr)


def _load_settings(config_path: str | None) -> Settings:
    try:
        settings = Settings.load(path=config_path) if config_path else Settings.load()
        configure_logging(settings.logging)
        return settings
    except FileNotFoundError as exc:
        _print_error(str(exc))
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - defensive
        _print_error(f"Failed to load settings: {exc}")
        sys.exit(1)


def _confirm_run_header(title: str) -> None:
    print(f"=== {title} ===")


def _handle_import_error(exc: ModuleNotFoundError, context: str) -> None:
    missing = getattr(exc, "name", None) or str(exc)
    _print_error(
        f"Missing dependency '{missing}' required for '{context}'. "
        "Install project requirements (e.g., `pip install -e .[dev,eval,dense]`)."
    )
    sys.exit(1)


def _maybe_disable_dense(disable: bool) -> None:
    if disable:
        os.environ["RAG_DISABLE_DENSE"] = "1"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    _confirm_run_header("Building indexes")
    _maybe_disable_dense(args.no_dense)
    settings = _load_settings(args.config)

    try:
        from .pipeline import RAGPipeline
    except ModuleNotFoundError as exc:
        _handle_import_error(exc, "index")

    pipeline = RAGPipeline(settings=settings)
    try:
        pipeline.load_corpus(force=args.force)
        pipeline.build_all_indices(force=args.force)
    except Exception as exc:  # pragma: no cover - runtime guard
        _print_error(f"Index build failed: {exc}")
        sys.exit(1)

    print(
        f"Indexed {len(pipeline.chunks)} chunks. Artifacts at {settings.output.artifacts_dir}/index"
    )


def cmd_eval(args: argparse.Namespace) -> None:
    _confirm_run_header("Running evaluation benchmark")
    _maybe_disable_dense(args.no_dense)
    settings = _load_settings(args.config)

    try:
        from .evaluation import run_benchmark
        from .retrieval import build_retriever_from_config
    except ModuleNotFoundError as exc:
        _handle_import_error(exc, "eval")

    retriever = None
    if args.retriever:
        retriever = build_retriever_from_config(settings, retriever_name=args.retriever)

    eval_path = Path(args.eval_path) if args.eval_path else None
    ks = tuple(args.k) if args.k else None

    try:
        metrics, summary = run_benchmark(
            settings=settings,
            eval_path=eval_path,
            retriever=retriever,
            generate_answers=not args.no_generation,
            ks=ks,
            report_prefix=args.report_prefix,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        _print_error(f"Evaluation failed: {exc}")
        sys.exit(1)

    print(f"Evaluated {len(metrics)} examples. Aggregates:")
    if not summary:
        print("(no metrics computed)")
    else:
        for name, value in sorted(summary.items()):
            print(f"- {name}: {value:.4f}")
    print(f"Reports written to {settings.output.reports_dir}")


def cmd_serve(args: argparse.Namespace) -> None:
    _confirm_run_header("Starting FastAPI server")
    _maybe_disable_dense(args.no_dense)
    settings = _load_settings(args.config)

    try:
        import uvicorn  # Local import so help/other commands work without the dep
    except Exception as exc:  # pragma: no cover - import guard
        _print_error(f"uvicorn is required for 'serve' (pip install uvicorn[standard]): {exc}")
        sys.exit(1)

    host = args.host or str(settings.app.host)
    port = args.port or settings.app.port
    reload = args.reload

    print(f"Listening on http://{host}:{port} (reload={'on' if reload else 'off'})")
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.logging.level.lower(),
    )


def cmd_query(args: argparse.Namespace) -> None:
    _confirm_run_header("Running single query")
    _maybe_disable_dense(args.no_dense)
    settings = _load_settings(args.config)

    try:
        from .pipeline import RAGPipeline
    except ModuleNotFoundError as exc:
        _handle_import_error(exc, "query")

    pipeline = RAGPipeline(settings=settings)
    try:
        with request_logging_context(
            query=args.query,
            retriever=args.retriever or pipeline.retriever.name,
        ):
            result = pipeline.run(
                args.query,
                retriever_name=args.retriever,
                top_k=args.top_k,
                max_context_chars=args.max_context_chars,
                system_instruction=args.system_instruction,
                use_reranker=args.use_reranker,
            )
    except Exception as exc:  # pragma: no cover - runtime guard
        _print_error(f"Query failed: {exc}")
        sys.exit(1)

    payload: Dict[str, Any] = result.to_dict()
    summary = {
        "query": result.query,
        "retriever": result.retriever,
        "reranker": result.reranker or "none",
        "latency_ms": round(result.trace.latency_ms, 2),
        "num_citations": len(result.answer.citations),
    }

    print("Summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print("Answer:")
    print(result.answer.answer)
    print("\nStructured output (JSON):")
    print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for Grounded Conversation RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index -----------------------------------------------------------
    p_index = subparsers.add_parser("index", help="Build retrieval indexes and artifacts")
    p_index.add_argument("--config", type=str, default=None, help="Path to settings YAML")
    p_index.add_argument("--force", action="store_true", help="Rebuild even if artifacts exist")
    p_index.add_argument(
        "--no-dense",
        action="store_true",
        help="Skip dense + hybrid retrievers (also sets RAG_DISABLE_DENSE=1 for this run)",
    )
    p_index.set_defaults(func=cmd_index)

    # eval ------------------------------------------------------------
    p_eval = subparsers.add_parser("eval", help="Run evaluation benchmark over the eval set")
    p_eval.add_argument("--config", type=str, default=None, help="Path to settings YAML")
    p_eval.add_argument("--eval-path", type=str, default=None, help="Override eval JSONL path")
    p_eval.add_argument("--retriever", type=str, default=None, help="Force a specific retriever")
    p_eval.add_argument(
        "--report-prefix",
        type=str,
        default="eval",
        help="Prefix for eval artifacts (writes <prefix>_results.csv and <prefix>_summary.json)",
    )
    p_eval.add_argument(
        "--k",
        type=int,
        nargs="+",
        help="Override list of k values (e.g., --k 3 5)",
    )
    p_eval.add_argument(
        "--no-generation",
        action="store_true",
        help="Skip answer generation (retrieval-only metrics)",
    )
    p_eval.add_argument(
        "--no-dense",
        action="store_true",
        help="Skip dense/hybrid retrievers when running the benchmark",
    )
    p_eval.set_defaults(func=cmd_eval)

    # serve -----------------------------------------------------------
    p_serve = subparsers.add_parser("serve", help="Launch the FastAPI backend")
    p_serve.add_argument("--config", type=str, default=None, help="Path to settings YAML")
    p_serve.add_argument(
        "--host", type=str, default=None, help="Bind address (default from settings)"
    )
    p_serve.add_argument("--port", type=int, default=None, help="Port (default from settings)")
    p_serve.add_argument("--reload", action="store_true", help="Enable autoreload for local dev")
    p_serve.add_argument(
        "--no-dense",
        action="store_true",
        help="Disable dense models when starting the API (sets RAG_DISABLE_DENSE=1)",
    )
    p_serve.set_defaults(func=cmd_serve)

    # query -----------------------------------------------------------
    p_query = subparsers.add_parser("query", help="Run a single query through the pipeline")
    p_query.add_argument("query", type=str, help="Query text to execute")
    p_query.add_argument("--config", type=str, default=None, help="Path to settings YAML")
    p_query.add_argument("--retriever", type=str, default=None, help="Retriever name to use")
    p_query.add_argument("--top-k", type=int, default=None, help="Override top-k retrieval")
    p_query.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="Context character budget for prompt assembly",
    )
    p_query.add_argument(
        "--system-instruction",
        type=str,
        default=None,
        help="Optional system prompt override for generation",
    )
    reranker_group = p_query.add_mutually_exclusive_group()
    reranker_group.add_argument(
        "--reranker",
        dest="use_reranker",
        action="store_true",
        help="Force reranker on",
    )
    reranker_group.add_argument(
        "--no-reranker",
        dest="use_reranker",
        action="store_false",
        help="Disable reranker",
    )
    p_query.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    p_query.add_argument(
        "--no-dense",
        action="store_true",
        help="Disable dense + hybrid retrievers for this query run",
    )
    p_query.set_defaults(func=cmd_query, use_reranker=None)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
