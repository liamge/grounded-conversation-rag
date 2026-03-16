"""Utility script to prebuild retrieval indexes for all available retrievers.

Usage:
    python -m scripts.build_indices [--config config/settings.yaml] [--force] [--disable-dense]

This avoids importing Streamlit or running the full app; it just ingests,
chunks, and materializes retriever artifacts under ``artifacts/index/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.pipeline import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prebuild RAG retrieval indexes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/settings.yaml"),
        help="Path to settings YAML (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild indexes even if artifacts already exist (clears existing artifact dirs).",
    )
    parser.add_argument(
        "--disable-dense",
        action="store_true",
        help="Skip dense/hybrid retrievers (also sets RAG_DISABLE_DENSE=1 for this run).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.disable_dense:
        os.environ["RAG_DISABLE_DENSE"] = "1"

    try:
        pipeline = RAGPipeline(config_path=args.config)
        pipeline.build_all_indices(force=args.force)
    except Exception as exc:  # pragma: no cover - CLI safety
        print(f"Failed to build indexes: {exc}", file=sys.stderr)
        return 1

    print("Index build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
