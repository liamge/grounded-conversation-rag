"""Centralized configuration for the RAG system.

This module provides a strongly typed configuration object with sensible defaults,
optional YAML loading, and environment variable overrides. It is intentionally
lightweight (no dependency on pydantic-settings) while still leveraging Pydantic's
validation for safety.
"""

from __future__ import annotations

import os
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:  # Pydantic v2
    from pydantic import BaseModel, ConfigDict, field_validator
    _HAS_PYDANTIC_V2 = True
except ImportError:  # Fallback for environments still on pydantic v1
    from pydantic import BaseModel, validator as field_validator  # type: ignore

    ConfigDict = None  # type: ignore
    _HAS_PYDANTIC_V2 = False


DEFAULT_CONFIG_PATH = Path("config/settings.yaml")
ENV_PREFIX = "RAG_"


class _BaseConfigModel(BaseModel):
    """Base model that ignores extra keys across Pydantic v1/v2."""

    if ConfigDict:
        model_config = ConfigDict(extra="ignore")
    else:  # pragma: no cover - compatibility path
        class Config:
            extra = "ignore"


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` without mutating inputs."""

    def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(a)
        for key, value in b.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge(result[key], value)
            else:
                result[key] = value
        return result

    return _merge(base, updates)


def _coerce_env_value(value: str) -> Any:
    """Best-effort conversion from string env values to Python types."""

    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    # Comma-separated lists, e.g., "a,b,c" -> ["a", "b", "c"].
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        # If all parts parse as numbers, return a numeric list; otherwise strings.
        numeric_parts: List[Any] = []
        for part in parts:
            try:
                numeric_parts.append(ast.literal_eval(part))
            except (ValueError, SyntaxError):
                numeric_parts.append(part)
        return numeric_parts

    # Try literal eval for numbers, lists, dicts, etc.
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _env_to_nested(prefix: str) -> Dict[str, Any]:
    """Convert environment variables with ``prefix`` into a nested dict.

    Environment variable format: ``{PREFIX}{SECTION}__{FIELD}=value``
    Example: ``RAG_RETRIEVAL__TOP_K=10`` -> {"retrieval": {"top_k": 10}}
    """

    nested: Dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(prefix):
            continue

        # Strip prefix and split on double underscores to get path components
        path = raw_key[len(prefix) :].split("__")
        if not path:
            continue

        current = nested
        for segment in path[:-1]:
            key = segment.lower()
            current = current.setdefault(key, {})  # type: ignore[assignment]

        leaf_key = path[-1].lower()
        current[leaf_key] = _coerce_env_value(raw_value)

    return nested


class DataPaths(_BaseConfigModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    embeddings_dir: Path = Path("data/embeddings")
    eval_path: Path = Path("data/eval/eval_set.jsonl")


class RetrievalConfig(_BaseConfigModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    min_score: float = 0.0
    bm25_k1: float = 1.6
    bm25_b: float = 0.75
    hybrid_weight_dense: float = 0.5
    hybrid_weight_lexical: float = 0.5
    use_reranker: bool = False
    reranker_top_n: int = 10
    max_chunks_per_document: int = 0
    duplicate_similarity_threshold: float = 0.9
    enable_diversity_filter: bool = True

    @field_validator("hybrid_weight_dense", "hybrid_weight_lexical")
    @classmethod
    def _weights_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("hybrid weights must be between 0 and 1")
        return v

    @field_validator("duplicate_similarity_threshold")
    @classmethod
    def _similarity_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("duplicate_similarity_threshold must be between 0 and 1")
        return v

    @field_validator("max_chunks_per_document")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_chunks_per_document cannot be negative")
        return v


class ModelConfig(_BaseConfigModel):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    tokenizer_name: Optional[str] = None


class OutputConfig(_BaseConfigModel):
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    cache_dir: Path = Path(".cache")
    logs_dir: Path = Path("logs")


class LoggingConfig(_BaseConfigModel):
    level: str = "INFO"
    log_file: Optional[Path] = Path("logs/app.log")
    json_logs: bool = True
    propagate: bool = False


class AppConfig(_BaseConfigModel):
    title: str = "Grounded Conversation RAG"
    default_retriever: str = "hybrid"  # lexical | dense | hybrid
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    max_history_turns: int = 3
    enable_feedback: bool = False


class Settings(_BaseConfigModel):
    """Container for all project configuration groups."""

    data: DataPaths = DataPaths()
    retrieval: RetrievalConfig = RetrievalConfig()
    models: ModelConfig = ModelConfig()
    output: OutputConfig = OutputConfig()
    logging: LoggingConfig = LoggingConfig()
    app: AppConfig = AppConfig()

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Settings":
        """Load settings from a YAML file. Missing keys fall back to defaults."""

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return cls._from_mapping(payload)

    @classmethod
    def load(
        cls,
        path: Path | str | None = None,
        env_prefix: str = ENV_PREFIX,
        allow_missing_file: bool = True,
    ) -> "Settings":
        """Load configuration with the following precedence (lowest to highest):

        1) Built-in defaults declared in the models
        2) YAML file values (if provided and found)
        3) Environment variables with the given prefix

        Environment variables follow the pattern ``{PREFIX}{SECTION}__{FIELD}``.
        Example: ``RAG_APP__PORT=8080`` or ``RAG_MODELS__EMBEDDING_MODEL=...``.
        """

        base_data: Dict[str, Any] = {}

        if path is None:
            path = DEFAULT_CONFIG_PATH

        yaml_path = Path(path)
        if yaml_path.exists():
            with yaml_path.open("r", encoding="utf-8") as f:
                base_data = yaml.safe_load(f) or {}
        elif not allow_missing_file:
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        env_overrides = _env_to_nested(env_prefix)
        merged = _deep_merge(base_data, env_overrides)

        return cls._from_mapping(merged)

    @classmethod
    def _from_mapping(cls, payload: Dict[str, Any]) -> "Settings":
        if _HAS_PYDANTIC_V2:
            return cls.model_validate(payload)
        return cls.parse_obj(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""

        return self.model_dump()

    def ensure_output_dirs(self) -> None:
        """Create output/log directories if they do not exist."""

        for path in [
            self.output.artifacts_dir,
            self.output.artifacts_dir / "index",
            self.output.reports_dir,
            self.output.cache_dir,
            self.output.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "Settings",
    "DataPaths",
    "RetrievalConfig",
    "ModelConfig",
    "OutputConfig",
    "LoggingConfig",
    "AppConfig",
    "DEFAULT_CONFIG_PATH",
    "ENV_PREFIX",
]
