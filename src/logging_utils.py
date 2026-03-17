"""Shared structured logging utilities for the RAG project.

This module centralizes JSON logging configuration and lightweight
context propagation (request IDs, query, retriever) so logs from the API,
pipeline, and CLI all carry the same fields.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# Context variables allow propagation across async tasks and threadpools.
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
_query: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("query", default=None)
_retriever: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "retriever", default=None
)


@dataclass
class LoggingSettings:
    level: str = "INFO"
    log_file: Optional[Path] = None
    json_logs: bool = True
    propagate: bool = False


class ContextFilter(logging.Filter):
    """Inject request-scoped fields into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple glue
        record.request_id = getattr(record, "request_id", None) or _request_id.get()
        record.query = getattr(record, "query", None) or _query.get()
        record.retriever = getattr(record, "retriever", None) or _retriever.get()
        record.latency = getattr(record, "latency", None)
        record.chunk_count = getattr(record, "chunk_count", None)
        record.event = getattr(record, "event", None)
        return True


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter keeping messages machine readable."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting only
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in ("event", "request_id", "retriever", "query", "latency", "chunk_count"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _build_handlers(config: LoggingSettings) -> Iterable[logging.Handler]:
    formatter: logging.Formatter = JsonFormatter()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    yield stream

    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        yield file_handler


def configure_logging(config: Optional[Any] = None, *, force: bool = True) -> None:
    """Configure root logging with JSON formatting and context propagation.

    ``config`` can be either ``LoggingSettings`` (defined here) or the
    project ``LoggingConfig`` (from ``config.py``). Only the attributes we
    need are read, keeping this function dependency-light.
    """

    settings = LoggingSettings()
    if config is not None:
        settings.level = getattr(config, "level", settings.level)
        settings.log_file = getattr(config, "log_file", settings.log_file)
        settings.json_logs = getattr(config, "json_logs", settings.json_logs)
        settings.propagate = getattr(config, "propagate", settings.propagate)

    level = logging.getLevelName(str(settings.level).upper())
    handlers = list(_build_handlers(settings))

    logging.basicConfig(level=level, handlers=handlers, force=force)
    root = logging.getLogger()
    root.propagate = settings.propagate
    root.addFilter(ContextFilter())


def log_event(
    logger: logging.Logger,
    event: str,
    message: Optional[str] = None,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Helper to emit structured log entries with a normalized ``event`` field."""

    logger.log(level, message or event, extra={"event": event, **fields})


@contextmanager
def request_logging_context(
    *,
    request_id: Optional[str] = None,
    query: Optional[str] = None,
    retriever: Optional[str] = None,
):
    """Context manager to push request-level fields onto contextvars."""

    tokens = []
    if request_id is not None:
        tokens.append((_request_id, _request_id.set(request_id)))
    if query is not None:
        tokens.append((_query, _query.set(query)))
    if retriever is not None:
        tokens.append((_retriever, _retriever.set(retriever)))

    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current_request_id() -> Optional[str]:
    return _request_id.get()
