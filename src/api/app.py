"""FastAPI application exposing the Grounded Conversation RAG pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from threading import Lock
from typing import List

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import Settings
from ..logging_utils import configure_logging, log_event, request_logging_context
from ..monitoring import aggregate_records
from ..pipeline import PipelineResult, RAGPipeline
from ..schemas import Chunk, GeneratedAnswer, QueryTrace, RetrievalResult, StageTimings
from .models import (
    AnswerModel,
    ChunkModel,
    ErrorResponse,
    HealthResponse,
    IndexRebuildRequest,
    IndexRebuildResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    RetrievalResultModel,
    StageTimingsModel,
    TraceModel,
)

settings = Settings.load()
configure_logging(settings.logging, force=True)
logger = logging.getLogger("rag.api")
app = FastAPI(
    title=f"{settings.app.title} API",
    version="0.1.0",
    description="FastAPI wrapper around the Grounded Conversation RAG pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None
_telemetry: List[QueryTrace] = []
_lock = Lock()


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    with request_logging_context(request_id=request_id):
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.on_event("startup")
async def _startup() -> None:
    """Initialize the pipeline once so indexes are hot when queries arrive."""

    global _pipeline
    logger.info("Starting up RAG API; loading settings and indexes...")
    pipeline = RAGPipeline(settings=settings)
    await run_in_threadpool(pipeline.load_corpus)
    await run_in_threadpool(pipeline.index)
    _pipeline = pipeline
    logger.info("Pipeline ready with %d chunks", len(_pipeline.chunks))


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except StarletteHTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime guard
        log_event(
            logger,
            "error",
            message="Unhandled API error",
            level=logging.ERROR,
        )
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="internal_server_error", detail=str(exc)).model_dump(),
        )

    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    log_event(
        logger,
        "http.request",
        message=f"{request.method} {request.url.path}",
        latency=round(duration_ms, 2),
        chunk_count=None,
    )
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="http_error", detail=exc.detail).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request, exc: Exception
):  # pragma: no cover - safety net
    logger.exception("Unhandled exception during request")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="internal_server_error", detail=str(exc)).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not ready",
        )
    return HealthResponse(
        status="ok",
        retriever_ready=_pipeline._retriever_ready,
        chunks_indexed=len(_pipeline.chunks),
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def query_rag(payload: QueryRequest, request: Request) -> QueryResponse:
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not ready",
        )

    retriever_name = payload.retriever_name or _pipeline.retriever.name
    try:
        with request_logging_context(
            request_id=getattr(request.state, "request_id", None),
            query=payload.query,
            retriever=retriever_name,
        ):
            result: PipelineResult = await run_in_threadpool(
                _pipeline.run,
                query=payload.query,
                retriever_name=payload.retriever_name,
                top_k=payload.top_k,
                max_context_chars=payload.max_context_chars,
                system_instruction=payload.system_instruction,
                use_reranker=payload.use_reranker,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime surface
        logger.exception("Pipeline execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    _record_trace(result.trace)
    return _result_to_response(result)


@app.post(
    "/index/rebuild",
    response_model=IndexRebuildResponse,
    responses={500: {"model": ErrorResponse}},
)
async def rebuild_index(payload: IndexRebuildRequest) -> IndexRebuildResponse:
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not ready",
        )

    try:
        await run_in_threadpool(_pipeline.load_corpus, force=payload.force)
        await run_in_threadpool(_pipeline.index, payload.retriever_name, payload.force)
    except Exception as exc:  # pragma: no cover - runtime surface
        logger.exception("Index rebuild failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return IndexRebuildResponse(
        retriever=_pipeline.retriever.name,
        chunks_indexed=len(_pipeline.chunks),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    with _lock:
        summary = aggregate_records(_telemetry)
        count = len(_telemetry)
    return MetricsResponse(metrics=summary, count=count)


def _record_trace(trace: QueryTrace) -> None:
    with _lock:
        _telemetry.append(trace)


def _chunk_to_model(chunk: Chunk) -> ChunkModel:
    return ChunkModel(
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source=chunk.source,
        title=chunk.title,
        section=chunk.section,
        doc_id=chunk.doc_id,
        metadata=dict(chunk.metadata),
    )


def _retrieval_to_model(result: RetrievalResult) -> RetrievalResultModel:
    return RetrievalResultModel(
        rank=result.rank,
        score=float(result.score),
        retriever=result.retriever,
        chunk=_chunk_to_model(result.chunk),
    )


def _answer_to_model(answer: GeneratedAnswer) -> AnswerModel:
    return AnswerModel(
        answer=answer.answer,
        citations=list(answer.citations),
        evidence_chunks=[_chunk_to_model(c) for c in answer.evidence_chunks],
        supported=answer.supported,
        prompt=answer.prompt,
        context=answer.context,
        provider=answer.provider,
        model=answer.model,
    )


def _trace_to_model(trace: QueryTrace) -> TraceModel:
    return TraceModel(
        query=trace.query,
        latency_ms=trace.latency_ms,
        retriever=trace.retriever,
        top_scores=list(trace.top_scores),
        num_citations=trace.num_citations,
        abstained=trace.abstained,
        prompt_chars=trace.prompt_chars,
        context_chars=trace.context_chars,
        answer_chars=trace.answer_chars,
        timestamp=trace.timestamp,
    )


def _timings_to_model(timings: StageTimings | None) -> StageTimingsModel | None:
    if timings is None:
        return None
    return StageTimingsModel(
        retrieval_ms=timings.retrieval_ms,
        rerank_ms=timings.rerank_ms,
        diversity_ms=timings.diversity_ms,
        generation_ms=timings.generation_ms,
        total_ms=timings.total_ms,
    )


def _result_to_response(result: PipelineResult) -> QueryResponse:
    answer_model = _answer_to_model(result.answer)
    trace_model = _trace_to_model(result.trace)
    retrieval_models = [_retrieval_to_model(r) for r in result.retrieval_results]
    rerank_candidates = [_retrieval_to_model(r) for r in result.reranked_candidates]
    reranked_models = [_retrieval_to_model(r) for r in result.reranked_results]
    timings_model = _timings_to_model(result.timings)

    return QueryResponse(
        answer=answer_model.answer,
        citations=answer_model.citations,
        evidence_chunks=answer_model.evidence_chunks,
        supported=answer_model.supported,
        retrieval_results=retrieval_models,
        reranked_candidates=rerank_candidates,
        retrieved_chunks=reranked_models,
        retriever=result.retriever,
        reranker=result.reranker,
        latency_ms=trace_model.latency_ms,
        trace=trace_model,
        prompt=answer_model.prompt,
        context=answer_model.context,
        model=answer_model.model,
        provider=answer_model.provider,
        timings=timings_model,
    )


__all__ = ["app"]
