"""Pydantic request/response schemas for the FastAPI service."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, conint, confloat


class HealthResponse(BaseModel):
    status: str = "ok"
    retriever_ready: bool
    chunks_indexed: int


class ChunkModel(BaseModel):
    chunk_id: str
    text: str
    source: str
    title: str
    section: str
    doc_id: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class RetrievalResultModel(BaseModel):
    rank: int
    score: float
    retriever: str
    chunk: ChunkModel


class AnswerModel(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    evidence_chunks: List[ChunkModel] = Field(default_factory=list)
    supported: bool = True
    prompt: str
    context: str
    provider: str
    model: str


class TraceModel(BaseModel):
    query: str
    latency_ms: float
    retriever: str
    top_scores: List[float] = Field(default_factory=list)
    num_citations: int
    abstained: bool
    prompt_chars: int
    context_chars: int
    answer_chars: int
    timestamp: datetime


class StageTimingsModel(BaseModel):
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    diversity_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to answer")
    retriever_name: Optional[str] = Field(
        None, description="Retriever to use: hybrid|bm25|tfidf|dense"
    )
    top_k: Optional[conint(gt=0, le=50)] = Field(
        None, description="Number of chunks to pass to the generator"
    )
    max_context_chars: Optional[conint(gt=0)] = Field(
        None, description="Cap on concatenated context passed to the LLM"
    )
    use_reranker: Optional[bool] = Field(None, description="Enable keyword reranker")
    system_instruction: Optional[str] = Field(
        None, description="Optional override for the system prompt"
    )


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    evidence_chunks: List[ChunkModel]
    supported: bool
    retrieval_results: List[RetrievalResultModel]
    reranked_candidates: List[RetrievalResultModel] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalResultModel]
    retriever: str
    reranker: Optional[str]
    latency_ms: float
    trace: TraceModel
    prompt: str
    context: str
    model: str
    provider: str
    timings: Optional[StageTimingsModel] = None


class IndexRebuildRequest(BaseModel):
    retriever_name: Optional[str] = None
    force: bool = True


class IndexRebuildResponse(BaseModel):
    status: str = "ok"
    retriever: str
    chunks_indexed: int


class MetricsResponse(BaseModel):
    metrics: Dict[str, float] = Field(default_factory=dict)
    count: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
