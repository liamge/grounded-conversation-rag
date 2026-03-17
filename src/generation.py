"""Grounded answer generation utilities.

This module keeps answer generation provider‑agnostic while enforcing
grounding requirements:

* Retrieved context is assembled with a token budget (approximate word
  tokens) to avoid overly long prompts, truncating oversized chunks
  rather than skipping them.
* Prompts require chunk‑level citations (``[chunk_id]``) and abstention
  when evidence is insufficient.
* A provider interface allows plugging in OpenAI or other chat models,
  with a deterministic fallback that works without external API keys.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Settings, is_truthy_env
from .schemas import Chunk, GeneratedAnswer, RetrievalResult

DEFAULT_CONTEXT_CHARS = 2_400


# ---------------------------------------------------------------------------
# Context assembly and prompt formatting
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Lightweight whitespace tokenizer used for budget accounting."""

    return [tok for tok in text.replace("\n", " ").split(" ") if tok]


def assemble_context_with_budget(
    chunks: Sequence[RetrievalResult], max_tokens: int = DEFAULT_CONTEXT_CHARS
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Assemble context under a token budget with truncation metadata.

    Args:
        chunks: Retrieval results ordered by relevance (already ranked).
        max_tokens: Rough token budget (whitespace-delimited words). A minimum
            of one token is enforced when chunks exist so the context is never
            empty.

    Returns:
        context_text: The assembled context string.
        used_chunk_ids: Chunk ids included (for citation mapping).
        truncation_metadata: Stats about truncation/dropping decisions.
    """

    if not chunks:
        return "", [], {
            "budget_tokens": max_tokens,
            "effective_budget_tokens": max_tokens,
            "used_tokens": 0,
            "truncated_chunks": [],
            "dropped_chunks": [],
        }

    effective_budget = max(max_tokens, 1)
    remaining = effective_budget

    context_blocks: List[str] = []
    used_chunk_ids: List[str] = []
    truncated_chunks: List[Dict[str, int]] = []
    dropped_chunks: List[str] = []

    for idx, res in enumerate(chunks):
        if remaining <= 0 and used_chunk_ids:
            dropped_chunks.extend(r.chunk.chunk_id for r in chunks[idx:])
            break

        chunk = res.chunk
        header = f"[{chunk.chunk_id}] {chunk.title or 'Untitled'} — {chunk.source}"
        body = chunk.text.strip()

        header_tokens = _tokenize(header)
        body_tokens = _tokenize(body)
        original_tokens = len(header_tokens) + len(body_tokens)

        tokens_allowed = remaining
        header_slice = header_tokens[:tokens_allowed]
        tokens_used = len(header_slice)
        tokens_allowed -= tokens_used

        body_slice: List[str] = []
        if tokens_allowed > 0:
            body_slice = body_tokens[:tokens_allowed]
            tokens_used += len(body_slice)
            tokens_allowed -= len(body_slice)

        if not header_slice and not body_slice:
            # Should only happen if effective_budget == 0, which we guard above.
            continue

        block_header = " ".join(header_slice).strip()
        block_body = " ".join(body_slice).strip()
        block_text = block_header if not block_body else f"{block_header}\n{block_body}"

        context_blocks.append(block_text)
        used_chunk_ids.append(chunk.chunk_id)
        remaining = max(remaining - tokens_used, 0)

        if tokens_used < original_tokens:
            truncated_chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "original_tokens": original_tokens,
                    "included_tokens": tokens_used,
                }
            )

    used_tokens = effective_budget - remaining
    context_text = "\n\n".join(context_blocks)

    truncation_metadata = {
        "budget_tokens": max_tokens,
        "effective_budget_tokens": effective_budget,
        "used_tokens": used_tokens,
        "truncated_chunks": truncated_chunks,
        "dropped_chunks": dropped_chunks,
    }

    return context_text, used_chunk_ids, truncation_metadata


def assemble_context(
    results: Sequence[RetrievalResult], max_chars: int = DEFAULT_CONTEXT_CHARS
) -> str:
    """Backward-compatible wrapper returning only the context text."""

    context_text, _, _ = assemble_context_with_budget(results, max_tokens=max_chars)
    return context_text


def format_grounded_prompt(
    question: str,
    retrieved: Sequence[RetrievalResult],
    *,
    max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    system_instruction: Optional[str] = None,
    context_block: Optional[str] = None,
) -> str:
    """Build the full prompt for grounded generation.

    The default instruction enforces:
    * cite sources using ``[chunk_id]``
    * avoid fabrication; abstain with a fixed sentence when unsure
    * concise, direct answers
    """

    if context_block is None:
        context_block, _, _ = assemble_context_with_budget(retrieved, max_tokens=max_context_chars)

    system_instruction = system_instruction or (
        "You are a grounded assistant. Use ONLY the provided context to answer. "
        "Cite sources inline using [chunk_id] after the sentence they support. "
        "If the context does not contain enough evidence, respond exactly with: "
        "I don't have enough evidence to answer confidently."
    )

    prompt = (
        f"SYSTEM:\n{system_instruction}\n\n"
        f"CONTEXT:\n{context_block or '[no context available]'}\n\n"
        f"QUESTION:\n{question.strip()}\n\n"
        "ANSWER:" 
    )

    return prompt


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter that avoids extra dependencies."""

    cleaned = " ".join(text.replace("\n", " ").split())
    if not cleaned:
        return []
    # Treat common sentence enders as boundaries while keeping determinism.
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Provider-agnostic generator interface
# ---------------------------------------------------------------------------


class BaseGenerator:
    """Abstract generator interface."""

    name: str = "base"
    model: str = ""

    def generate(
        self,
        question: str,
        retrieved: Sequence[RetrievalResult],
        *,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
        system_instruction: Optional[str] = None,
    ) -> GeneratedAnswer:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIChatGenerator(BaseGenerator):
    """Chat completion generator using the OpenAI Python SDK."""

    name = "openai"

    def __init__(self, model: str, client: Any | None = None) -> None:
        self.model = model
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> Any:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "openai package is required for OpenAIChatGenerator. Install the optional 'eval' extra."  # noqa: E501
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAI provider.")
        return OpenAI(api_key=api_key)

    def generate(
        self,
        question: str,
        retrieved: Sequence[RetrievalResult],
        *,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
        system_instruction: Optional[str] = None,
    ) -> GeneratedAnswer:
        context_block, used_chunk_ids, truncation_metadata = assemble_context_with_budget(
            retrieved, max_tokens=max_context_chars
        )

        prompt = format_grounded_prompt(
            question,
            retrieved,
            max_context_chars=max_context_chars,
            system_instruction=system_instruction,
            context_block=context_block,
        )

        messages = [
            {
                "role": "system",
                "content": prompt.split("CONTEXT:")[0].replace("SYSTEM:\n", "").strip(),
            },
            {"role": "user", "content": prompt[prompt.index("CONTEXT:"):]},
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )

        answer_text = response.choices[0].message.content.strip()

        return _finalize_generated_answer(
            raw_answer=answer_text,
            retrieved=retrieved,
            used_chunk_ids=used_chunk_ids,
            prompt=prompt,
            context_block=context_block,
            truncation_metadata=truncation_metadata,
            provider=self.name,
            model=self.model,
        )


@dataclass(slots=True)
class _CandidateSentence:
    chunk_id: str
    text: str
    score: float


def _normalize_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Trim stray quotes or trailing punctuation fragments (ASCII only)
    cleaned = cleaned.strip('"\' ')
    cleaned = cleaned.rstrip(" ,;:-")
    cleaned = re.sub(r"\b(\w+)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    if len(cleaned) > 320:
        cleaned = cleaned[:320].rstrip(" ,;:-") + "…"
    return cleaned


def _jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(_tokenize(a.lower()))
    b_tokens = set(_tokenize(b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union


class LightweightExtractiveGenerator(BaseGenerator):
    """Sentence-level extractive generator with inline citations.

    Designed for demo/hosted defaults: deterministic, citation-aware, and
    dependency-light (no external APIs or local model weights).
    """

    name = "lightweight_extractive"

    def __init__(self, max_sentences: int = 4, model: str = "lite-extractive-1") -> None:
        self.max_sentences = max_sentences
        self.model = model
        self._redundancy_threshold = 0.82

    def _score_sentence(
        self,
        sentence: str,
        query_terms: set[str],
        retrieval_norm: float,
        rank: int,
        position: int,
    ) -> float:
        tokens = set(_tokenize(sentence.lower()))
        if not tokens:
            return 0.0

        overlap = len(tokens & query_terms)
        coverage = overlap / max(len(tokens), 1)

        keyword_score = overlap * 1.3 + coverage
        retrieval_score = retrieval_norm * 1.4
        rank_bonus = max(0.0, 0.6 - 0.08 * (rank - 1))
        position_bonus = max(0.0, 0.25 - 0.05 * position)

        return keyword_score + retrieval_score + rank_bonus + position_bonus

    def _collect_candidates(
        self,
        question: str,
        retrieved: Sequence[RetrievalResult],
        allowed_ids: Sequence[str],
    ) -> List[_CandidateSentence]:
        query_terms = {tok.lower() for tok in _tokenize(question)}
        max_score = max((res.score for res in retrieved), default=0.0)

        candidates: List[_CandidateSentence] = []
        for res in retrieved:
            if res.chunk.chunk_id not in allowed_ids:
                continue

            retrieval_norm = (res.score / max_score) if max_score > 0 else 0.0
            sentences = _split_sentences(res.chunk.text) or [res.chunk.text.strip()]
            for position, sentence in enumerate(sentences):
                normalized = _normalize_sentence(sentence)
                if not normalized:
                    continue
                tokens = set(_tokenize(normalized.lower()))
                overlap = len(tokens & query_terms)
                if overlap == 0:
                    continue
                score = self._score_sentence(
                    normalized, query_terms, retrieval_norm, res.rank, position
                )
                if score <= 0:
                    continue
                # small deterministic tie-breaker favors earlier sentences
                score -= position * 0.001
                candidates.append(_CandidateSentence(res.chunk.chunk_id, normalized, score))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _select_sentences(
        self, candidates: Sequence[_CandidateSentence]
    ) -> List[_CandidateSentence]:
        selected: List[_CandidateSentence] = []
        for cand in candidates:
            if len(selected) >= self.max_sentences:
                break
            if any(
                _jaccard_similarity(cand.text, kept.text) >= self._redundancy_threshold
                for kept in selected
            ):
                continue
            selected.append(cand)
        return selected

    def _compose_answer(self, selected: Sequence[_CandidateSentence]) -> str:
        if not selected:
            return "I don't have enough evidence to answer confidently."

        lead = selected[0]
        bullets = selected[1:]

        lines: List[str] = [f"{lead.text} [{lead.chunk_id}]"]
        if bullets:
            lines.append("")
            for cand in bullets:
                lines.append(f"- {cand.text} [{cand.chunk_id}]")
        lines.append("")
        lines.append("Lightweight grounded generator (no external API).")

        return "\n".join(line for line in lines if line.strip())

    def generate(
        self,
        question: str,
        retrieved: Sequence[RetrievalResult],
        *,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
        system_instruction: Optional[str] = None,
    ) -> GeneratedAnswer:
        context_block, used_chunk_ids, truncation_metadata = assemble_context_with_budget(
            retrieved, max_tokens=max_context_chars
        )

        prompt = format_grounded_prompt(
            question,
            retrieved,
            max_context_chars=max_context_chars,
            system_instruction=system_instruction,
            context_block=context_block,
        )

        candidates = self._collect_candidates(question, retrieved, used_chunk_ids)
        selected = self._select_sentences(candidates)

        raw_answer = self._compose_answer(selected)

        return _finalize_generated_answer(
            raw_answer=raw_answer,
            retrieved=retrieved,
            used_chunk_ids=used_chunk_ids,
            prompt=prompt,
            context_block=context_block,
            truncation_metadata=truncation_metadata,
            provider=self.name,
            model=self.model,
        )


class LocalTransformersGenerator(BaseGenerator):
    """Optional local generator using tiny Transformers models.

    Requires the ``local_llm`` extra (transformers + torch). Only invoked when
    explicitly configured via ``RAG_MODELS__LLM_PROVIDER=local``.
    """

    name = "local"

    def __init__(self, model: str = "sshleifer/tiny-gpt2") -> None:
        self.model = model
        self._pipeline: Any | None = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                (
                    "transformers is required for LocalTransformersGenerator. "
                    "Install the 'local_llm' extra."
                )
            ) from exc

        self._pipeline = pipeline(
            "text-generation",
            model=self.model,
            # keep deterministic and small for local laptops
            do_sample=False,
        )
        return self._pipeline

    def generate(
        self,
        question: str,
        retrieved: Sequence[RetrievalResult],
        *,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
        system_instruction: Optional[str] = None,
    ) -> GeneratedAnswer:
        context_block, used_chunk_ids, truncation_metadata = assemble_context_with_budget(
            retrieved, max_tokens=max_context_chars
        )

        prompt = format_grounded_prompt(
            question,
            retrieved,
            max_context_chars=max_context_chars,
            system_instruction=system_instruction,
            context_block=context_block,
        )

        pipe = self._get_pipeline()
        raw = pipe(prompt, max_new_tokens=120, num_return_sequences=1)[0]["generated_text"]
        # Strip the prompt prefix if the model echoes it back.
        answer_text = raw[len(prompt) :].strip() or raw.strip()

        return _finalize_generated_answer(
            raw_answer=answer_text,
            retrieved=retrieved,
            used_chunk_ids=used_chunk_ids,
            prompt=prompt,
            context_block=context_block,
            truncation_metadata=truncation_metadata,
            provider=self.name,
            model=self.model,
        )


class DeterministicFallbackGenerator(BaseGenerator):
    """Deterministic generator that works without external APIs.

    Strategy: synthesize a concise fallback answer from the top retrieved
    chunks (up to three), preserving citations and clearly indicating the
    response is a fallback. This keeps behavior deterministic while providing
    a useful, non-empty answer even when LLM providers are unavailable.
    """

    name = "fallback"

    def __init__(self, model: str = "stub-grounded") -> None:
        self.model = model

    @staticmethod
    def _summarize_chunk(text: str, max_chars: int = 260) -> str:
        """Derive a short, deterministic summary from a chunk body."""

        normalized = " ".join(text.split())
        if not normalized:
            return "No content available."

        # Take the first sentence-like span if available.
        sentence_endings = [". ", "? ", "! "]
        end_positions = [
            normalized.find(sep) + len(sep) for sep in sentence_endings if sep in normalized
        ]
        cutoff = min(end_positions) if end_positions else len(normalized)
        snippet = normalized[:cutoff]

        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip(" ,.;:") + "…"

        return snippet

    def generate(
        self,
        question: str,
        retrieved: Sequence[RetrievalResult],
        *,
        max_context_chars: int = DEFAULT_CONTEXT_CHARS,
        system_instruction: Optional[str] = None,
    ) -> GeneratedAnswer:
        context_block, used_chunk_ids, truncation_metadata = assemble_context_with_budget(
            retrieved, max_tokens=max_context_chars
        )

        prompt = format_grounded_prompt(
            question,
            retrieved,
            max_context_chars=max_context_chars,
            system_instruction=system_instruction,
            context_block=context_block,
        )

        if not retrieved:
            fallback_answer = (
                "Grounded fallback: no source snippets are available yet, "
                "so I cannot answer confidently."
            )
            return _finalize_generated_answer(
                raw_answer=fallback_answer,
                retrieved=retrieved,
                used_chunk_ids=used_chunk_ids,
                prompt=prompt,
                context_block=context_block,
                truncation_metadata=truncation_metadata,
                provider=self.name,
                model=self.model,
            )

        # Synthesize from the top available context chunks (cap at three for brevity).
        chunk_lookup = {res.chunk.chunk_id: res.chunk for res in retrieved}
        target_ids = used_chunk_ids[:3] if used_chunk_ids else [retrieved[0].chunk.chunk_id]

        summaries: List[str] = []
        for cid in target_ids:
            chunk = chunk_lookup.get(cid)
            if not chunk:
                continue
            summary = self._summarize_chunk(chunk.text)
            summaries.append(f"- {summary} [{cid}]")

        if not summaries:
            summaries = ["- No content available. [unknown]"]

        answer = "\n".join([
            "Grounded fallback (deterministic synthesis):",
            *summaries,
        ])

        return _finalize_generated_answer(
            raw_answer=answer,
            retrieved=retrieved,
            used_chunk_ids=used_chunk_ids,
            prompt=prompt,
            context_block=context_block,
            truncation_metadata=truncation_metadata,
            provider=self.name,
            model=self.model,
        )


def build_generator(settings: Settings, client: Any | None = None) -> BaseGenerator:
    """Factory that prefers the configured provider but falls back gracefully."""

    provider = (settings.models.llm_provider or "lightweight").lower()
    model = settings.models.llm_model
    demo_mode = settings.demo_mode or is_truthy_env("RAG_DEMO_MODE")
    api_key = os.getenv("OPENAI_API_KEY")
    prefer_openai = bool(api_key) and not demo_mode and provider == "openai"

    if prefer_openai:
        try:
            return OpenAIChatGenerator(model=model, client=client)
        except Exception:
            # Fall through to a lightweight path if OpenAI is unavailable.
            pass

    if provider in {"local", "transformers", "hf"}:
        try:
            return LocalTransformersGenerator(model=model)
        except Exception:
            return LightweightExtractiveGenerator()

    if provider in {"demo", "extractive", "fallback", "lightweight"} or demo_mode:
        return LightweightExtractiveGenerator()

    if provider == "deterministic":
        return DeterministicFallbackGenerator(model="deterministic-fallback")

    return LightweightExtractiveGenerator()


def generate_answer(
    question: str,
    retrieved: Sequence[RetrievalResult],
    settings: Settings,
    *,
    max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    system_instruction: Optional[str] = None,
    client: Any | None = None,
) -> GeneratedAnswer:
    """High-level convenience function to produce a grounded answer."""

    generator = build_generator(settings, client=client)
    return generator.generate(
        question,
        retrieved,
        max_context_chars=max_context_chars,
        system_instruction=system_instruction,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_citations(text: str) -> List[str]:
    """Pull chunk_id-style citations (``[chunk_xxx]``) from model output."""

    pattern = re.compile(r"\[(chunk_[a-zA-Z0-9]+)\]")
    return list(dict.fromkeys(pattern.findall(text)))  # preserve order, drop dupes


def _validate_citations(
    citations: Sequence[str],
    allowed_chunk_ids: Sequence[str],
    retrieved: Sequence[RetrievalResult],
) -> Tuple[List[str], List[Chunk]]:
    """Filter citations to those present in the assembled context.

    Returns ordered, deduplicated citations plus the corresponding ``Chunk`` objects.
    """

    allowed = set(allowed_chunk_ids)
    chunk_lookup = {res.chunk.chunk_id: res.chunk for res in retrieved}

    valid: List[str] = []
    evidence: List[Chunk] = []
    for cid in citations:
        if cid in allowed and cid in chunk_lookup and cid not in valid:
            valid.append(cid)
            evidence.append(chunk_lookup[cid])

    return valid, evidence


def _strip_invalid_citations(text: str, valid_citations: Sequence[str]) -> str:
    """Remove references to invalid/missing citations from the answer text."""

    if not text:
        return ""

    valid_set = set(valid_citations)
    pattern = re.compile(r"\[(chunk_[a-zA-Z0-9]+)\]")

    def _sub(match: re.Match[str]) -> str:
        return match.group(0) if match.group(1) in valid_set else ""

    cleaned = pattern.sub(_sub, text)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def _finalize_generated_answer(
    *,
    raw_answer: str,
    retrieved: Sequence[RetrievalResult],
    used_chunk_ids: Sequence[str],
    prompt: str,
    context_block: str,
    truncation_metadata: Dict[str, Any],
    provider: str,
    model: str,
) -> GeneratedAnswer:
    """Validate citations and produce a structured ``GeneratedAnswer``."""

    raw_citations = _extract_citations(raw_answer)
    valid_citations, evidence_chunks = _validate_citations(raw_citations, used_chunk_ids, retrieved)
    cleaned_answer = _strip_invalid_citations(raw_answer, valid_citations)
    supported = bool(valid_citations)

    metadata = {
        "context_chunks": list(used_chunk_ids),
        "truncation": truncation_metadata,
        "raw_citations": raw_citations,
        "invalid_citations": [c for c in raw_citations if c not in valid_citations],
    }

    return GeneratedAnswer(
        answer=cleaned_answer,
        citations=valid_citations,
        evidence_chunks=evidence_chunks,
        supported=supported,
        prompt=prompt,
        context=context_block,
        provider=provider,
        model=model,
        metadata=metadata,
    )


__all__ = [
    "GeneratedAnswer",
    "assemble_context",
    "format_grounded_prompt",
    "BaseGenerator",
    "OpenAIChatGenerator",
    "LightweightExtractiveGenerator",
    "LocalTransformersGenerator",
    "DeterministicFallbackGenerator",
    "build_generator",
    "generate_answer",
    "assemble_context_with_budget",
]
