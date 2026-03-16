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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Settings
from .schemas import GeneratedAnswer, RetrievalResult

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


def assemble_context(results: Sequence[RetrievalResult], max_chars: int = DEFAULT_CONTEXT_CHARS) -> str:
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
            {"role": "system", "content": prompt.split("CONTEXT:")[0].replace("SYSTEM:\n", "").strip()},
            {"role": "user", "content": prompt[prompt.index("CONTEXT:"):]},
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
        )

        answer_text = response.choices[0].message.content.strip()
        citations = _extract_citations(answer_text)

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            prompt=prompt,
            context=context_block,
            provider=self.name,
            model=self.model,
            metadata={"context_chunks": used_chunk_ids, "truncation": truncation_metadata},
        )


class DeterministicFallbackGenerator(BaseGenerator):
    """Deterministic generator that works without external APIs.

    Strategy: echo the highest-ranked chunk as the answer with a citation.
    If no context is present, return the prescribed abstention string.
    """

    name = "fallback"

    def __init__(self, model: str = "stub-grounded") -> None:
        self.model = model

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
            abstain = "I don't have enough evidence to answer confidently."
            return GeneratedAnswer(
                answer=abstain,
                citations=[],
                prompt=prompt,
                context=context_block,
                provider=self.name,
                model=self.model,
                metadata={"context_chunks": used_chunk_ids, "truncation": truncation_metadata},
            )

        top_chunk = retrieved[0].chunk
        snippet = top_chunk.text.strip()
        answer = f"{snippet[:280].rstrip()} [{top_chunk.chunk_id}]"

        return GeneratedAnswer(
            answer=answer,
            citations=[top_chunk.chunk_id],
            prompt=prompt,
            context=context_block,
            provider=self.name,
            model=self.model,
            metadata={"context_chunks": used_chunk_ids, "truncation": truncation_metadata},
        )


def build_generator(settings: Settings, client: Any | None = None) -> BaseGenerator:
    """Factory that prefers the configured provider but falls back gracefully."""

    provider = (settings.models.llm_provider or "fallback").lower()
    model = settings.models.llm_model

    if provider == "openai":
        try:
            return OpenAIChatGenerator(model=model, client=client)
        except Exception:
            # Fall through to deterministic generator if OpenAI unavailable.
            pass

    return DeterministicFallbackGenerator(model="deterministic-fallback")


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

    import re

    pattern = re.compile(r"\[(chunk_[a-zA-Z0-9]+)\]")
    return list(dict.fromkeys(pattern.findall(text)))  # preserve order, drop dupes


__all__ = [
    "GeneratedAnswer",
    "assemble_context",
    "format_grounded_prompt",
    "BaseGenerator",
    "OpenAIChatGenerator",
    "DeterministicFallbackGenerator",
    "build_generator",
    "generate_answer",
    "assemble_context_with_budget",
]
