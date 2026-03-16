"""Grounded answer generation utilities.

This module keeps answer generation provider‑agnostic while enforcing
grounding requirements:

* Retrieve context is assembled with a character budget to avoid overly
  long prompts.
* Prompts require chunk‑level citations (``[chunk_id]``) and abstention
  when evidence is insufficient.
* A provider interface allows plugging in OpenAI or other chat models,
  with a deterministic fallback that works without external API keys.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

from .config import Settings
from .schemas import GeneratedAnswer, RetrievalResult

DEFAULT_CONTEXT_CHARS = 2_400


# ---------------------------------------------------------------------------
# Context assembly and prompt formatting
# ---------------------------------------------------------------------------


def assemble_context(results: Sequence[RetrievalResult], max_chars: int = DEFAULT_CONTEXT_CHARS) -> str:
    """Format retrieved chunks into a prompt-ready context block.

    Chunks are appended in rank order until ``max_chars`` is reached. Each
    block includes the chunk id (for citations), title, and source so models
    can reference them explicitly.
    """

    lines: List[str] = []
    remaining = max_chars

    for res in results:
        chunk = res.chunk
        header = f"[{chunk.chunk_id}] {chunk.title or 'Untitled'} — {chunk.source}"
        body = chunk.text.strip()
        entry = f"{header}\n{body}"

        entry_len = len(entry) + 2  # include spacing
        if entry_len > remaining:
            break

        lines.append(entry)
        remaining -= entry_len

    return "\n\n".join(lines)


def format_grounded_prompt(
    question: str,
    retrieved: Sequence[RetrievalResult],
    *,
    max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    system_instruction: Optional[str] = None,
) -> str:
    """Build the full prompt for grounded generation.

    The default instruction enforces:
    * cite sources using ``[chunk_id]``
    * avoid fabrication; abstain with a fixed sentence when unsure
    * concise, direct answers
    """

    context_block = assemble_context(retrieved, max_chars=max_context_chars)

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
        prompt = format_grounded_prompt(
            question, retrieved, max_context_chars=max_context_chars, system_instruction=system_instruction
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
            context=assemble_context(retrieved, max_context_chars),
            provider=self.name,
            model=self.model,
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
        prompt = format_grounded_prompt(
            question, retrieved, max_context_chars=max_context_chars, system_instruction=system_instruction
        )

        context_block = assemble_context(retrieved, max_context_chars)

        if not retrieved:
            abstain = "I don't have enough evidence to answer confidently."
            return GeneratedAnswer(
                answer=abstain,
                citations=[],
                prompt=prompt,
                context=context_block,
                provider=self.name,
                model=self.model,
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
]
