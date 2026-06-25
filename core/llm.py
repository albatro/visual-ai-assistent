"""LLM streaming wrapper shared by all rendering branches.

Provides:
  * make_client()       — returns AsyncOpenAI or None (when OPENAI_API_KEY unset)
  * ask()               — one-shot, returns the full reply text
  * stream_sentences()  — async generator that yields (sentence, elapsed_ms, is_tail)
                          as each completed sentence becomes available

stream_sentences lets a branch start TTS+render on sentence 1 while sentences
2..N are still being generated, cutting time-to-first-voice from ~7s to ~2s.
"""
import os
import re
import time
from typing import AsyncIterator

from openai import AsyncOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 250
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SYSTEM = (
    "Ты — Пётр Первый, основатель Санкт-Петербурга, российский император. "
    "Отвечай кратко (2-4 предложения), от первого лица, как Пётр I в речи, "
    "но современным русским языком. Без эмодзи и markdown."
)

SENTENCE_END = re.compile(r"[.!?…](?=\s|$)")


def make_client() -> AsyncOpenAI | None:
    """Return an AsyncOpenAI client if OPENAI_API_KEY is set; else None.
    Branches should treat None as 'ask mode disabled, direct-text mode still works'."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    return AsyncOpenAI()


async def ask(
    client: AsyncOpenAI,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str = DEFAULT_SYSTEM,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


async def stream_sentences(
    client: AsyncOpenAI,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str = DEFAULT_SYSTEM,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> AsyncIterator[tuple[str, float, bool]]:
    """Stream completion and yield (sentence, elapsed_ms_since_request_start, is_tail).
    The trailing fragment (text without terminating punctuation) is delivered last
    with is_tail=True."""
    t0 = time.time()
    stream = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    buffer = ""
    async for ev in stream:
        delta = ev.choices[0].delta.content or ""
        if not delta:
            continue
        buffer += delta
        while True:
            m = SENTENCE_END.search(buffer)
            if not m:
                break
            cut = m.end()
            sentence = buffer[:cut].strip()
            buffer = buffer[cut:].lstrip()
            if sentence:
                yield sentence, (time.time() - t0) * 1000, False

    tail = buffer.strip()
    if tail:
        yield tail, (time.time() - t0) * 1000, True
