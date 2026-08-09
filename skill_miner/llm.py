"""Thin Anthropic API wrapper.

This is the ONLY module that talks to the API. Every outbound string passes
through redact() one more time — the session cache is already redacted at
parse time, but this second fence guarantees the "never send unredacted
transcript content" constraint even if a caller slips raw text in.
"""

from __future__ import annotations

import json
import os
import re

from .redaction import redact


class LLMError(RuntimeError):
    pass


def _client():
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("The 'anthropic' package is required: pip install anthropic") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMError("ANTHROPIC_API_KEY is not set; stages 2+ need API access.")
    return anthropic.Anthropic()


def complete(prompt: str, model: str, max_tokens: int = 1024, system: str | None = None) -> str:
    """Single-turn completion. Redacts the outbound prompt as a final fence."""
    client = _client()
    kwargs = {}
    if system:
        kwargs["system"] = redact(system)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": redact(prompt)}],
        **kwargs,
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def complete_json(prompt: str, model: str, max_tokens: int = 4096,
                  system: str | None = None, retries: int = 2):
    """Completion that must parse as JSON; retries with an error nudge."""
    attempt_prompt = prompt
    last_err = None
    for _ in range(retries + 1):
        text = complete(attempt_prompt, model, max_tokens=max_tokens, system=system)
        try:
            return _parse_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            attempt_prompt = (
                prompt + "\n\nYour previous reply was not valid JSON "
                f"({e}). Reply with ONLY a valid JSON document, no prose, no code fences.")
    raise LLMError(f"Model did not return valid JSON after retries: {last_err}")


def _parse_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start > 0:
        text = text[start:]
    return json.loads(text)
