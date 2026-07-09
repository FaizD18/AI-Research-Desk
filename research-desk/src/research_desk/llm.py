"""Shared, disk-cached Anthropic client.

Every LLM call in the project goes through :func:`cached_call`. Nothing else
constructs an ``anthropic`` client. The cache key is a hash of everything that
affects the output — model, system prompt, user content, response schema, and
token limit — so a full-pipeline rerun with an unchanged prompt reads from disk
and costs ~$0. The API key comes from ``ANTHROPIC_API_KEY`` only; importing this
module never requires it, only *calling* into the API on a cache miss does.

Structured output uses the Anthropic SDK's ``messages.parse`` with a Pydantic
``output_format``, so responses are schema-validated at the API layer and the
model retries on mismatch. We disable thinking: categorization and boundary
selection are bounded, deterministic-ish tasks where reasoning tokens add cost
without improving a well-constrained JSON answer.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError

from research_desk import config

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a call cannot be satisfied even after a retry."""


@lru_cache(maxsize=1)
def _client():
    """Construct the Anthropic client lazily (only a real call needs the key)."""
    import anthropic

    return anthropic.Anthropic()


def _cache_key(model: str, system: str, user: str, schema_name: str, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "user": user,
            "schema": schema_name,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return config.LLM_CACHE_DIR / f"{key}.json"


def cached_call(
    system: str,
    user: str,
    response_model: type[M],
    *,
    model: str = config.LLM_MODEL,
    max_tokens: int = config.LLM_MAX_TOKENS,
) -> M:
    """Return a schema-validated structured response, reading disk cache first.

    On a cache miss, calls the API once, validates, and persists the raw JSON.
    A malformed response triggers exactly one retry before raising ``LLMError``
    so a single bad call is caught by the caller and skipped, never crashing a
    batch run.
    """
    key = _cache_key(model, system, user, response_model.__name__, max_tokens)
    path = _cache_path(key)
    if path.exists():
        return response_model.model_validate_json(path.read_text())

    client = _client()
    last_err: Exception | None = None
    for attempt in range(2):
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": user}],
            output_format=response_model,
        )
        parsed = response.parsed_output
        if parsed is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(parsed.model_dump_json())
            return parsed
        last_err = ValidationError.from_exception_data(response_model.__name__, [])
        log.warning("LLM returned no parseable output (attempt %d/2)", attempt + 1)

    raise LLMError(f"no valid {response_model.__name__} after retry: {last_err}")


# --- Extraction fallback -------------------------------------------------


class _Boundaries(BaseModel):
    """Two block indexes bounding a section, chosen from an enumerated outline."""

    start_index: int = Field(description="Index of the section's heading block")
    end_index: int = Field(description="Index of the next section's heading block")
    confidence: float = Field(ge=0.0, le=1.0)


_BOUNDARY_SYSTEM = (
    "You are a precise SEC-filing structure parser. You are given a numbered "
    "outline of candidate heading blocks from a 10-K. Return the block index "
    "where the requested section's heading appears (start_index) and the index "
    "of the very next top-level Item heading that follows it (end_index). "
    "Choose only from the provided indexes. Never invent text; you are selecting "
    "boundaries, not writing content."
)


def pick_section_boundaries(
    outline: list[dict], section: str
) -> tuple[int, int] | None:
    """LLM fallback for Item-boundary selection; returns ``None`` on any failure.

    The model only ever picks two integers from ``outline`` — it never sees or
    emits section prose — so hallucination cannot inject text into the pipeline.
    """
    user = (
        f"Section to locate: {section}\n\n"
        f"Candidate heading blocks (index, text):\n{json.dumps(outline, ensure_ascii=True)}"
    )
    try:
        result = cached_call(_BOUNDARY_SYSTEM, user, _Boundaries)
    except Exception:
        log.exception("LLM boundary selection failed")
        return None
    return result.start_index, result.end_index
