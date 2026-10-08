"""Structured LLM call with a cross-model fallback chain (trimmed from the speech agent's llm.py).

404/400 -> next model; 429 -> wait for retry-after, then retry; 5xx/timeout -> backoff;
output that fails the pydantic schema -> next model. All models failing raises LLMUnavailable."""
import asyncio
import json
import random
from typing import TypeVar

import groq
from pydantic import BaseModel, ValidationError

from triage.config import settings

T = TypeVar("T", bound=BaseModel)
_client: groq.AsyncGroq | None = None


class LLMUnavailable(RuntimeError):
    pass


def get_client() -> groq.AsyncGroq:
    global _client
    if _client is None:
        _client = groq.AsyncGroq(api_key=settings.groq_api_key, max_retries=0,
                                 timeout=settings.llm_timeout_s)
    return _client


def _extra(model: str) -> dict:
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    if model.startswith("qwen/"):
        return {"reasoning_format": "hidden"}
    return {}


def _retryable(e: Exception) -> bool:
    return (isinstance(e, (groq.APITimeoutError, groq.APIConnectionError, groq.RateLimitError))
            or (isinstance(e, groq.APIStatusError) and e.status_code >= 500))


def _delay(e: Exception, attempt: int) -> float:
    resp = getattr(e, "response", None)
    ra = resp.headers.get("retry-after") if resp is not None else None
    try:
        if ra is not None:
            return min(float(ra), settings.llm_max_wait_s) + random.random() * 0.25
    except ValueError:
        pass
    return min(8.0, 0.5 * 2 ** attempt) + random.random() * 0.25


async def structured_call(system: str, user: str, schema: type[T], temperature: float = 0.0) -> T:
    fmt = {"type": "json_schema", "json_schema": {
        "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}
    errors = []
    for model in settings.llm_models:
        for attempt in range(settings.llm_max_retries + 1):
            try:
                r = await get_client().chat.completions.create(
                    model=model, temperature=temperature, response_format=fmt,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    **_extra(model))
                return schema.model_validate(json.loads(r.choices[0].message.content))
            except (ValidationError, json.JSONDecodeError):
                errors.append(f"{model}: invalid output")
                break
            except Exception as e:
                errors.append(f"{model}: {type(e).__name__}")
                if not _retryable(e) or attempt == settings.llm_max_retries:
                    break
                await asyncio.sleep(_delay(e, attempt))
    raise LLMUnavailable("; ".join(errors))
