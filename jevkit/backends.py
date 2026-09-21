"""
Transport to Jev. Both official endpoints take the same body
{model, state, questions} and return {model, answers, usage}:

  TypeSafe direct:  POST https://api.typesafe.ai/v1/systemone      model "jev-latest"
  OpenRouter:       POST https://openrouter.ai/api/alpha/decisions model "~typesafe/jev-latest"

Retry belongs here (the backend knows the status codes); the overall time budget and
the circuit breaker belong in the client. Per the docs, 429/529 should be retried with
backoff; 520/522/524 (Cloudflare) have been observed in practice.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529})

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"


@dataclass(frozen=True)
class RawResponse:
    model: str
    answers: dict[str, Any]
    usage: dict[str, Any]


class BackendError(Exception):
    """Transport or HTTP error. `retryable` tells the caller whether retrying makes sense."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class Backend(Protocol):
    async def ask(self, state: Any, questions: dict[str, dict], *, model: str | None,
                  timeout_s: float) -> RawResponse: ...


class HTTPBackend:
    """Generic HTTP transport; TypeSafeBackend/OpenRouterBackend only set URL, model, env var."""

    url: str = TYPESAFE_URL
    default_model: str = "jev-latest"
    env_var: str = "TYPESAFE_API_KEY"

    def __init__(self, api_key: str | None = None, *, url: str | None = None, model: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, retries: int = 2,
                 backoff_s: float = 0.5) -> None:
        key = api_key or os.environ.get(self.env_var, "")
        if not key:
            raise BackendError(f"no API key: pass api_key or set ${self.env_var}", retryable=False)
        self.api_key = key
        if url:
            self.url = url
        if model:
            self.default_model = model
        self.transport = transport
        self.retries = max(0, retries)
        self.backoff_s = backoff_s

    async def ask(self, state: Any, questions: dict[str, dict], *, model: str | None = None,
                  timeout_s: float = 10.0) -> RawResponse:
        body = {"model": model or self.default_model, "state": state, "questions": questions}
        payload = json.dumps(body, ensure_ascii=False).encode()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: BackendError | None = None
        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout_s, transport=self.transport) as client:
                    resp = await client.post(self.url, content=payload, headers=headers)
            except httpx.HTTPError as e:
                last = BackendError(f"network error: {e!r}", retryable=True)
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        answers = data["answers"]
                        if not isinstance(answers, dict):
                            raise TypeError(f"answers is not a dict: {type(answers)}")
                        return RawResponse(str(data.get("model", "")), answers, data.get("usage") or {})
                    except (ValueError, KeyError, TypeError) as e:
                        last = BackendError(f"broken response: {e!r}", retryable=False)
                        raise last from e
                last = BackendError(f"HTTP {resp.status_code}: {resp.text[:200]}",
                                    retryable=resp.status_code in RETRY_STATUS)
                if not last.retryable:
                    raise last
            if attempt < self.retries:
                await asyncio.sleep(self.backoff_s * (2 ** attempt))
        assert last is not None
        raise last


class TypeSafeBackend(HTTPBackend):
    url = TYPESAFE_URL
    default_model = "jev-latest"
    env_var = "TYPESAFE_API_KEY"


class OpenRouterBackend(HTTPBackend):
    url = OPENROUTER_URL
    default_model = "~typesafe/jev-latest"
    env_var = "OPENROUTER_API_KEY"


Handler = Callable[[Any, dict[str, dict]], dict[str, Any]]


@dataclass
class StaticBackend:
    """For tests: fixed answers or a handler (state, questions) -> answers. Records calls."""

    answers: dict[str, Any] | Handler
    model: str = "static"
    calls: list[tuple[Any, dict[str, dict]]] = field(default_factory=list)

    async def ask(self, state: Any, questions: dict[str, dict], *, model: str | None = None,
                  timeout_s: float = 1.0) -> RawResponse:
        self.calls.append((state, questions))
        answers = self.answers(state, questions) if callable(self.answers) else dict(self.answers)
        return RawResponse(self.model, answers, {"input_tokens": 0, "output_tokens": 0})
