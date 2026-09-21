"""
Jev emulation over a text LLM (e.g. Ollama) - for local/private decisions and as a
fallback. UNCALIBRATED: Choice/Score return one-hot distributions with confidence 1,
Noul a percentage estimated by the LLM. That's why this backend reports itself as
"prompt:<name>", and the router demotes bands: either because `calibrated_model`
doesn't match, or - when this backend answers a CLOUD spec as a fallback -
automatically, regardless of `calibrated_model` (see `registry.Router._verdicts`).
One question per prompt (text LLMs don't answer fan-outs in isolation). Requests run
through an `asyncio.Semaphore(concurrency)` so a local LLM isn't overloaded by fan-outs.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from jevkit.backends import BackendError, RawResponse

_INT = re.compile(r"-?\d+")
_NUM = re.compile(r"\b(\d+(?:\.\d+)?)\s*%?")


def _fmt(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def build_prompt(state: Any, qid: str, question: dict) -> str:
    head = (
        "Evaluate the CONTENT and answer ONE question. "
        "Reply with the answer only, no explanation.\n\n"
        f"CONTENT:\n{_fmt(state)}\n\n"
        f"QUESTION ({qid}): {_fmt(question.get('instructions'))}\n"
    )
    kind = question["type"]
    crit = question.get("criteria")
    if kind == "noul":
        tail = ""
        if isinstance(crit, dict):
            tail = (
                f"YES means: {_fmt(crit.get('true'))}\n"
                f"NO means: {_fmt(crit.get('false'))}\n"
            )
        return (
            head + tail +
            "Reply with a single integer from 0 to 100 = probability (%) "
            "that the answer is YES."
        )
    if kind == "choice":
        opts = "\n".join(
            f"- {k}" + (f": {_fmt(v)}" if v is not None else "")
            for k, v in crit.items()
        )
        return (
            head + f"OPTIONS:\n{opts}\n"
            "Reply with exactly one option name from the list."
        )
    if kind == "score":
        levels = "\n".join(f"{i}: {_fmt(lvl)}" for i, lvl in enumerate(crit))
        return (
            head + f"LEVELS:\n{levels}\n"
            "Reply with the level number only."
        )
    raise BackendError(f"unknown question type {kind!r}", retryable=False)


def parse_reply(question: dict, reply: str) -> dict:
    kind = question["type"]
    text = reply.strip()
    if kind == "noul":
        m = _NUM.search(text)
        if not m:
            raise BackendError(
                f"Noul answer without a number: {text[:80]!r}", retryable=False
            )
        num = m.group(1)
        value = float(num)
        # If the match contains a dot, it's a fraction (0.7 -> 0.7);
        # otherwise a percentage (85 -> 0.85, 7% -> 0.07, 1 -> 0.01).
        frac = value if "." in num else value / 100
        return {"type": "noul", "noul": min(max(frac, 0.0), 1.0)}
    if kind == "choice":
        options = list(question["criteria"])
        low = text.lower()
        hit = next((o for o in options if o.lower() == low), None) or (
            next((o for o in sorted(options, key=len, reverse=True)
                  if re.search(rf"\b{re.escape(o.lower())}\b", low)), None)
        )
        if hit is None:
            raise BackendError(
                f"Choice answer matches no option: {text[:80]!r}",
                retryable=False
            )
        return {
            "type": "choice", "choice": hit,
            "probabilities": {o: 1.0 if o == hit else 0.0 for o in options},
            "confidence": 1.0
        }
    if kind == "score":
        levels = list(question["criteria"])
        m = _INT.search(text)
        if not m or not 0 <= int(m.group()) < len(levels):
            raise BackendError(
                f"Score answer is not a valid level: {text[:80]!r}",
                retryable=False
            )
        idx = int(m.group())
        return {
            "type": "score", "score": float(idx),
            "legend": {str(i): lvl for i, lvl in enumerate(levels)},
            "probabilities": {
                str(i): 1.0 if i == idx else 0.0
                for i in range(len(levels))
            },
            "confidence": 1.0
        }
    raise BackendError(f"unknown question type {kind!r}", retryable=False)


class PromptBackend:
    def __init__(self, ask: Callable[[str], Awaitable[str]], *, name: str = "local",
                 concurrency: int = 4) -> None:
        self._ask = ask
        self.model = f"prompt:{name}"
        self._sem = asyncio.Semaphore(concurrency)

    async def ask(self, state: Any, questions: dict[str, dict], *, model: str | None = None,
                  timeout_s: float = 30.0) -> RawResponse:
        async def one(qid: str, q: dict) -> tuple[str, dict]:
            async with self._sem:
                reply = await self._ask(build_prompt(state, qid, q))
            return qid, parse_reply(q, reply)

        pairs = await asyncio.gather(*(one(qid, q) for qid, q in questions.items()))
        return RawResponse(self.model, dict(pairs), {"input_tokens": 0, "output_tokens": 0})
