"""
Ein Call pro State, alle Fragen parallel. Der Client besitzt das Gesamt-Zeitbudget
(asyncio.timeout über Connect+Read+Parse), den Circuit-Breaker und den Cache.

Vertrag: `decide()` liefert entweder vollständige, typgeprüfte Antworten oder wirft
JevUnavailable — nie halbe Ergebnisse. Caller-Bugs (leere Fragen, nicht
serialisierbarer State) fliegen als ValueError/TypeError VOR dem Netz und öffnen
den Breaker nicht.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from jevkit.answers import Answer, parse_answer
from jevkit.backends import Backend, BackendError, RawResponse
from jevkit.cache import MemoryCache, cache_key
from jevkit.questions import Question

log = logging.getLogger(__name__)


class JevUnavailable(Exception):
    """Jev nicht nutzbar (Cooldown, Timeout, Transport/HTTP-Fehler, kaputte Antwort)."""


@dataclass(frozen=True)
class Decision:
    answers: dict[str, Answer]
    model: str
    usage: dict[str, Any]
    cached: bool
    latency_s: float

    def __getitem__(self, qid: str) -> Answer:
        return self.answers[qid]

    def __contains__(self, qid: object) -> bool:
        return qid in self.answers


class Client:
    def __init__(self, backend: Backend, *, model: str | None = None, timeout_s: float = 3.0,
                 cooldown_s: float = 120.0, cache: MemoryCache | None = None,
                 expected_model: str | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.backend = backend
        self.model = model
        self.timeout_s = timeout_s
        self.cooldown_s = cooldown_s
        self.cache = cache
        self.expected_model = expected_model
        self._clock = clock
        self._down_until = 0.0

    @property
    def available(self) -> bool:
        return self._clock() >= self._down_until

    def reset(self) -> None:
        self._down_until = 0.0

    def _typed(self, raw: RawResponse, questions: Mapping[str, Question]) -> dict[str, Answer]:
        out: dict[str, Answer] = {}
        for qid, q in questions.items():
            if qid not in raw.answers:
                raise ValueError(f"Antwort für {qid!r} fehlt")
            ans = parse_answer(raw.answers[qid])
            if ans.kind != q.kind:
                raise ValueError(f"{qid!r}: Frage ist {q.kind}, Antwort ist {ans.kind}")
            out[qid] = ans
        return out

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> Decision:
        if not questions:
            raise ValueError("mindestens eine Frage nötig")
        payload = {qid: q.payload() for qid, q in questions.items()}
        key = cache_key(self.model, state, payload)   # json.dumps → TypeError bei Caller-Bug, vor dem Netz
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return Decision(self._typed(hit, questions), hit.model, hit.usage, True, 0.0)
        if not self.available:
            raise JevUnavailable("Jev im Cooldown")
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout_s):
                raw = await self.backend.ask(state, payload, model=self.model, timeout_s=self.timeout_s)
            answers = self._typed(raw, questions)
        except (TimeoutError, BackendError, ValueError, TypeError, KeyError) as e:
            self._down_until = self._clock() + self.cooldown_s
            log.warning("Jev nicht nutzbar, %.0fs Cooldown: %r", self.cooldown_s, e)
            raise JevUnavailable(str(e)) from e
        latency = time.perf_counter() - t0
        if self.expected_model and not raw.model.startswith(self.expected_model):
            log.warning(
                "Jev-Modell %r statt erwartet %r — Bänder neu kalibrieren",
                raw.model,
                self.expected_model,
            )
        if self.cache is not None:
            self.cache.put(key, raw)
        return Decision(answers, raw.model, raw.usage, False, latency)
