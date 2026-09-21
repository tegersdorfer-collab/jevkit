"""
Typed answers. Every answer has `kind`, `value`, `p` (probability of the chosen value)
and `confidence` (0 = coin flip, 1 = certain). Per the API, Noul has no confidence — we
derive |p - 0.5| * 2 so the gate can treat all types the same.

`parse_answer` is the only place that touches raw API dicts. Anything broken becomes a
ValueError so the client can open the breaker (a changed schema shouldn't cost a
roundtrip on every turn).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NoulAnswer:
    p: float
    kind: str = field(default="noul", init=False)

    @property
    def value(self) -> bool:
        return self.p >= 0.5

    @property
    def confidence(self) -> float:
        return abs(self.p - 0.5) * 2

    def to_dict(self) -> dict:
        return {"type": "noul", "noul": self.p}


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    kind: str = field(default="choice", init=False)

    @property
    def value(self) -> str:
        return self.choice

    @property
    def p(self) -> float:
        return self.probabilities[self.choice]

    def to_dict(self) -> dict:
        return {"type": "choice", "choice": self.choice, "probabilities": dict(self.probabilities),
                "confidence": self.confidence}


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: float
    kind: str = field(default="score", init=False)

    @property
    def value(self) -> float:
        return self.score

    @property
    def level(self) -> int:
        return int(round(self.score))

    def _levels(self) -> list[int]:
        """Level indices from the `probabilities` keys. If the keys are numeric (e.g.
        from `legend`/API) they are sorted; otherwise (e.g. `{"low": .., "high": ..}`)
        we count them positionally in key order (0..n-1)."""
        keys = list(self.probabilities)
        try:
            return sorted(int(k) for k in keys)
        except (ValueError, TypeError):
            return list(range(len(keys)))

    def _key_for_level(self, level: int) -> str | None:
        keys = list(self.probabilities)
        try:
            for k in keys:
                if int(k) == level:
                    return k
            return None
        except (ValueError, TypeError):
            return keys[level] if 0 <= level < len(keys) else None

    @property
    def normalized(self) -> float:
        """Score on 0-1, regardless of whether the levels are 0- or 1-based."""
        levels = self._levels()
        span = levels[-1] - levels[0]
        return 0.0 if span == 0 else (self.score - levels[0]) / span

    @property
    def p(self) -> float:
        key = self._key_for_level(self.level)
        return self.probabilities.get(key, 0.0) if key is not None else 0.0

    def to_dict(self) -> dict:
        return {"type": "score", "score": self.score, "legend": dict(self.legend),
                "probabilities": dict(self.probabilities), "confidence": self.confidence}


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def _probs(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("probabilities must be an object")
    return {str(k): float(v) for k, v in raw.items()}


def parse_answer(raw: Any) -> Answer:
    """Raw API dict -> Answer. Raises ValueError on any deviation from the schema."""
    if not isinstance(raw, dict):
        raise ValueError(f"answer is not an object: {type(raw).__name__}")
    try:
        kind = raw["type"]
        if kind == "noul":
            return NoulAnswer(float(raw["noul"]))
        if kind == "choice":
            probs = _probs(raw["probabilities"])
            choice = str(raw["choice"])
            if choice not in probs:
                raise ValueError(f"choice {choice!r} missing from probabilities")
            return ChoiceAnswer(choice, probs, float(raw["confidence"]))
        if kind == "score":
            # legend is purely informational (display) and must never make the
            # decision fail: dict, list (-> {"0": .., "1": ..}) or missing/broken.
            raw_legend = raw.get("legend")
            if isinstance(raw_legend, dict):
                legend = {str(k): v for k, v in raw_legend.items()}
            elif isinstance(raw_legend, list):
                legend = {str(i): v for i, v in enumerate(raw_legend)}
            else:
                legend = {}
            return ScoreAnswer(float(raw["score"]), legend,
                               _probs(raw["probabilities"]), float(raw["confidence"]))
    except (KeyError, TypeError) as e:
        raise ValueError(f"answer incomplete: {e!r}") from e
    raise ValueError(f"unknown answer type {raw.get('type')!r}")
