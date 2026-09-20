"""
Typisierte Antworten. Jede Antwort hat `kind`, `value`, `p` (Wahrscheinlichkeit des
gewählten Werts) und `confidence` (0 = Münzwurf, 1 = sicher). Noul hat laut API keine
Confidence — wir leiten |p − 0,5|·2 ab, damit das Gate alle Typen gleich behandelt.

`parse_answer` ist die einzige Stelle, die rohe API-Dicts anfasst. Alles Kaputte wird
ValueError, damit der Client den Breaker öffnen kann (ein geändertes Schema soll nicht
jeden Turn erneut den Roundtrip kosten).
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
        keys = self.legend or self.probabilities
        return sorted(int(k) for k in keys)

    @property
    def normalized(self) -> float:
        """Score auf 0–1, unabhängig davon, ob die Level 0- oder 1-basiert sind."""
        levels = self._levels()
        span = levels[-1] - levels[0]
        return 0.0 if span == 0 else (self.score - levels[0]) / span

    @property
    def p(self) -> float:
        return self.probabilities.get(str(self.level), 0.0)

    def to_dict(self) -> dict:
        return {"type": "score", "score": self.score, "legend": dict(self.legend),
                "probabilities": dict(self.probabilities), "confidence": self.confidence}


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def _probs(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("probabilities muss ein Objekt sein")
    return {str(k): float(v) for k, v in raw.items()}


def parse_answer(raw: Any) -> Answer:
    """Rohes API-Dict → Answer. Wirft ValueError bei jeder Abweichung vom Schema."""
    if not isinstance(raw, dict):
        raise ValueError(f"Antwort ist kein Objekt: {type(raw).__name__}")
    try:
        kind = raw["type"]
        if kind == "noul":
            return NoulAnswer(float(raw["noul"]))
        if kind == "choice":
            probs = _probs(raw["probabilities"])
            choice = str(raw["choice"])
            if choice not in probs:
                raise ValueError(f"choice {choice!r} fehlt in probabilities")
            return ChoiceAnswer(choice, probs, float(raw["confidence"]))
        if kind == "score":
            legend = raw.get("legend") or {}
            if not isinstance(legend, dict):
                raise ValueError("legend muss ein Objekt sein")
            return ScoreAnswer(float(raw["score"]), {str(k): v for k, v in legend.items()},
                               _probs(raw["probabilities"]), float(raw["confidence"]))
    except (KeyError, TypeError) as e:
        raise ValueError(f"Antwort unvollständig: {e!r}") from e
    raise ValueError(f"unbekannter Antworttyp {raw.get('type')!r}")
