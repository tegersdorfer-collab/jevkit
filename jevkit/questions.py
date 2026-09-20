"""
Die drei Jev-Primitive auf der Request-Seite.

Regeln aus der TypeSafe-Doku (docs.typesafe.ai/primitives): Choice ≤ 255 Optionen,
Score 2–10 geordnete Level, Instructions und Criteria dürfen Strings oder
JSON-Strukturen sein. Die Validierung passiert hier, damit ein Fehler im
Fragenkatalog beim Import auffällt und nicht als 422 im Betrieb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Json = str | int | float | bool | None | dict[str, Any] | list[Any]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


@dataclass(frozen=True)
class Noul:
    """Ja/Nein-Frage → P(ja). `criteria` = {"true": ..., "false": ...} schärft die Grenze."""
    instructions: Json
    criteria: dict[str, Json] | None = None
    kind: str = "noul"

    def payload(self) -> dict:
        d: dict = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            d["criteria"] = self.criteria
        return d


@dataclass(frozen=True)
class Choice:
    """Eine Option aus einer festen Menge → Option + Verteilung + Confidence."""
    instructions: Json
    criteria: dict[str, Json]
    kind: str = "choice"

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if not 1 <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"Choice braucht 1–{MAX_CHOICE_OPTIONS} Optionen, hat {n}")

    def payload(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True)
class Score:
    """Position auf geordneten Leveln → Erwartungswert + Verteilung + Confidence."""
    instructions: Json
    criteria: tuple[Json, ...]
    kind: str = "score"

    def __init__(self, instructions: Json, criteria: list[Json] | tuple[Json, ...]) -> None:
        levels = tuple(criteria)
        if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
            raise ValueError(f"Score braucht {MIN_SCORE_LEVELS}–{MAX_SCORE_LEVELS} Level, hat {len(levels)}")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "criteria", levels)
        object.__setattr__(self, "kind", "score")

    def payload(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score
