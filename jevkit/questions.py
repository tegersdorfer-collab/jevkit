"""
The three Jev primitives on the request side.

Rules from the TypeSafe docs (docs.typesafe.ai/primitives): Choice <= 255 options,
Score 2-10 ordered levels, instructions and criteria may be strings or JSON
structures. Validation happens here so an error in the question catalog surfaces
at import time instead of as a 422 in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Json = str | int | float | bool | None | dict[str, Any] | list[Any]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


@dataclass(frozen=True)
class Noul:
    """Yes/no question -> P(yes). `criteria` = {"true": ..., "false": ...} sharpens the boundary."""
    instructions: Json
    criteria: dict[str, Json] | None = None
    kind: str = field(default="noul", init=False)

    def payload(self) -> dict:
        d: dict = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            d["criteria"] = self.criteria
        return d


@dataclass(frozen=True)
class Choice:
    """One option out of a fixed set -> option + distribution + confidence."""
    instructions: Json
    criteria: dict[str, Json]
    kind: str = field(default="choice", init=False)

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if not 1 <= n <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"Choice needs 1-{MAX_CHOICE_OPTIONS} options, has {n}")

    def payload(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True)
class Score:
    """Position on ordered levels -> expected value + distribution + confidence."""
    instructions: Json
    criteria: tuple[Json, ...]
    kind: str = field(default="score", init=False)

    def __init__(self, instructions: Json, criteria: list[Json] | tuple[Json, ...]) -> None:
        levels = tuple(criteria)
        if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
            raise ValueError(f"Score needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, has {len(levels)}")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "criteria", levels)
        object.__setattr__(self, "kind", "score")

    def payload(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Noul | Choice | Score
