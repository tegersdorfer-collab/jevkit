"""
Three bands instead of one threshold. The answer says WHAT, the band says WHETHER the
code may act on it:

  ACT       - execute automatically
  CONFIRM   - execute only with confirmation / a second source (e.g. local LLM, human)
  ESCALATE  - don't rely on the answer; hand off to a human or a reasoning model

Everything runs on `confidence` (0 = coin flip, 1 = certain). For Nouls that's
|p - 0.5| * 2, so the cookbook band "0.30-0.70 = uncertain" corresponds to
escalate = 0.4. Bands grow with the severity of the action (docs.typesafe.ai/confidence:
"different actions should be gated at different levels").

Jev guarantees no invariants (P(A)+P(not A) != 1). `check_pairs` finds questions whose
negation contradicts them - the caller then demotes those.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum

from jevkit.answers import Answer, NoulAnswer
from jevkit.client import Decision


class Band(StrEnum):
    ACT = "act"
    CONFIRM = "confirm"
    ESCALATE = "escalate"


class Severity(Enum):
    READ = "read"              # read/display only
    REVERSIBLE = "reversible"  # changes something, but can be undone
    DESTRUCTIVE = "destructive"  # delete, send, pay


@dataclass(frozen=True)
class Bands:
    act: float        # confidence >= act -> ACT
    escalate: float   # confidence <  escalate -> ESCALATE; in between is CONFIRM

    def __post_init__(self) -> None:
        if not 0.0 <= self.escalate <= self.act <= 1.0:
            raise ValueError(f"expected 0 <= escalate <= act <= 1, got {self}")


DEFAULT_BANDS: dict[Severity, Bands] = {
    Severity.READ: Bands(act=0.4, escalate=0.2),          # Noul: p >= 0.70 acts
    Severity.REVERSIBLE: Bands(act=0.6, escalate=0.3),    # Noul: p >= 0.80
    Severity.DESTRUCTIVE: Bands(act=0.8, escalate=0.5),   # Noul: p >= 0.90
}


def band(answer: Answer, bands: Bands) -> Band:
    c = answer.confidence
    if c >= bands.act:
        return Band.ACT
    if c < bands.escalate:
        return Band.ESCALATE
    return Band.CONFIRM


def demote(b: Band) -> Band:
    return {Band.ACT: Band.CONFIRM, Band.CONFIRM: Band.ESCALATE}.get(b, Band.ESCALATE)


def consistent(p_a: float, p_not_a: float, tol: float = 0.25) -> bool:
    """P(A) and 1 - P(not A) should be close to each other."""
    return abs(p_a - (1.0 - p_not_a)) <= tol


def check_pairs(decision: Decision, pairs: Sequence[tuple[str, str]], tol: float = 0.25) -> set[str]:
    """Returns the IDs of the original questions whose negation contradicts them. Nouls only."""
    bad: set[str] = set()
    for qid, neg in pairs:
        a, n = decision[qid], decision[neg]
        if not isinstance(a, NoulAnswer) or not isinstance(n, NoulAnswer):
            raise TypeError(f"check_pairs needs Nouls, {qid!r}/{neg!r} are {a.kind}/{n.kind}")
        if not consistent(a.p, n.p, tol):
            bad.add(qid)
    return bad
