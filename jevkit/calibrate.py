"""
Calibration from labeled outcomes. Bands are measured, not guessed: `suggest_bands`
finds the smallest confidence at which precision reaches a target (ACT: 95%, CONFIRM:
75% by default). Too little data -> conservative: `Bands(1.0, 1.0)`, i.e. only
confidence exactly 1.0 becomes ACT, everything below is ESCALATE (CONFIRM is then
unreachable - that's intentional as long as nothing has been measured).
Rerun after a model change (Decision.model).
"""
from __future__ import annotations

from collections.abc import Sequence

from jevkit.answers import NoulAnswer
from jevkit.gate import Bands
from jevkit.log import Record

Pairs = Sequence[tuple[float, bool]]


def _nonempty(pairs: Pairs) -> None:
    if not pairs:
        raise ValueError("no pairs")


def brier(pairs: Pairs) -> float:
    """pairs = (p, label). Mean squared error."""
    _nonempty(pairs)
    return sum((p - float(y)) ** 2 for p, y in pairs) / len(pairs)


def accuracy(pairs: Pairs) -> float:
    """pairs = (p, label). Hit rate at a 0.5 threshold."""
    _nonempty(pairs)
    return sum(1 for p, y in pairs if (p >= 0.5) == y) / len(pairs)


def ece(pairs: Pairs, bins: int = 10) -> float:
    """Expected Calibration Error over equal-width bins."""
    _nonempty(pairs)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    total = len(pairs)
    err = 0.0
    for b in buckets:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(1 for _, y in b if y) / len(b)
        err += len(b) / total * abs(conf - acc)
    return err


def pairs_by_question(records: Sequence[Record]) -> dict[str, list[tuple[float, bool]]]:
    """(confidence, correct) per question - only records with an outcome."""
    out: dict[str, list[tuple[float, bool]]] = {}
    for r in records:
        if r.correct is None:
            continue
        out.setdefault(r.qid, []).append((r.answer.confidence, r.correct))
    return out


def noul_pairs(records: Sequence[Record], qid: str) -> list[tuple[float, bool]]:
    """(p, label) for Brier/ECE of a Noul question. label = value if correct, else its opposite."""
    out: list[tuple[float, bool]] = []
    for r in records:
        if r.qid != qid or r.correct is None or not isinstance(r.answer, NoulAnswer):
            continue
        out.append((r.answer.p, r.answer.value if r.correct else not r.answer.value))
    return out


def _min_conf_for_precision(pairs: Pairs, target: float, min_n: int) -> float:
    """Smallest confidence c such that, among all pairs with conf >= c, precision >= target."""
    ordered = sorted(pairs, key=lambda t: t[0], reverse=True)
    best = 1.0
    hits = 0
    for i, (c, ok) in enumerate(ordered, start=1):
        hits += int(ok)
        # all pairs with the same confidence belong together
        if i < len(ordered) and ordered[i][0] == c:
            continue
        if i >= min_n and hits / i >= target:
            best = c
    return best


def suggest_bands(pairs: Pairs, *, act_precision: float = 0.95, escalate_precision: float = 0.75,
                  min_n: int = 5) -> Bands:
    if escalate_precision > act_precision:
        raise ValueError("escalate_precision must not exceed act_precision")
    if len(pairs) < min_n:
        return Bands(1.0, 1.0)
    act = _min_conf_for_precision(pairs, act_precision, min_n)
    esc = _min_conf_for_precision(pairs, escalate_precision, min_n)
    return Bands(act=act, escalate=min(esc, act))
