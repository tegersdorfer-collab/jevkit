"""
Drei Bänder statt einer Schwelle. Antwort sagt WAS, Band sagt OB der Code handeln darf:

  ACT       – automatisch ausführen
  CONFIRM   – ausführen nur mit Bestätigung / zweiter Quelle (z.B. lokales LLM, Mensch)
  ESCALATE  – nicht auf die Antwort verlassen; an Mensch oder Reasoning-Modell

Alles läuft über `confidence` (0 = Münzwurf, 1 = sicher). Für Nouls ist das |p − 0,5|·2,
d.h. das Cookbook-Band "0,30–0,70 = unsicher" entspricht escalate = 0,4. Bänder wachsen
mit der Schwere der Aktion (docs.typesafe.ai/confidence: "different actions should be
gated at different levels").

Jev garantiert keine Invarianten (P(A)+P(¬A) ≠ 1). `check_pairs` findet Fragen, deren
Gegenfrage widerspricht — der Aufrufer demotet die dann.
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
    READ = "read"              # nur lesen/anzeigen
    REVERSIBLE = "reversible"  # ändern, aber rückgängig machbar
    DESTRUCTIVE = "destructive"  # löschen, senden, bezahlen


@dataclass(frozen=True)
class Bands:
    act: float        # confidence >= act → ACT
    escalate: float   # confidence <  escalate → ESCALATE; dazwischen CONFIRM

    def __post_init__(self) -> None:
        if not 0.0 <= self.escalate <= self.act <= 1.0:
            raise ValueError(f"erwartet 0 <= escalate <= act <= 1, bekommen {self}")


DEFAULT_BANDS: dict[Severity, Bands] = {
    Severity.READ: Bands(act=0.4, escalate=0.2),          # Noul: p >= 0.70 handelt
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
    """P(A) und 1 − P(¬A) sollten nahe beieinander liegen."""
    return abs(p_a - (1.0 - p_not_a)) <= tol


def check_pairs(decision: Decision, pairs: Sequence[tuple[str, str]], tol: float = 0.25) -> set[str]:
    """Gibt die IDs der Erstfragen zurück, deren Gegenfrage widerspricht. Nur für Nouls."""
    bad: set[str] = set()
    for qid, neg in pairs:
        a, n = decision[qid], decision[neg]
        if not isinstance(a, NoulAnswer) or not isinstance(n, NoulAnswer):
            raise TypeError(f"check_pairs braucht Nouls, {qid!r}/{neg!r} sind {a.kind}/{n.kind}")
        if not consistent(a.p, n.p, tol):
            bad.add(qid)
    return bad
