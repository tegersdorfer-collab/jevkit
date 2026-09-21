"""
Kompositions-Rezepte: Jev beantwortet atomare Fragen, der Code setzt zusammen.

  count      – Jev kann nicht zählen: eine Noul pro Element, Summe hier.
  composite  – zusammengesetztes Urteil = gewichtete Summe normalisierter Scores
               (docs.typesafe.ai/patterns/composite-scoring). Gewichte sind Config.
  extract    – Jev generiert nicht: Kandidaten kommen aus Code (Regex, Parser), Jev
               wählt per Choice; eine "stated"-Noul sagt, ob der Wert überhaupt genannt
               wurde (docs.typesafe.ai/cookbooks/function_calling). Es gibt immer eine
               Option "keiner der Kandidaten", damit Jev nicht raten muss.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from jevkit.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jevkit.client import Decision
from jevkit.questions import Choice, Json, Noul, Question

NONE_OPTION = "__none__"


def count_questions(prefix: str, items: Sequence[Json], make: Callable[[Json], Noul]) -> dict[str, Noul]:
    return {f"{prefix}:{i}": make(item) for i, item in enumerate(items)}


def count(decision: Decision, prefix: str, threshold: float = 0.5) -> int:
    head = f"{prefix}:"
    return sum(1 for qid, a in decision.answers.items()
               if qid.startswith(head) and isinstance(a, NoulAnswer) and a.p >= threshold)


def composite(decision: Decision, weights: Mapping[str, float]) -> float:
    """Gewichtete Summe der normalisierten Scores, Gewichte werden auf 1 normiert."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("composite: Gewichte müssen positiv sein")
    acc = 0.0
    for qid, w in weights.items():
        a = decision[qid]
        if not isinstance(a, ScoreAnswer):
            raise TypeError(f"composite: {qid!r} ist {a.kind}, erwartet score")
        acc += (w / total) * a.normalized
    return acc


def extract_questions(qid: str, instructions: Json, candidates: Sequence[str],  # noqa: E501
                      stated: Json | None = None) -> dict[str, Question]:
    if not candidates:
        raise ValueError("extract_questions: mindestens ein Kandidat nötig")
    criteria: dict[str, Json] = {c: None for c in candidates}
    criteria[NONE_OPTION] = "None of the listed candidates applies"
    return {
        qid: Choice(instructions, criteria),
        f"{qid}:stated": Noul(stated or {"question": f"Does the content explicitly state a value for `{qid}`?",  # noqa: E501
                                          "field": qid}),
    }


def extract(decision: Decision, qid: str, *, min_confidence: float = 0.5, stated_at: float = 0.5) -> str | None:  # noqa: E501
    """Gewählter Kandidat oder None (nicht genannt, keiner passt, zu unsicher)."""
    stated = decision[f"{qid}:stated"]
    choice = decision[qid]
    if not isinstance(stated, NoulAnswer) or not isinstance(choice, ChoiceAnswer):
        raise TypeError(f"extract: {qid!r} braucht Choice + Noul")
    if stated.p < stated_at or choice.choice == NONE_OPTION or choice.confidence < min_confidence:
        return None
    return choice.choice
