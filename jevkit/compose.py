"""
Composition recipes: Jev answers atomic questions, the code assembles them.

  count      - Jev can't count: one Noul per item, summed here.
  composite  - a composite verdict = weighted sum of normalized scores
               (docs.typesafe.ai/patterns/composite-scoring). Weights are config.
  extract    - Jev doesn't generate: candidates come from code (regex, parser), Jev
               picks via Choice; a "stated" Noul says whether the value was mentioned
               at all (docs.typesafe.ai/cookbooks/function_calling). There is always a
               "none of the candidates" option so Jev never has to guess.
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
    """Weighted sum of the normalized scores; weights are normalized to 1."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("composite: weights must be positive")
    acc = 0.0
    for qid, w in weights.items():
        a = decision[qid]
        if not isinstance(a, ScoreAnswer):
            raise TypeError(f"composite: {qid!r} is {a.kind}, expected score")
        acc += (w / total) * a.normalized
    return acc


def extract_questions(
    qid: str,
    instructions: Json,
    candidates: Sequence[str],
    stated: Json | None = None,
) -> dict[str, Question]:
    if not candidates:
        raise ValueError("extract_questions: at least one candidate is required")
    criteria: dict[str, Json] = {c: None for c in candidates}
    criteria[NONE_OPTION] = "None of the listed candidates applies"
    default_stated = {
        "question": f"Does the content explicitly state a value for `{qid}`?",
        "field": qid,
    }
    return {
        qid: Choice(instructions, criteria),
        f"{qid}:stated": Noul(stated or default_stated),
    }


def extract(
    decision: Decision,
    qid: str,
    *,
    min_confidence: float = 0.5,
    stated_at: float = 0.5,
) -> str | None:
    """Chosen candidate or None (not mentioned, none fits, too uncertain)."""
    stated = decision[f"{qid}:stated"]
    choice = decision[qid]
    if not isinstance(stated, NoulAnswer) or not isinstance(choice, ChoiceAnswer):
        raise TypeError(f"extract: {qid!r} needs a Choice + Noul")
    if stated.p < stated_at or choice.choice == NONE_OPTION or choice.confidence < min_confidence:
        return None
    return choice.choice
