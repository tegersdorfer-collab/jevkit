import pytest

from jevkit.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jevkit.client import Decision
from jevkit.compose import NONE_OPTION, composite, count, count_questions, extract, extract_questions
from jevkit.questions import Choice, Noul


def _d(answers):
    return Decision(answers, "m", {}, False, 0.0)


def test_count_questions_and_count():
    items = ["Milch kaufen", "Zahnarzt anrufen", "Joggen"]
    qs = count_questions(
        "chore",
        items,
        lambda it: Noul({"item": it, "question": "Is this a household chore?"}),
    )
    assert list(qs) == ["chore:0", "chore:1", "chore:2"]
    assert qs["chore:1"].instructions == {
        "item": "Zahnarzt anrufen",
        "question": "Is this a household chore?",
    }
    d = _d({
        "chore:0": NoulAnswer(0.9),
        "chore:1": NoulAnswer(0.4),
        "chore:2": NoulAnswer(0.55),
        "x": NoulAnswer(1.0),
    })
    assert count(d, "chore") == 2
    assert count(d, "chore", threshold=0.8) == 1
    assert count(d, "nothing") == 0


def test_composite_weighted_sum():
    s = lambda score: ScoreAnswer(score, {"0": "a", "1": "b", "2": "c", "3": "d", "4": "e"},  # noqa: E731
                                  {str(i): 0.2 for i in range(5)}, 0.5)
    d = _d({"python": s(4.0), "lead": s(2.0), "design": s(0.0)})
    assert abs(composite(d, {"python": 0.5, "lead": 0.3, "design": 0.2}) - (0.5 * 1.0 + 0.3 * 0.5)) < 1e-9
    assert abs(composite(d, {"python": 2, "lead": 2}) - 0.75) < 1e-9   # weights are normalized
    with pytest.raises(TypeError):
        composite(_d({"n": NoulAnswer(0.5)}), {"n": 1.0})
    with pytest.raises(ValueError):
        composite(d, {})


def test_extract_questions_form():
    qs = extract_questions("due", "Which of these dates is the deadline the user mentions?",
                           ["2026-09-24", "2026-10-01"])
    assert set(qs) == {"due", "due:stated"}
    q = qs["due"]
    assert isinstance(q, Choice)
    assert list(q.criteria) == ["2026-09-24", "2026-10-01", NONE_OPTION]
    assert q.criteria[NONE_OPTION] is not None
    assert isinstance(qs["due:stated"], Noul) and "due" in str(qs["due:stated"].instructions)
    custom = extract_questions("amt", "?", ["1"], stated="Does the user mention an amount?")
    assert custom["amt:stated"].instructions == "Does the user mention an amount?"
    with pytest.raises(ValueError):
        extract_questions("x", "?", [])


def test_extract_resolution():
    ok = _d({
        "due": ChoiceAnswer(
            "2026-09-24",
            {"2026-09-24": 0.9, "2026-10-01": 0.05, NONE_OPTION: 0.05},
            0.85,
        ),
        "due:stated": NoulAnswer(0.95),
    })
    assert extract(ok, "due") == "2026-09-24"
    not_stated = _d({"due": ok["due"], "due:stated": NoulAnswer(0.2)})
    assert extract(not_stated, "due") is None
    none = _d({
        "due": ChoiceAnswer(
            NONE_OPTION,
            {"2026-09-24": 0.1, NONE_OPTION: 0.9},
            0.8,
        ),
        "due:stated": NoulAnswer(0.9),
    })
    assert extract(none, "due") is None
    weak = _d({
        "due": ChoiceAnswer(
            "2026-09-24",
            {"2026-09-24": 0.5, "2026-10-01": 0.5},
            0.0,
        ),
        "due:stated": NoulAnswer(0.9),
    })
    assert extract(weak, "due") is None and extract(
        weak, "due", min_confidence=0.0
    ) == "2026-09-24"
