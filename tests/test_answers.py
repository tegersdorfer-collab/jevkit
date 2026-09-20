import pytest

from jevkit.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer, parse_answer


def test_noul_answer_ableitungen():
    a = NoulAnswer(0.93)
    assert a.kind == "noul" and a.value is True and a.p == 0.93
    assert abs(a.confidence - 0.86) < 1e-9
    assert NoulAnswer(0.5).value is True and NoulAnswer(0.5).confidence == 0.0
    assert NoulAnswer(0.1).value is False and abs(NoulAnswer(0.1).confidence - 0.8) < 1e-9


def test_choice_answer_ableitungen():
    a = ChoiceAnswer("flipper", {"flipper": 0.9, "robot": 0.1}, 0.88)
    assert a.kind == "choice" and a.value == "flipper" and a.p == 0.9 and a.confidence == 0.88


def test_score_answer_normalisierung():
    a = ScoreAnswer(2.4, {"0": "none", "1": "low", "2": "mid", "3": "high"},
                    {"0": 0.0, "1": 0.1, "2": 0.4, "3": 0.5}, 0.35)
    assert a.kind == "score" and a.value == 2.4 and a.level == 2
    assert abs(a.normalized - 0.8) < 1e-9          # (2.4 - 0) / (3 - 0)
    assert abs(a.p - 0.4) < 1e-9                   # P(level 2)
    one_based = ScoreAnswer(3.0, {"1": "a", "2": "b", "3": "c"}, {"1": 0.0, "2": 0.0, "3": 1.0}, 1.0)
    assert one_based.normalized == 1.0


def test_parse_answer_alle_typen():
    assert parse_answer({"type": "noul", "noul": 0.6}) == NoulAnswer(0.6)
    c = parse_answer({
        "type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.4
    })
    assert c == ChoiceAnswer("a", {"a": 0.7, "b": 0.3}, 0.4)
    s = parse_answer({"type": "score", "score": 1.5, "legend": {"0": "x", "1": "y", "2": "z"},
                      "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}, "confidence": 0.25})
    assert isinstance(s, ScoreAnswer) and s.score == 1.5


@pytest.mark.parametrize("raw", [
    "kaputt",
    {"type": "unknown"},
    {"type": "noul"},
    {"type": "choice", "choice": "a", "probabilities": [0.9, 0.1], "confidence": 0.9},
    {"type": "choice", "choice": "robot", "probabilities": {"flipper": 0.9}, "confidence": 0.9},
    {"type": "score", "score": 0.5},
])
def test_parse_answer_lehnt_kaputtes_ab(raw):
    with pytest.raises(ValueError):
        parse_answer(raw)


def test_to_dict_roundtrip():
    a = ChoiceAnswer("a", {"a": 1.0}, 1.0)
    assert parse_answer(a.to_dict()) == a
    assert parse_answer(NoulAnswer(0.2).to_dict()) == NoulAnswer(0.2)
