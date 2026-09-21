import pytest

from jevkit.answers import NoulAnswer
from jevkit.calibrate import accuracy, brier, ece, noul_pairs, pairs_by_question, suggest_bands
from jevkit.gate import Band, Bands
from jevkit.log import Record


def test_brier_ece_accuracy():
    perfect = [(1.0, True), (0.0, False)]
    assert brier(perfect) == 0.0 and accuracy(perfect) == 1.0 and ece(perfect) == 0.0
    coin = [(0.5, True), (0.5, False)]
    assert abs(brier(coin) - 0.25) < 1e-9 and accuracy(coin) == 0.5 and abs(ece(coin) - 0.0) < 1e-9
    assert abs(brier([(0.8, True), (0.8, False)]) - (0.04 + 0.64) / 2) < 1e-9
    assert abs(ece([(0.9, True), (0.9, True), (0.9, False), (0.9, False)]) - 0.4) < 1e-9
    for f in (brier, ece, accuracy):
        with pytest.raises(ValueError):
            f([])


def _rec(qid, p, correct, model="m"):
    return Record(0.0, qid, "h", NoulAnswer(p), Band.ACT, model, 0.0, correct)


def test_pairs_by_question_and_noul_pairs():
    recs = [
        _rec("a", 0.9, True), _rec("a", 0.6, False),
        _rec("a", 0.3, None), _rec("b", 0.1, True)
    ]
    pq = pairs_by_question(recs)
    rounded = {k: [(round(c, 6), ok) for c, ok in v] for k, v in pq.items()}
    expected = {"a": [(0.8, True), (0.2, False)], "b": [(0.8, True)]}
    assert rounded == expected  # (confidence, correct); dropped without an outcome
    assert (
        noul_pairs(recs, "a") == [(0.9, True), (0.6, False)]  # label = value XNOR correct
    )
    assert (
        noul_pairs(recs, "b") == [(0.1, False)]  # value False, correct -> label False
    )


def test_suggest_bands():
    # Confidence pairs: from conf 0.8 everything correct, 0.5-0.7 mixed, below that bad
    pairs = ([(0.9, True)] * 10 + [(0.8, True)] * 10
             + [(0.6, True)] * 6 + [(0.6, False)] * 2
             + [(0.3, False)] * 8)
    b = suggest_bands(pairs)
    assert isinstance(b, Bands)
    assert b.act == 0.8                       # smallest conf at which precision >= 0.95
    assert b.escalate == 0.6                  # smallest conf at which precision >= 0.75
    assert suggest_bands([(0.9, True)] * 3) == Bands(1.0, 1.0)      # too little data -> conservative
    assert suggest_bands([(0.9, False)] * 10) == Bands(1.0, 1.0)    # never good enough -> conservative
    with pytest.raises(ValueError):
        suggest_bands(pairs, act_precision=0.5, escalate_precision=0.9)
