import pytest

from jevkit.answers import ChoiceAnswer, NoulAnswer
from jevkit.client import Decision
from jevkit.gate import DEFAULT_BANDS, Band, Bands, Severity, band, check_pairs, consistent, demote


def test_bands_validierung():
    Bands(act=0.8, escalate=0.4)
    with pytest.raises(ValueError):
        Bands(act=0.3, escalate=0.4)
    with pytest.raises(ValueError):
        Bands(act=1.2, escalate=0.4)


def test_band_ueber_confidence_fuer_alle_typen():
    b = Bands(act=0.6, escalate=0.3)
    assert band(NoulAnswer(0.9), b) is Band.ACT          # conf 0.8
    assert band(NoulAnswer(0.1), b) is Band.ACT          # conf 0.8 (klares Nein ist auch ACT)
    assert band(NoulAnswer(0.7), b) is Band.CONFIRM      # conf 0.4
    assert band(NoulAnswer(0.55), b) is Band.ESCALATE    # conf 0.1
    assert band(ChoiceAnswer("a", {"a": 0.5, "b": 0.5}, 0.0), b) is Band.ESCALATE
    assert band(ChoiceAnswer("a", {"a": 0.9, "b": 0.1}, 0.8), b) is Band.ACT
    assert band(NoulAnswer(0.8), b) is Band.ACT          # conf 0.6 == act → inklusive


def test_default_bands_steigen_mit_severity():
    r, v, d = (DEFAULT_BANDS[s] for s in (Severity.READ, Severity.REVERSIBLE, Severity.DESTRUCTIVE))
    assert r.act < v.act < d.act and r.escalate <= v.escalate <= d.escalate
    assert band(NoulAnswer(0.75), r) is Band.ACT          # lesen darf bei p=0.75
    assert band(NoulAnswer(0.75), d) is not Band.ACT      # löschen nicht


def test_demote():
    assert demote(Band.ACT) is Band.CONFIRM
    assert demote(Band.CONFIRM) is Band.ESCALATE
    assert demote(Band.ESCALATE) is Band.ESCALATE


def test_consistent_und_check_pairs():
    assert consistent(0.9, 0.1) and consistent(0.8, 0.35)          # |0.8 − 0.65| = 0.15
    assert not consistent(0.9, 0.6)                                # |0.9 − 0.4| = 0.5
    d = Decision({"a": NoulAnswer(0.9), "not_a": NoulAnswer(0.6),
                  "b": NoulAnswer(0.2), "not_b": NoulAnswer(0.85)}, "m", {}, False, 0.0)
    assert check_pairs(d, [("a", "not_a"), ("b", "not_b")]) == {"a"}
    assert check_pairs(d, [("a", "not_a")], tol=0.6) == set()
    with pytest.raises(TypeError):
        d_choice = Decision(
            {"c": ChoiceAnswer("x", {"x": 1.0}, 1.0), "nc": NoulAnswer(0.1)},
            "m", {}, False, 0.0,
        )
        check_pairs(d_choice, [("c", "nc")])
