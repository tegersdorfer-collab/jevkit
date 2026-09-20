import pytest

from jevkit.questions import Choice, Noul, Score


def test_noul_payload_ohne_und_mit_criteria():
    assert Noul("Is it a refund request?").payload() == {
        "type": "noul", "instructions": "Is it a refund request?"
    }
    q = Noul({"question": "Refund?"}, criteria={"true": "asks for money back", "false": "anything else"})
    assert q.payload() == {"type": "noul", "instructions": {"question": "Refund?"},
                           "criteria": {"true": "asks for money back", "false": "anything else"}}
    assert q.kind == "noul"


def test_choice_payload_und_grenzen():
    q = Choice("Which category?", {"bug": "software defect", "feature": None})
    assert q.payload() == {"type": "choice", "instructions": "Which category?",
                           "criteria": {"bug": "software defect", "feature": None}}
    assert q.kind == "choice"
    with pytest.raises(ValueError):
        Choice("?", {})
    with pytest.raises(ValueError):
        Choice("?", {f"o{i}": None for i in range(256)})
    Choice("?", {f"o{i}": None for i in range(255)})  # Obergrenze inklusive


def test_score_payload_und_grenzen():
    q = Score("Severity?", ["none", "low", "high"])
    assert q.payload() == {"type": "score", "instructions": "Severity?", "criteria": ["none", "low", "high"]}
    assert q.kind == "score"
    assert q.criteria == ("none", "low", "high")
    with pytest.raises(ValueError):
        Score("?", ["only one"])
    with pytest.raises(ValueError):
        Score("?", [str(i) for i in range(11)])


def test_fragen_sind_immutable():
    q = Noul("x")
    with pytest.raises(AttributeError):
        q.instructions = "y"  # type: ignore[misc]


def test_kind_nicht_ueberschreibbar():
    """kind ist init=False und kann nicht übergeben werden."""
    with pytest.raises(TypeError):
        Noul("x", kind="override")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Choice("x", {}, kind="override")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Score("x", [], kind="override")  # type: ignore[call-arg]

    # Korrekte kind-Werte
    assert Noul("x").kind == "noul"
    assert Choice("x", {"a": None}).kind == "choice"
    assert Score("x", ["a", "b"]).kind == "score"
