import pytest

from jevkit.questions import Choice, Noul, Score


def test_noul_payload_with_and_without_criteria():
    assert Noul("Is it a refund request?").payload() == {
        "type": "noul", "instructions": "Is it a refund request?"
    }
    q = Noul({"question": "Refund?"}, criteria={"true": "asks for money back", "false": "anything else"})
    assert q.payload() == {"type": "noul", "instructions": {"question": "Refund?"},
                           "criteria": {"true": "asks for money back", "false": "anything else"}}
    assert q.kind == "noul"


def test_choice_payload_and_limits():
    q = Choice("Which category?", {"bug": "software defect", "feature": None})
    assert q.payload() == {"type": "choice", "instructions": "Which category?",
                           "criteria": {"bug": "software defect", "feature": None}}
    assert q.kind == "choice"
    with pytest.raises(ValueError):
        Choice("?", {})
    with pytest.raises(ValueError):
        Choice("?", {f"o{i}": None for i in range(256)})
    Choice("?", {f"o{i}": None for i in range(255)})  # upper bound inclusive


def test_score_payload_and_limits():
    q = Score("Severity?", ["none", "low", "high"])
    assert q.payload() == {"type": "score", "instructions": "Severity?", "criteria": ["none", "low", "high"]}
    assert q.kind == "score"
    assert q.criteria == ("none", "low", "high")
    with pytest.raises(ValueError):
        Score("?", ["only one"])
    with pytest.raises(ValueError):
        Score("?", [str(i) for i in range(11)])


def test_questions_are_immutable():
    q = Noul("x")
    with pytest.raises(AttributeError):
        q.instructions = "y"  # type: ignore[misc]


def test_kind_is_not_overridable():
    """kind is init=False and cannot be passed in."""
    with pytest.raises(TypeError):
        Noul("x", kind="override")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Choice("x", {}, kind="override")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Score("x", [], kind="override")  # type: ignore[call-arg]

    # correct kind values
    assert Noul("x").kind == "noul"
    assert Choice("x", {"a": None}).kind == "choice"
    assert Score("x", ["a", "b"]).kind == "score"
