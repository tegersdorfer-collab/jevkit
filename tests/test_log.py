import json

from jevkit.answers import ChoiceAnswer, NoulAnswer
from jevkit.client import Decision
from jevkit.gate import Band
from jevkit.log import DecisionLog, state_hash


def test_write_records_outcome(tmp_path):
    lg = DecisionLog(tmp_path / "d.jsonl")
    d = Decision(
        {
            "a": NoulAnswer(0.9),
            "k": ChoiceAnswer("x", {"x": 0.7, "y": 0.3}, 0.4)
        },
        "jev-1.13.0", {}, False, 0.12
    )
    h = lg.write(d, {"a": Band.ACT, "k": Band.CONFIRM}, {"t": "hallo"})
    assert h == state_hash({"t": "hallo"}) and len(h) == 64
    recs = lg.records()
    assert [r.qid for r in recs] == ["a", "k"]
    assert (
        recs[0].answer == NoulAnswer(0.9)
        and recs[0].band is Band.ACT
        and recs[0].model == "jev-1.13.0"
    )
    assert (
        recs[1].answer == ChoiceAnswer("x", {"x": 0.7, "y": 0.3}, 0.4)
        and recs[0].correct is None
    )
    lg.outcome("a", h, correct=False)
    recs = lg.records()
    assert recs[0].correct is False and recs[1].correct is None
    lg.outcome("a", h, correct=True)                       # last entry wins
    assert lg.records()[0].correct is True
    assert DecisionLog(tmp_path / "leer.jsonl").records() == []
    assert (tmp_path / "d.jsonl").read_text().count("\n") == 4


def test_state_hash_is_stable():
    assert state_hash({"a": 1, "b": 2}) == state_hash({"b": 2, "a": 1})
    assert state_hash("x") != state_hash("y")


def test_corrupt_line_is_skipped(tmp_path, caplog):
    lg = DecisionLog(tmp_path / "d.jsonl")
    d = Decision({"a": NoulAnswer(0.9)}, "jev-1.0", {}, False, 0.1)
    lg.write(d, {"a": Band.ACT}, {"x": 1})
    # Append a truncated line (crash mid-append)
    with (tmp_path / "d.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"kind": "decision", "ts": 1')  # no newline
    # records() must skip the broken line and return the valid record
    recs = lg.records()
    assert len(recs) == 1 and recs[0].qid == "a"
    # Check caplog for warning
    assert any(
        "skipped" in record.message
        and record.name == "jevkit.log"
        and record.levelname == "WARNING"
        for record in caplog.records
    )


def test_corrupt_decision_line_is_skipped(tmp_path, caplog):
    lg = DecisionLog(tmp_path / "d.jsonl")
    d = Decision({"a": NoulAnswer(0.9), "b": NoulAnswer(0.8)}, "jev-1.0", {}, False, 0.1)
    lg.write(d, {"a": Band.ACT, "b": Band.ACT}, {"x": 1})
    # Valid JSON, but not a valid band -> records() must not crash.
    with (tmp_path / "d.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "kind": "decision", "ts": 1.0, "qid": "c", "state_hash": "h",
            "answer": {"type": "noul", "noul": 0.5}, "band": "unsinn",
            "model": "jev-1.0", "latency_s": 0.1
        }) + "\n")
    recs = lg.records()
    assert [r.qid for r in recs] == ["a", "b"]
    assert any(
        "skipped" in record.message and "corrupt decision" in record.message
        and record.name == "jevkit.log" and record.levelname == "WARNING"
        for record in caplog.records
    )
