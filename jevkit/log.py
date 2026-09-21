"""
JSONL log of every question: answer, band, model, latency, state hash. Outcomes
(correct / incorrect, e.g. because the user corrected it) are appended as separate
lines and merged on read by (qid, state_hash) - the last entry wins. `calibrate` builds
Brier/ECE and band suggestions from this. The state itself is NOT logged (privacy),
only its hash.

Note: for Choice/Score, `answer.to_dict()` also logs the full option names (`choice`,
`legend`) - for `compose.extract` these are literal content candidates from the state,
not anonymous labels. If you don't want that logged, log Nouls only or redact the
answer before `write`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevkit.answers import Answer, parse_answer
from jevkit.client import Decision
from jevkit.gate import Band

log = logging.getLogger(__name__)


def state_hash(state: Any) -> str:
    blob = json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class Record:
    ts: float
    qid: str
    state_hash: str
    answer: Answer
    band: Band
    model: str
    latency_s: float
    correct: bool | None


class DecisionLog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write(self, decision: Decision, bands: Mapping[str, Band], state: Any) -> str:
        h = state_hash(state)
        now = time.time()
        for qid, ans in decision.answers.items():
            self._append({
                "kind": "decision", "ts": now, "qid": qid, "state_hash": h,
                "answer": ans.to_dict(), "band": bands[qid].value,
                "model": decision.model, "latency_s": decision.latency_s
            })
        return h

    def outcome(self, qid: str, state_hash_: str, correct: bool) -> None:
        self._append({
            "kind": "outcome", "ts": time.time(), "qid": qid,
            "state_hash": state_hash_, "correct": correct
        })

    def records(self) -> list[Record]:
        if not self.path.exists():
            return []
        decisions: list[tuple[int, dict]] = []
        outcomes: dict[tuple[str, str], bool] = {}
        for lineno, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Log line %d skipped: %r", lineno, line[:80])
                continue
            if row.get("kind") == "outcome":
                outcomes[(row["qid"], row["state_hash"])] = bool(row["correct"])
            else:
                decisions.append((lineno, row))
        out: list[Record] = []
        for lineno, r in decisions:
            try:
                out.append(Record(
                    r["ts"], r["qid"], r["state_hash"], parse_answer(r["answer"]),
                    Band(r["band"]), r["model"], r["latency_s"],
                    outcomes.get((r["qid"], r["state_hash"]))
                ))
            except (ValueError, KeyError, TypeError):
                log.warning("Log line %d skipped (corrupt decision): %r",
                           lineno, str(r)[:80])
        return out
