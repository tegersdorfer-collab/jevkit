"""
JSONL-Log jeder Frage: Antwort, Band, Modell, Latenz, State-Hash. Outcomes (richtig /
falsch, z.B. weil der Nutzer korrigiert hat) werden als eigene Zeilen nachgetragen und
beim Lesen per (qid, state_hash) gemergt — der letzte Eintrag gewinnt. Daraus baut
`calibrate` Brier/ECE und Band-Vorschläge. Der State selbst wird NICHT geloggt (Privacy),
nur sein Hash.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jevkit.answers import Answer, parse_answer
from jevkit.client import Decision
from jevkit.gate import Band


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
        decisions: list[dict] = []
        outcomes: dict[tuple[str, str], bool] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "outcome":
                outcomes[(row["qid"], row["state_hash"])] = bool(row["correct"])
            else:
                decisions.append(row)
        return [
            Record(
                r["ts"], r["qid"], r["state_hash"], parse_answer(r["answer"]),
                Band(r["band"]), r["model"], r["latency_s"],
                outcomes.get((r["qid"], r["state_hash"]))
            )
            for r in decisions
        ]
