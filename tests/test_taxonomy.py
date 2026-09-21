import asyncio
import math

import pytest

from jevkit.backends import StaticBackend
from jevkit.client import Client
from jevkit.taxonomy import Path, beam_search

TREE = {
    "hardware": {"laptop": "portable computers", "phone": "mobile phones"},
    "software": {"os": "operating systems", "apps": {"games": "video games", "office": "productivity"}},
    "other": "anything else",
}


def _choice(probs):
    best = max(probs, key=probs.get)
    k = len(probs)
    conf = (k * probs[best] - 1) / (k - 1) if k > 1 else 1.0
    return {"type": "choice", "choice": best, "probabilities": probs, "confidence": conf}


def _handler(state, questions):
    """Ambig auf Ebene 1 (software 0.45 vs hardware 0.4), eindeutig darunter."""
    (qid, q), = questions.items()
    opts = list(q["criteria"])
    if opts == ["hardware", "software", "other"]:
        return {qid: _choice({"hardware": 0.4, "software": 0.45, "other": 0.15})}
    if opts == ["laptop", "phone"]:
        return {qid: _choice({"laptop": 0.55, "phone": 0.45})}
    if opts == ["os", "apps"]:
        return {qid: _choice({"os": 0.1, "apps": 0.9})}
    if opts == ["games", "office"]:
        return {qid: _choice({"games": 0.95, "office": 0.05})}
    raise AssertionError(opts)


def test_path_score_geometrisches_mittel():
    p = Path(("a", "b"), (0.5, 0.8))
    assert abs(p.score - math.sqrt(0.4)) < 1e-9 and p.leaf == "b"
    assert Path((), ()).score == 0.0


def test_beam_findet_tiefen_pfad_den_greedy_verpasst():
    b = StaticBackend(_handler)
    paths = asyncio.run(beam_search(Client(b), "I love playing Zelda", TREE, k=3))
    assert paths[0].nodes == ("software", "apps", "games")
    assert all(paths[i].score >= paths[i + 1].score for i in range(len(paths) - 1))
    assert len(paths) <= 3
    # Greedy (k=1) landet ebenfalls bei software → apps → games, aber "hardware/laptop" ist mit
    # k=1 nie im Beam
    greedy = asyncio.run(beam_search(Client(StaticBackend(_handler)), "x", TREE, k=1))
    assert len(greedy) == 1 and greedy[0].nodes == ("software", "apps", "games")


def test_beam_fragen_zeigen_teilbaum_und_beschreibung():
    seen = []
    def h(state, questions):
        seen.append(questions)
        return _handler(state, questions)
    asyncio.run(beam_search(Client(StaticBackend(h)), "x", TREE, k=1))
    root_q = seen[0][next(iter(seen[0]))]
    assert root_q["criteria"]["other"] == "anything else"
    assert root_q["criteria"]["software"] == {"contains": ["os", "apps"]}


def test_beam_leerer_baum():
    with pytest.raises(ValueError):
        asyncio.run(beam_search(Client(StaticBackend({})), "x", {}))
