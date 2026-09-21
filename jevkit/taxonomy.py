"""
Hierarchical classification via beam search (docs.typesafe.ai/cookbooks/hierarchical_classification):
one Choice per level over the direct children, K paths stay in the running, and all
frontier paths are asked in parallel. Path score = geometric mean of the edge
probabilities, so shallow and deep leaves compare fairly.
Greedy can't fix an ambiguous early decision, beam can (cookbook: 2/4 vs 4/4).

Tree: {node: children-dict | description}. A non-dict value is a leaf.
"""
from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jevkit.answers import ChoiceAnswer
from jevkit.client import Client
from jevkit.questions import Choice, Json

Tree = Mapping[str, Any]


@dataclass(frozen=True)
class Path:
    nodes: tuple[str, ...]
    probs: tuple[float, ...]

    @property
    def score(self) -> float:
        if not self.probs:
            return 0.0
        return math.exp(sum(math.log(max(p, 1e-12)) for p in self.probs) / len(self.probs))

    @property
    def leaf(self) -> str | None:
        return self.nodes[-1] if self.nodes else None


def _subtree(tree: Tree, nodes: tuple[str, ...]) -> Any:
    cur: Any = tree
    for n in nodes:
        cur = cur[n]
    return cur


def _criteria(children: Tree, max_listed: int) -> dict[str, Json]:
    out: dict[str, Json] = {}
    for name, sub in children.items():
        out[name] = {"contains": list(sub)[:max_listed]} if isinstance(sub, Mapping) else sub
    return out


async def beam_search(client: Client, state: Any, tree: Tree, *, k: int = 3,
                      instructions: Json = "Which direct child category best matches the content?",
                      max_listed: int = 20) -> list[Path]:
    if not tree:
        raise ValueError("beam_search: empty tree")
    frontier = [Path((), ())]
    finished: list[Path] = []
    qid = "child"
    while frontier:
        expandable = [p for p in frontier if isinstance(_subtree(tree, p.nodes), Mapping)]
        finished.extend(p for p in frontier if p not in expandable)
        if not expandable:
            break

        async def ask(p: Path) -> list[Path]:
            children = _subtree(tree, p.nodes)
            q = Choice({"question": instructions, "parent_path": list(p.nodes)},
                      _criteria(children, max_listed))
            d = await client.decide(state, {qid: q})
            ans = d[qid]
            assert isinstance(ans, ChoiceAnswer)
            return [Path((*p.nodes, name), (*p.probs, prob)) for name, prob in ans.probabilities.items()]

        expanded = await asyncio.gather(*(ask(p) for p in expandable))
        candidates = [c for group in expanded for c in group]
        frontier = sorted(candidates, key=lambda p: p.score, reverse=True)[:k]
    return sorted(finished, key=lambda p: p.score, reverse=True)[:k]
