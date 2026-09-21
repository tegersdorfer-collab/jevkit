import asyncio
import json

import httpx
import pytest

from jevkit.backends import OpenRouterBackend, StaticBackend
from jevkit.cache import MemoryCache
from jevkit.client import Client, Decision, JevUnavailable
from jevkit.questions import Choice, Noul, Score


def _noul(p):
    return {"type": "noul", "noul": p}


def test_decide_types_all_answers():
    b = StaticBackend(
        {
            "a": _noul(0.93),
            "k": {
                "type": "choice",
                "choice": "x",
                "probabilities": {"x": 0.8, "y": 0.2},
                "confidence": 0.6,
            },
            "s": {
                "type": "score",
                "score": 1.0,
                "legend": {"0": "lo", "1": "hi"},
                "probabilities": {"0": 0.0, "1": 1.0},
                "confidence": 1.0,
            },
        },
        model="jev-1.13.0",
    )
    c = Client(b)
    d = asyncio.run(c.decide({"t": "x"}, {"a": Noul("?"), "k": Choice("?", {"x": None, "y": None}),
                                          "s": Score("?", ["lo", "hi"])}))
    assert isinstance(d, Decision) and d.model == "jev-1.13.0" and d.cached is False and d.latency_s >= 0
    assert d["a"].p == 0.93 and d["k"].choice == "x" and d["s"].normalized == 1.0
    assert "a" in d and "zz" not in d
    assert b.calls[0] == (
        {"t": "x"},
        {
            "a": {"type": "noul", "instructions": "?"},
            "k": {
                "type": "choice",
                "instructions": "?",
                "criteria": {"x": None, "y": None},
            },
            "s": {"type": "score", "instructions": "?", "criteria": ["lo", "hi"]},
        },
    )


def test_empty_questions_is_caller_bug():
    with pytest.raises(ValueError):
        asyncio.run(Client(StaticBackend({})).decide("x", {}))


def test_non_serializable_state_is_caller_bug_and_does_not_open_breaker():
    b = StaticBackend({"q": _noul(0.5)})
    c = Client(b)
    with pytest.raises(TypeError):
        asyncio.run(c.decide({"x": object()}, {"q": Noul("?")}))
    assert c.available is True and b.calls == []


def test_type_mismatch_opens_breaker():
    """Answer type doesn't match the question (e.g. schema changed) -> JevUnavailable + cooldown."""
    b = StaticBackend({"q": {"type": "score", "score": 0.5}})
    c = Client(b, cooldown_s=100)
    with pytest.raises(JevUnavailable):
        asyncio.run(c.decide("x", {"q": Noul("?")}))
    assert c.available is False


def test_missing_answer_and_broken_shape_become_unavailable():
    for answers in ({}, {"q": "kaputt"},
                    {"q": {"type": "choice", "choice": "r", "probabilities": {"f": 0.9}, "confidence": 0.9}}):
        c = Client(StaticBackend(answers))
        with pytest.raises(JevUnavailable):
            asyncio.run(c.decide("x", {"q": Choice("?", {"f": None, "r": None})}))


def test_backend_error_opens_breaker_and_closes_after_cooldown():
    t = {"now": 0.0}
    calls = {"n": 0}
    def h(request):
        calls["n"] += 1
        return httpx.Response(520, text="cf") if calls["n"] == 1 else httpx.Response(
            200, json={"model": "m", "answers": {"q": _noul(0.9)}, "usage": {}})
    b = OpenRouterBackend("k", transport=httpx.MockTransport(h), retries=0)
    c = Client(b, cooldown_s=120, clock=lambda: t["now"])
    with pytest.raises(JevUnavailable):
        asyncio.run(c.decide("x", {"q": Noul("?")}))
    assert c.available is False
    with pytest.raises(JevUnavailable):          # in cooldown: no network call
        asyncio.run(c.decide("x", {"q": Noul("?")}))
    assert calls["n"] == 1
    t["now"] = 121.0
    assert c.available is True
    assert asyncio.run(c.decide("x", {"q": Noul("?")}))["q"].p == 0.9
    c.reset()
    assert c.available is True


def test_overall_budget_timeout_becomes_unavailable():
    async def h(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={"model": "m", "answers": {"q": _noul(0.9)}, "usage": {}})
    b = OpenRouterBackend("k", transport=httpx.MockTransport(h), retries=0)
    c = Client(b, timeout_s=0.05)
    with pytest.raises(JevUnavailable):
        asyncio.run(c.decide("x", {"q": Noul("?")}))
    assert c.available is False


def test_cache_hits_on_same_input():
    b = StaticBackend({"q": _noul(0.7)})
    c = Client(b, cache=MemoryCache())
    d1 = asyncio.run(c.decide({"a": 1}, {"q": Noul("?")}))
    d2 = asyncio.run(c.decide({"a": 1}, {"q": Noul("?")}))
    d3 = asyncio.run(c.decide({"a": 2}, {"q": Noul("?")}))
    assert d1.cached is False and d2.cached is True and d3.cached is False
    assert len(b.calls) == 2 and d2["q"].p == 0.7


def test_expected_model_warns_on_different_version(caplog):
    b = StaticBackend({"q": _noul(0.7)}, model="typesafe/jev-2.0")
    c = Client(b, expected_model="typesafe/jev-1.13")
    with caplog.at_level("WARNING", logger="jevkit.client"):
        d = asyncio.run(c.decide("x", {"q": Noul("?")}))
    assert d.model == "typesafe/jev-2.0" and "jev-2.0" in caplog.text and "jev-1.13" in caplog.text


def test_model_is_passed_through_to_backend():
    seen = {}
    def h(request):
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"model": "m", "answers": {"q": _noul(0.9)}, "usage": {}})
    b = OpenRouterBackend("k", transport=httpx.MockTransport(h), retries=0)
    asyncio.run(Client(b, model="~typesafe/jev-1.13").decide("x", {"q": Noul("?")}))
    assert seen["model"] == "~typesafe/jev-1.13"
