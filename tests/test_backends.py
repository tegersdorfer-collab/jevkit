import asyncio
import json

import httpx
import pytest

from jevkit.backends import (
    BackendError,
    OpenRouterBackend,
    StaticBackend,
    TypeSafeBackend,
)


def _ok(request):
    body = json.loads(request.content)
    answers = {qid: {"type": "noul", "noul": 0.9} for qid in body["questions"]}
    return httpx.Response(200, json={"model": "typesafe/jev-1.13", "answers": answers,
                                     "usage": {"input_tokens": 5, "output_tokens": 0}})


def test_typesafe_backend_sendet_richtigen_body_und_header():
    seen = {}
    def h(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _ok(request)
    b = TypeSafeBackend("sk-ts", transport=httpx.MockTransport(h))
    r = asyncio.run(b.ask({"t": "x"}, {"q": {"type": "noul", "instructions": "?"}}, model=None, timeout_s=5))
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer sk-ts"
    assert seen["body"] == {"model": "jev-latest", "state": {"t": "x"},
                            "questions": {"q": {"type": "noul", "instructions": "?"}}}
    assert r.model == "typesafe/jev-1.13" and r.answers["q"]["noul"] == 0.9 and r.usage["input_tokens"] == 5


def test_openrouter_backend_url_und_modell():
    seen = {}
    def h(request):
        seen["url"] = str(request.url)
        seen["model"] = json.loads(request.content)["model"]
        return _ok(request)
    b = OpenRouterBackend("sk-or", transport=httpx.MockTransport(h))
    asyncio.run(b.ask("s", {"q": {"type": "noul", "instructions": "?"}}, model=None, timeout_s=5))
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["model"] == "~typesafe/jev-latest"


def test_api_key_aus_umgebung(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-env")
    b = TypeSafeBackend(transport=httpx.MockTransport(_ok))
    assert b.api_key == "sk-env"
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(BackendError):
        TypeSafeBackend(transport=httpx.MockTransport(_ok))


def test_retry_bei_520_dann_erfolg():
    calls = {"n": 0}
    def h(request):
        calls["n"] += 1
        return httpx.Response(520, text="cloudflare") if calls["n"] == 1 else _ok(request)
    b = TypeSafeBackend("k", transport=httpx.MockTransport(h), retries=2, backoff_s=0.0)
    r = asyncio.run(b.ask("s", {"q": {"type": "noul", "instructions": "?"}}, model=None, timeout_s=5))
    assert calls["n"] == 2 and r.answers["q"]["noul"] == 0.9


def test_kein_retry_bei_422_und_retries_0():
    calls = {"n": 0}
    def h(request):
        calls["n"] += 1
        return httpx.Response(422, json={"error": "bad"})
    b = TypeSafeBackend("k", transport=httpx.MockTransport(h), retries=2, backoff_s=0.0)
    with pytest.raises(BackendError) as ei:
        asyncio.run(b.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert calls["n"] == 1 and ei.value.retryable is False and "422" in str(ei.value)

    calls["n"] = 0
    def h503(request):
        calls["n"] += 1
        return httpx.Response(503)
    b0 = TypeSafeBackend("k", transport=httpx.MockTransport(h503), retries=0)
    with pytest.raises(BackendError) as ei:
        asyncio.run(b0.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert calls["n"] == 1 and ei.value.retryable is True


def test_netzfehler_wird_backend_error():
    def h(request):
        raise httpx.ReadTimeout("langsam")
    b = TypeSafeBackend("k", transport=httpx.MockTransport(h), retries=0)
    with pytest.raises(BackendError) as ei:
        asyncio.run(b.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert ei.value.retryable is True


def test_static_backend_dict_und_handler():
    s = StaticBackend({"q": {"type": "noul", "noul": 0.2}})
    r = asyncio.run(s.ask("st", {"q": {}}, model=None, timeout_s=1))
    assert r.answers["q"]["noul"] == 0.2 and r.model == "static" and s.calls == [("st", {"q": {}})]

    def handler(state, questions):
        return {qid: {"type": "noul", "noul": 1.0 if state == "yes" else 0.0} for qid in questions}
    d = StaticBackend(handler, model="fake-1")
    assert asyncio.run(d.ask("yes", {"a": {}}, model=None, timeout_s=1)).answers["a"]["noul"] == 1.0
    assert asyncio.run(d.ask("no", {"a": {}}, model=None, timeout_s=1)).model == "fake-1"


def test_kaputter_200_body_wird_backend_error():
    # Test case (a): non-JSON response body
    calls_a = {"n": 0}
    def h_html(request):
        calls_a["n"] += 1
        return httpx.Response(200, text="<html>")
    b_a = TypeSafeBackend("k", transport=httpx.MockTransport(h_html), retries=2)
    with pytest.raises(BackendError) as ei:
        asyncio.run(b_a.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert calls_a["n"] == 1 and ei.value.retryable is False

    # Test case (b): JSON response missing "answers" key
    calls_b = {"n": 0}
    def h_no_answers(request):
        calls_b["n"] += 1
        return httpx.Response(200, json={"model": "m"})
    b_b = TypeSafeBackend("k", transport=httpx.MockTransport(h_no_answers), retries=2)
    with pytest.raises(BackendError) as ei:
        asyncio.run(b_b.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert calls_b["n"] == 1 and ei.value.retryable is False
