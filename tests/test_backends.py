import asyncio
import json

import httpx
import pytest

import jevkit.backends as backends
from jevkit.backends import (
    BackendError,
    OpenRouterBackend,
    StaticBackend,
    TypeSafeBackend,
)
from jevkit.models import JEV_1_13, JEV_1_14


def _ok(request):
    body = json.loads(request.content)
    answers = {qid: {"type": "noul", "noul": 0.9} for qid in body["questions"]}
    return httpx.Response(200, json={"model": "typesafe/jev-1.13", "answers": answers,
                                     "usage": {"input_tokens": 5, "output_tokens": 0}})


def test_typesafe_backend_sends_correct_body_and_header():
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
    assert seen["body"] == {"model": "jev-1.14.0", "state": {"t": "x"},
                            "questions": {"q": {"type": "noul", "instructions": "?"}}}
    assert r.model == "typesafe/jev-1.13" and r.answers["q"]["noul"] == 0.9 and r.usage["input_tokens"] == 5


def test_openrouter_backend_url_and_model():
    seen = {}
    def h(request):
        seen["url"] = str(request.url)
        seen["model"] = json.loads(request.content)["model"]
        return _ok(request)
    b = OpenRouterBackend("sk-or", transport=httpx.MockTransport(h))
    asyncio.run(b.ask("s", {"q": {"type": "noul", "instructions": "?"}}, model=None, timeout_s=5))
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["model"] == "typesafe/jev-1.14-20261014"


@pytest.mark.parametrize(("backend_type", "model_id"), [
    (TypeSafeBackend, "jev-1.13.0"),
    (OpenRouterBackend, "typesafe/jev-1.13-20260917"),
])
def test_jev_model_selects_the_provider_specific_id(backend_type, model_id):
    seen = {}

    def handler(request):
        seen["model"] = json.loads(request.content)["model"]
        return _ok(request)

    backend = backend_type("key", jev=JEV_1_13, transport=httpx.MockTransport(handler))
    asyncio.run(backend.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert seen["model"] == model_id


def test_model_override_accepts_latest_alias_and_cannot_mix_with_jev():
    seen = {}

    def handler(request):
        seen["model"] = json.loads(request.content)["model"]
        return _ok(request)

    backend = TypeSafeBackend("key", model="jev-latest", transport=httpx.MockTransport(handler))
    asyncio.run(backend.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert seen["model"] == "jev-latest"
    with pytest.raises(ValueError, match="model or jev"):
        TypeSafeBackend("key", model="jev-latest", jev=JEV_1_13)


def test_backend_defaults_are_explicitly_pinned():
    assert TypeSafeBackend("key").default_model == JEV_1_14.typesafe_id
    assert OpenRouterBackend("key").default_model == JEV_1_14.openrouter_id


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-env")
    b = TypeSafeBackend(transport=httpx.MockTransport(_ok))
    assert b.api_key == "sk-env"
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(BackendError):
        TypeSafeBackend(transport=httpx.MockTransport(_ok))


def test_retry_on_520_then_success():
    calls = {"n": 0}
    def h(request):
        calls["n"] += 1
        return httpx.Response(520, text="cloudflare") if calls["n"] == 1 else _ok(request)
    b = TypeSafeBackend("k", transport=httpx.MockTransport(h), retries=2, backoff_s=0.0)
    r = asyncio.run(b.ask("s", {"q": {"type": "noul", "instructions": "?"}}, model=None, timeout_s=5))
    assert calls["n"] == 2 and r.answers["q"]["noul"] == 0.9


def test_permanent_server_error_is_not_retried():
    calls = {"n": 0}

    def h(request):
        calls["n"] += 1
        return httpx.Response(501)

    b = TypeSafeBackend("k", transport=httpx.MockTransport(h), retries=3, backoff_s=0.0)
    with pytest.raises(BackendError) as e:
        asyncio.run(b.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert calls["n"] == 1 and e.value.retryable is False


def test_retry_reuses_one_async_client(monkeypatch):
    created = []
    client_type = backends.httpx.AsyncClient

    class TrackingClient(client_type):
        def __init__(self, *args, **kwargs):
            created.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(backends.httpx, "AsyncClient", TrackingClient)
    calls = {"n": 0}

    def h(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else _ok(request)

    b = TypeSafeBackend("k", transport=httpx.MockTransport(h), retries=1, backoff_s=0.0)
    r = asyncio.run(b.ask("s", {"q": {}}, model=None, timeout_s=5))

    assert r.answers["q"]["noul"] == 0.9
    assert calls["n"] == 2 and len(created) == 1


def test_no_retry_on_422_and_retries_0():
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


def test_network_error_becomes_backend_error():
    def h(request):
        raise httpx.ReadTimeout("langsam")
    b = TypeSafeBackend("k", transport=httpx.MockTransport(h), retries=0)
    with pytest.raises(BackendError) as ei:
        asyncio.run(b.ask("s", {"q": {}}, model=None, timeout_s=5))
    assert ei.value.retryable is True


def test_static_backend_dict_and_handler():
    s = StaticBackend({"q": {"type": "noul", "noul": 0.2}})
    r = asyncio.run(s.ask("st", {"q": {}}, model=None, timeout_s=1))
    assert r.answers["q"]["noul"] == 0.2 and r.model == "static" and s.calls == [("st", {"q": {}})]

    def handler(state, questions):
        return {qid: {"type": "noul", "noul": 1.0 if state == "yes" else 0.0} for qid in questions}
    d = StaticBackend(handler, model="fake-1")
    assert asyncio.run(d.ask("yes", {"a": {}}, model=None, timeout_s=1)).answers["a"]["noul"] == 1.0
    assert asyncio.run(d.ask("no", {"a": {}}, model=None, timeout_s=1)).model == "fake-1"


def test_broken_200_body_becomes_backend_error():
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
