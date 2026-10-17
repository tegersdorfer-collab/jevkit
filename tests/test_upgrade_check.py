import asyncio
import json

import httpx

from jevkit.backends import OpenRouterBackend, TypeSafeBackend
from jevkit.models import PINNED
from jevkit.upgrade_check import check_latest, main


def _backend(backend_type, response_model, seen):
    def handler(request):
        body = json.loads(request.content)
        seen.append((backend_type.__name__, body))
        return httpx.Response(200, json={
            "model": response_model,
            "answers": {"version_check": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    return backend_type("test-key", transport=httpx.MockTransport(handler), retries=0)


def test_upgrade_check_pins_alias_probe_and_accepts_provider_ids():
    seen = []
    lines = []
    backends = [
        _backend(TypeSafeBackend, "jev-1.14.0", seen),
        _backend(OpenRouterBackend, "typesafe/jev-1.14-20261014", seen),
    ]

    result = asyncio.run(check_latest(backends, output=lines.append))

    assert result == 0
    assert len(seen) == 2
    assert seen[0][1]["model"] == "jev-latest"
    assert seen[1][1]["model"] == "~typesafe/jev-latest"
    assert all(body["questions"]["version_check"]["type"] == "noul" for _, body in seen)
    assert len(lines) == 2 and all(PINNED.version in line for line in lines)


def test_upgrade_check_reports_newer_model_with_docs_path():
    lines = []
    backend = _backend(OpenRouterBackend, "typesafe/jev-1.15-20261201", [])
    result = asyncio.run(check_latest([backend], output=lines.append))
    assert result == 3
    assert "1.15" in lines[0] and "docs/upgrading-jev.md" in lines[0]


def test_upgrade_check_treats_unknown_model_as_drift():
    lines = []
    result = asyncio.run(check_latest([_backend(TypeSafeBackend, "jev-next", [])], output=lines.append))
    assert result == 3 and "unknown version" in lines[0]


def test_upgrade_check_ignores_alias_lagging_behind_pin():
    lines = []
    result = asyncio.run(check_latest([_backend(TypeSafeBackend, "jev-1.13.0", [])], output=lines.append))
    assert result == 0 and "nothing to upgrade" in lines[0]


def test_upgrade_check_drift_wins_over_error_on_other_provider():
    def broken(request):
        return httpx.Response(500, json={"error": "down"})
    lines = []
    backends = [TypeSafeBackend("k", transport=httpx.MockTransport(broken), retries=0),
                _backend(OpenRouterBackend, "typesafe/jev-1.15-20261201", [])]
    assert asyncio.run(check_latest(backends, output=lines.append)) == 3


def test_upgrade_check_returns_error_when_no_provider_key_is_configured(monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main() == 1
    assert "API key" in capsys.readouterr().out
