import asyncio
import json
from pathlib import Path

import httpx
import pytest

from jevkit.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from jevkit.backends import OpenRouterBackend, TypeSafeBackend
from jevkit.client import Client
from jevkit.questions import Choice, Noul, Score

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.json"))


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda path: path.name)
def test_recorded_wire_reply_is_parsed_by_real_http_backend(fixture_path):
    fixture = json.loads(fixture_path.read_text())
    request_data = fixture["request"]
    response_data = fixture["response"]
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(fixture["status"], json=response_data)

    backend_type = OpenRouterBackend if fixture_path.name.startswith("openrouter_") else TypeSafeBackend
    backend = backend_type(
        "fixture-key",
        model=request_data["model"],
        transport=httpx.MockTransport(handler),
        retries=0,
    )
    payloads = request_data["questions"]
    questions = {
        "action": Noul(payloads["action"]["instructions"]),
        "device": Choice(payloads["device"]["instructions"], payloads["device"]["criteria"]),
        "urgency": Score(payloads["urgency"]["instructions"], payloads["urgency"]["criteria"]),
    }

    decision = asyncio.run(Client(backend).decide(request_data["state"], questions))

    assert seen == [request_data]
    assert decision.model == response_data["model"]
    assert isinstance(decision["action"], NoulAnswer)
    assert isinstance(decision["device"], ChoiceAnswer)
    assert isinstance(decision["urgency"], ScoreAnswer)
