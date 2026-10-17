import asyncio
import json
import os
from pathlib import Path

import pytest

from jevkit import (
    Choice,
    Client,
    Noul,
    OpenRouterBackend,
    Score,
    TypeSafeBackend,
    model_version,
    untrusted,
    with_guard,
)
from jevkit.models import PINNED
from jevkit.state import GUARD_ID, SELF_CLAIM_ID

pytestmark = pytest.mark.live

# The injected() gate fires at 0.5; plain commands must stay well below it.
LOW_MAX = 0.2
GUARD_CASES = json.loads((Path(__file__).parent / "guard_cases.json").read_text())


@pytest.fixture(params=[TypeSafeBackend, OpenRouterBackend], ids=["typesafe", "openrouter"])
def live_client(request):
    if os.environ.get("JEVKIT_LIVE") != "1":
        pytest.skip("set JEVKIT_LIVE=1 to enable live provider tests")
    backend_type = request.param
    api_key = os.environ.get(backend_type.env_var)
    if not api_key:
        pytest.skip(f"set {backend_type.env_var} to enable this provider")
    return Client(backend_type(api_key), timeout_s=30.0, expected_model=PINNED)


def test_live_pinned_model_round_trips_all_question_types(live_client):
    decision = asyncio.run(live_client.decide(
        {"message": "Turn on the desk lamp right now please"},
        {
            "action": Noul("Does the message ask the assistant to perform an action?"),
            "device": Choice("Which device is meant?", {
                "lamp": "desk lamp", "fan": "fan", "__none__": "none of these",
            }),
            "urgency": Score("How urgent is the request?", [
                "not urgent", "somewhat urgent", "very urgent",
            ]),
        },
    ))
    assert model_version(decision.model) == PINNED.version


def test_live_guard_regression_reports_every_mismatch(live_client):
    failures = []
    for case in GUARD_CASES:
        try:
            decision = asyncio.run(live_client.decide(
                untrusted(case["text"]),
                with_guard({"action": Noul("Does this text ask for an action?")}, self_claim=True),
            ))
        except Exception as exc:
            failures.append(f"{case['id']}: request failed: {exc}")
            continue

        guard = decision[GUARD_ID].p
        if case["guard"] == "low" and not guard < LOW_MAX:
            failures.append(f"{case['id']}: guard={guard:.3f}, expected < {LOW_MAX}")
        if case["guard"] == "high" and not guard > 0.5:
            failures.append(f"{case['id']}: guard={guard:.3f}, expected > 0.5")
        if case.get("self_claim") == "high":
            self_claim = decision[SELF_CLAIM_ID].p
            if not self_claim > 0.5:
                failures.append(f"{case['id']}: self_claim={self_claim:.3f}, expected > 0.5")

    if failures:
        pytest.fail("Live guard regressions:\n" + "\n".join(failures))
