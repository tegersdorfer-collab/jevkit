import asyncio

import pytest

from jevkit.backends import BackendError
from jevkit.prompt_backend import PromptBackend, build_prompt, parse_reply

NOUL = {
    "type": "noul",
    "instructions": "Is this a refund request?",
    "criteria": {"true": "asks for money", "false": "else"}
}
CHOICE = {
    "type": "choice",
    "instructions": "Category?",
    "criteria": {"bug": "defect", "feature": None}
}
SCORE = {
    "type": "score",
    "instructions": "Severity?",
    "criteria": ["none", "low", "high"]
}


def test_build_prompt_enthaelt_state_frage_und_format():
    p = build_prompt({"msg": "Geld zurück!"}, "q", NOUL)
    assert "Geld zurück!" in p and "Is this a refund request?" in p and "asks for money" in p
    assert "0" in p and "100" in p
    pc = build_prompt("x", "q", CHOICE)
    assert "bug" in pc and "feature" in pc and "defect" in pc
    ps = build_prompt("x", "q", SCORE)
    assert "0: none" in ps and "2: high" in ps


def test_parse_reply_noul():
    assert parse_reply(NOUL, "85") == {"type": "noul", "noul": 0.85}
    assert parse_reply(NOUL, "Probability: 7%") == {"type": "noul", "noul": 0.07}
    assert parse_reply(NOUL, "150") == {"type": "noul", "noul": 1.0}
    with pytest.raises(BackendError):
        parse_reply(NOUL, "keine Ahnung")


def test_parse_reply_choice_und_score():
    result = parse_reply(CHOICE, "  Bug  ")
    assert result == {
        "type": "choice", "choice": "bug",
        "probabilities": {"bug": 1.0, "feature": 0.0},
        "confidence": 1.0
    }
    assert parse_reply(CHOICE, "I think feature.")["choice"] == "feature"
    with pytest.raises(BackendError):
        parse_reply(CHOICE, "docs")
    s = parse_reply(SCORE, "2")
    assert s == {
        "type": "score", "score": 2.0,
        "legend": {"0": "none", "1": "low", "2": "high"},
        "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
        "confidence": 1.0
    }
    with pytest.raises(BackendError):
        parse_reply(SCORE, "7")


def test_backend_fragt_pro_frage_und_liefert_raw_response():
    prompts = []
    async def ask(prompt):
        prompts.append(prompt)
        return "90" if "refund" in prompt else "bug"
    b = PromptBackend(ask, name="qwen")
    r = asyncio.run(b.ask(
        {"msg": "x"}, {"n": NOUL, "c": CHOICE},
        model=None, timeout_s=1
    ))
    assert r.model == "prompt:qwen" and b.model == "prompt:qwen"
    assert r.answers == {
        "n": {"type": "noul", "noul": 0.9},
        "c": {
            "type": "choice", "choice": "bug",
            "probabilities": {"bug": 1.0, "feature": 0.0},
            "confidence": 1.0
        }
    }
    assert len(prompts) == 2
