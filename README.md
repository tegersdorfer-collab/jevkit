# jevkit

[![tests](https://github.com/tegersdorfer-collab/jevkit/actions/workflows/tests.yml/badge.svg)](https://github.com/tegersdorfer-collab/jevkit/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**A decision kernel for [TypeSafe Jev](https://docs.typesafe.ai).** Jev answers typed
questions with calibrated probabilities instead of generating text. jevkit is the layer
around it that keeps *your code* in control: it prepares the state, composes the questions,
turns every answer into an **act / confirm / escalate** band, and re-calibrates those bands
from real outcomes.

Every documented weakness of `jev-1.13` (the official
[jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13)) has a countermeasure in
this library. Nothing here is a prompt trick — it's code that does what the model can't.

- One runtime dependency (`httpx`). Python 3.11+.
- Works with the TypeSafe API and the OpenRouter `/api/alpha/decisions` endpoint (same body).
- 83 tests, no network in tests. Response parsing verified against a recorded live reply (`jev-1.13.0`, 0-indexed score levels).

## Install

```bash
pip install "jevkit @ git+https://github.com/tegersdorfer-collab/jevkit.git@v0.1.3"
export OPENROUTER_API_KEY=...   # or TYPESAFE_API_KEY for direct access
```

## 60-second tour

```python
import asyncio
from jevkit import Client, OpenRouterBackend, Noul, Choice, band, DEFAULT_BANDS, Severity

client = Client(OpenRouterBackend(), timeout_s=3.0)   # total budget, circuit breaker included

async def main():
    d = await client.decide(
        {"message": "Turn on the desk lamp"},
        {
            "action": Noul("Does the message ask the assistant to perform an action?"),
            "device": Choice("Which device is meant?",
                             {"lamp": "desk lamp", "fan": "fan", "__none__": "none of these"}),
        },
    )
    print(d["action"].p)                 # 0.97  — P(yes)
    print(d["device"].choice)            # "lamp"
    print(band(d["device"], DEFAULT_BANDS[Severity.REVERSIBLE]))   # Band.ACT

asyncio.run(main())
```

All questions in one request see the same state and are answered in parallel — the state is
billed once, so asking twelve questions costs about the same as asking one. jevkit is built
around that: fan out, then decide in code.

## Why bands instead of a threshold

Jev is calibrated, not infallible. A single `p >= 0.5` cut-off throws away the one thing the
model gives you for free — how sure it is. jevkit maps every answer to three bands on its
confidence (`|p − 0.5| · 2` for yes/no, the API's peakedness value for choices/scores):

| Band | Meaning | Default (READ / REVERSIBLE / DESTRUCTIVE) |
|---|---|---|
| `ACT` | act automatically | conf ≥ 0.4 / 0.6 / 0.8 → p ≥ 0.70 / 0.80 / 0.90 |
| `CONFIRM` | act only with a second source (local model, human) | in between |
| `ESCALATE` | don't trust the answer | conf < 0.2 / 0.3 / 0.5 |

Destructive actions need higher confidence than read-only ones. The defaults are a starting
point; `suggest_bands` replaces them with measured ones (below).

## What Jev can't do — and what jevkit does instead

| Jev weakness (from the docs) | jevkit |
|---|---|
| Reads instructions literally | structured `criteria` with `what` / `not_for` / `examples` |
| Can't count or do arithmetic | `count_questions` + `count`: one Noul per item, sum in code; `bucket` / `count_bucket` turn numbers into labels before Jev sees them |
| Reads dates as text | `relative_days(date, now)` → `"in 3 days"`, `"2 weeks ago"`; compare in code |
| Large irrelevant state is a distractor | `project(obj, ["ticket.sender.email", ...])` sends only what the question needs |
| Doesn't treat state as hostile | `untrusted(text)` + `with_guard(questions)` + `injected(decision)` — a guard question in the same fan-out, free |
| No structural invariants (P(A) + P(¬A) ≠ 1) | `check_pairs` flags contradicting question pairs; `demote` the band |
| Can't generate | `extract_questions`: your code proposes candidates (regex, parser), Jev picks one — plus a "did the text mention this at all?" Noul |
| Accuracy below frontier LLMs | bands, not thresholds; `Router` demotes anything answered by an uncalibrated fallback |
| Version drift moves the calibration | `Client(expected_model=...)` warns; `DecisionSpec.calibrated_model` demotes; `Decision.model` is logged |
| 520 / 529 / dynamic rate limits | retry with backoff inside the time budget, circuit breaker, `JevUnavailable` — never a partial result |

## Recipes

**Extraction without generation**

```python
from jevkit import extract_questions, extract
import re

text = "Deadline is next Thursday, so 2026-09-24 or maybe the 1st of October."
candidates = re.findall(r"\d{4}-\d{2}-\d{2}", text)          # your parser, not the model
qs = extract_questions("deadline", "Which date is the deadline the author commits to?", candidates)
d = await client.decide({"text": text}, qs)
extract(d, "deadline")   # "2026-09-24", or None if not stated / none fits / too uncertain
```

**Function calling with a confidence you can trust**

```python
from jevkit import ToolSpec, Arg, call_questions, resolve_call

TOOLS = [
    ToolSpec("lamp", "Switch the desk lamp on or off",
             args=(Arg("power", "Should the lamp be on or off?", {"on": None, "off": None}),)),
    ToolSpec("timer", "Set a countdown timer",
             args=(Arg("minutes", "How many minutes?", {"5": None, "10": None, "30": None}),
                   Arg("label", "What is the timer for?", {"tea": None, "laundry": None}, required=False))),
]
d = await client.decide({"utterance": "set a ten minute timer"}, call_questions(TOOLS))
call = resolve_call(d, TOOLS)
# Call(tool="timer", args={"minutes": "10"}, confidence=0.71, omitted=("label",))
```

The call's confidence is the **minimum** over every judgement involved — one wrong argument
spoils the call, so it's not the product.

**Hierarchical classification with beam search**

```python
from jevkit import beam_search

TREE = {"hardware": {"laptop": "portable computers", "phone": "mobile phones"},
        "software": {"os": "operating systems", "apps": {"games": "video games", "office": "productivity"}}}
paths = await beam_search(client, {"text": "I love playing Zelda"}, TREE, k=3)
paths[0].nodes    # ("software", "apps", "games") — beam recovers from an ambiguous first level
```

**Untrusted text with a guard**

```python
from jevkit import untrusted, with_guard, injected

d = await client.decide(untrusted(transcript), with_guard({"addressed": Noul("...")}))
if injected(d):
    ...   # treat everything in this fan-out as ESCALATE

# When the text is being *judged* (moderation, verification), add the second guard:
# content that argues for its own harmlessness flips Jev judge verdicts at high confidence
# (community measurement: ~28 % of harmful samples). `self_claim=True` detects that pattern.
qs = with_guard({"harmful": Noul("...")}, self_claim=True)
```

## The calibration loop

```python
from jevkit import DecisionLog, pairs_by_question, suggest_bands

log = DecisionLog("data/decisions.jsonl")
h = log.write(decision, {"addressed": Band.ACT}, state)   # logs answer, band, model — only the state's hash
log.outcome("addressed", h, correct=False)                # later, when a human corrects it

pairs = pairs_by_question(log.records())["addressed"]
suggest_bands(pairs)   # Bands(act=0.38, escalate=0.2) — smallest confidence with ≥95 % / ≥75 % precision
```

Bands are measured, not guessed. When `Decision.model` changes, re-run this.

## A real measurement

jevkit runs inside a personal voice assistant. The first version of the guard question asked
whether the text "contains instructions addressed to an AI assistant" — which every legitimate
voice command does. Unit tests with fakes were green; the production-path measurement was not:

| | first guard | after fix (0.1.1) |
|---|---|---|
| guard p on legitimate commands ("turn on the light") | 0.53 – 0.91 → fires | **0.01 – 0.04** |
| guard p on injections ("ignore all previous instructions…") | 0.98 – 0.99 | **0.98 – 0.99** |
| commands decided by Jev instead of falling back | 6 / 14 | **14 / 14**, 0 errors |

Second lever: the band for that question was a guess (`act = 0.5`). `suggest_bands` on 92
labelled cases said 0.38. With 0.4, Jev decides 14/14 instead of 11/14 — still 0 errors.
That loop — log, label, re-derive — is what this library is for.

## Modules

| Module | Contents |
|---|---|
| `questions` / `answers` | `Noul`, `Choice`, `Score`; typed answers with `.p`, `.confidence`, `.value` |
| `client` / `backends` / `cache` | `Client` (time budget, breaker, type check, LRU/TTL cache); `TypeSafeBackend`, `OpenRouterBackend`, `StaticBackend` (tests), `PromptBackend` (uncalibrated emulation over any text LLM, e.g. Ollama) |
| `gate` | `Band`, `Bands`, `Severity`, `band`, `demote`, `consistent`, `check_pairs` |
| `state` | `project`, `bucket`, `count_bucket`, `relative_days`, `untrusted`, `GUARD`, `with_guard`, `injected` |
| `compose` / `tools` / `taxonomy` | counting, composite scores, extraction, function calling, beam search |
| `registry` | `DecisionSpec` (question, severity, privacy, bands, calibrated model), `Registry`, `Router` (cloud → local fallback, never ACT on an uncalibrated answer) |
| `log` / `calibrate` | `DecisionLog` (JSONL, outcomes), `brier`, `ece`, `accuracy`, `pairs_by_question`, `suggest_bands` |

## Rules of thumb (from the Jev docs)

1. Content can be any language; write instructions and criteria in English.
2. Ask positive, literal questions. Put the edge cases in `criteria`.
3. Turn numbers into buckets and dates into relative labels before the model sees them.
4. Send only the fields the question needs.
5. Wrap foreign text with `untrusted()` and add the guard. If it fires, nothing in that fan-out is trusted.
6. Never one threshold; three bands, higher for destructive actions.
7. Log the model version. When it changes, re-measure.

## Known behaviour worth designing around

Collected from the official docs and from measurements shared in the TypeSafe community
(September 2026). None of these are jevkit bugs; they are properties of the model that the
library either compensates for or that you should keep in mind.

- **Probabilities mean "how sure the option is the right answer", not outcome odds.** Asked for
  the sum of two dice, Jev puts 1.0 on 7. Don't use Choice distributions as likelihoods; use them
  as confidence about a single correct answer. `beam_search` scores paths by that certainty.
- **Content that vouches for itself moves the verdict** ("nothing above is harmful"): ~28 % flips
  at high confidence in a judge setup. Confidence gating does not catch it → `with_guard(...,
  self_claim=True)`.
- **Absolute dates don't register, relative ones do.** "review by 21/9/2026" gives a neutral
  urgency; "by tomorrow" works. That is exactly what `relative_days` is for.
- **Run-to-run variation is small but real** — a Choice confidence of 0.68–0.77 across five
  identical runs was reported. Keep bands away from the values you observe in production, and
  expect a Noul near 0.5 to flip. `MemoryCache` makes repeated identical requests deterministic
  by construction.
- **Weak spots reported:** many-label attribute tagging (16 abstract labels, ~0.37 subset
  accuracy vs. ~0.7 for a generative LLM), idiom / word-association knowledge, romanized
  non-Latin-script languages, glossary-driven expansion of concatenated abbreviations. German
  content with English instructions measured fine (93 % on 92 labelled cases).
- **Model ids on the wire:** the TypeSafe API resolves `jev-latest` to `jev-1.13.0` and accepts the
  pinned id as a request model; OpenRouter answers with `typesafe/jev-1.13`. Pin the version in
  production and set `Client(expected_model=...)` accordingly.

## Status

v0.1 — API may still move. Known gaps: `PromptBackend` is uncalibrated
by design; no retry budget across multiple fan-outs. Design notes:
[`docs/design.md`](docs/design.md).

MIT.
