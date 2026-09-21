# jevkit - Design (decision kernel for TypeSafe Jev)

As of: 2026-09-21. Sources: docs.typesafe.ai (concepts, API, patterns, cookbooks, jaggedness jev-1.13) plus the Mantis benchmark (bench/jev, 86/92, Brier 0.046).

## What Jev is

- Not a generator. An RLCD-trained model that returns a probability distribution over options *supplied by the caller* for each question. Never a value outside that set.
- One request = one `state` (str | JSON object | array) + `questions` (map id -> question). All questions see the same state and are evaluated in parallel and in isolation. The state is billed once -> fan-out is nearly free (13 questions in one call: 12.2x cheaper, 10x faster than 13 calls).
- Primitives: **Noul** (P(yes)), **Choice** (<=255 options; `choice`, `probabilities`, `confidence`), **Score** (2-10 ordered levels; `score` as expected value, `legend`, `probabilities`, `confidence`).
- Confidence = peakedness: `(k*p_max - 1)/(k - 1)`. Noul has none.
- Limits: 64k context/request, 32k state + longest question. $0.042/1M input, output free. Rate limits are dynamic. Errors: 401/422/429/529 (+ Cloudflare 520 observed).
- Instructions and criteria may be JSON (objects with `what`/`not_for`/`examples`, levels with `summary`/`signals`, dot references into the state).
- Nearly deterministic (std dev 0.01 across repeats), but Nouls near 0.5 flip.
- Endpoints with an identical body: `POST https://api.typesafe.ai/v1/systemone` (model `jev-latest`), `POST https://openrouter.ai/api/alpha/decisions` (model `~typesafe/jev-latest`). Response: `{model, answers, usage}`.

## Weaknesses (official + external) -> countermeasure

| # | Weakness | Countermeasure in the kernel |
|---|---|---|
| 1 | Reads literally | Exact condition in `instructions`, edge cases in `criteria` (structured) |
| 2 | Can't do arithmetic/counting | `compose.count`: one Noul per item, summed in code; numbers pre-bucketed |
| 3 | Date = text | `state.relative_days` returns a label, comparison happens in code |
| 4 | Negation/indirection | Positive questions only; a negated counterpart as a consistency check (`gate.consistent`) |
| 5 | Irrelevant state | `state.project` (field projection), prefer a lean bin-packing over a fat one |
| 6 | No injection defense | `state.untrusted` + a guard Noul in the same fan-out; guard positive -> everything to ESCALATE |
| 7 | P(A)+P(not A) != 1 | Consistency check demotes to CONFIRM |
| 8 | No generation | `compose.extract`: candidates from code, Jev picks (Choice) + a "stated" Noul |
| 9 | Weaker in German | Content stays in German, instructions/criteria are English |
| 10 | No rationale | Not in the kernel; the caller asks an LLM if needed |
| 11 | Accuracy below frontier | Three bands ACT/CONFIRM/ESCALATE instead of a threshold; bands per severity |
| 12 | 520/529/dynamic limits | Retry with backoff within the timeout budget, circuit breaker, `JevUnavailable` |
| 13 | Version changes shift bands | `Decision.model` is logged; registry warns when the responding version differs from the calibrated one |

## Layers

0. **Registry** (`registry.py`): `DecisionSpec(id, question, severity, privacy, bands, calibrated_model)`; `Router` picks the backend by `privacy` (CLOUD -> Jev, LOCAL -> local backend).
1. **State builder** (`state.py`): `project`, `bucket`, `count_bucket`, `relative_days`, `untrusted`, the `GUARD` question, `with_guard`.
2. **Compose** (`compose.py`, `tools.py`, `taxonomy.py`): count, composite, extract, function_call, beam_search.
3. **Client** (`client.py`, `cache.py`, `backends.py`): one call per state, cache keyed on hash, retry/breaker/timeout, backends TypeSafe / OpenRouter / Prompt (local LLM, uncalibrated) / Static (tests).
4. **Gate** (`gate.py`): `Band`, `Bands`, `Severity`, `band()`, `consistent()`.
5. **Escalation**: the outcome of the bands; the cascade itself is the caller's responsibility (the kernel returns band + answer).
6. **Log & calibration** (`log.py`, `calibrate.py`): a JSONL log per question, outcomes appended later, Brier/ECE/accuracy per question, `suggest_bands` from labeled outcomes.

## Not in v1

An auto-research loop (LLM proposes questions, a classifier scores them), a rationale LLM, an Ollama adapter with calibrated probabilities.

## Mantis integration

`core/decide.py` keeps its interface (`Noul`, `Choice`, `Answer`, `decide()`, `JevUnavailable`, `enabled()`, `reset_for_tests()`, `_transport`) and delegates to `jevkit.Client` with `OpenRouterBackend`. `core/decisions.py` gets bands (instead of `SURE`), a guard on the voice transcript, and a `DecisionLog` under `data/decisions.jsonl`.
