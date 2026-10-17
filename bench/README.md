# Jev upgrade benchmark

`bench/` is a repository tool, not part of the installed `jevkit` package. It compares pinned Jev model versions
on labeled decisions and evaluates current guard wording against an optional historical wording. Runs make live
provider requests; analysis is fully offline.

## Run a comparison

Set one provider key in the environment, or give `--env-file` a file containing `KEY=VALUE` lines. The key is
read locally and is never printed or added to result rows.

```sh
export TYPESAFE_API_KEY='…'  # or OPENROUTER_API_KEY for OpenRouter
python3 -m bench.ab_decisions bench/local/decisions.json \
  --models JEV_1_13,JEV_1_14 --provider typesafe --reps 3 --concurrency 8 \
  --out bench/results/decisions-example.jsonl
python3 -m bench.ab_guard --models JEV_1_13,JEV_1_14 --provider typesafe \
  --split test --reps 3 --baseline-ref v0.2.0 --out bench/results/guard-example.jsonl
python3 -m bench.analyze decisions bench/results/decisions-example.jsonl --report decisions.md
python3 -m bench.analyze guard bench/results/guard-example.jsonl --report guard.md
```

Both runners accept `--out PATH`; by default they write JSONL under `bench/results/` with a timestamped name.
Decision cases are supplied as a JSON array. Guard cases default to the tracked `bench/data/guard_eval.json`;
choose `dev`, `test`, or `all` with `--split`. `--baseline-ref` loads `GUARD` from that git ref's
`jevkit/state.py` into a temporary isolated module. The current wording is always `jevkit.state.GUARD`, and
`SELF_CLAIM` is scored alongside it.

Guard cases may include `"source": "user_utterance"` or `"source": "third_party_content"`; rows record the
effective source. Cases without a source keep the legacy behavior by default. `--default-source` can apply one
of those labels to source-less cases, and `--categories a,b,...` limits a run to selected categories.

The default models are the previous known version and the current `PINNED` version. `--models` accepts comma-
separated Jev versions such as `1.13,1.14`, constant labels such as `JEV_1_13,JEV_1_14`, or provider ids.
TypeSafe and OpenRouter ids are selected from each `JevModel`; the benchmark does not use a moving `latest`
alias. Transient HTTP errors (408, 425, 429, 500, 502–504, 520, 522, 524, 529) are retried by the backend; result
rows store only the error type and HTTP status, never response bodies.

## Decision case format

Keep private decision datasets in the ignored `bench/local/` directory. Do not copy their text into tracked
files. Build your own JSON file using this generic shape:

```json
[
  {
    "id": "case-001",
    "kind": "permission",
    "state": {"request": "Turn on the desk lamp."},
    "question": {
      "type": "noul",
      "instructions": "Does the request ask to turn on a light?",
      "criteria": {"true": "It asks for a light", "false": "It does not"}
    },
    "expected": true
  }
]
```

`question` uses Jev's normal `noul` or `choice` wire shape (Score questions are rejected: their continuous
expectation has no consistent exact-match accuracy yet). `expected` is the labeled boolean for
Noul or the selected option for Choice. Keep IDs stable across model runs so
paired analysis can compare the same examples.

## Reading the report

Decision analysis reports accuracy across every repetition and by majority vote, Noul Brier score, log loss,
10-bin ECE, confidence on right and wrong answers, high-confidence mistakes, repeat instability, and results by
`kind`. For the first two models in the selected order it also reports exact McNemar's p-value, a seeded paired
bootstrap 95% confidence interval for the log-loss difference, and every case whose majority prediction flipped.
Negative log-loss difference means the second model has lower loss. Brier is omitted for non-Noul questions.

Guard analysis averages repetitions per text and reports recall at 0.5, false positives at 0.5, benign texts at
or above 0.2, AUC, per-category scores, `SELF_CLAIM` recall/false positives, and each threshold error with its
text. AUC is `n/a` if the selected split has no positive or no negative examples. A lower FP count means fewer
ordinary requests would be escalated.

## Upgrade decision

Treat the benchmark as evidence for review, not an automatic release gate. On the held-out `test` split, require
no significant accuracy drop by exact McNemar, guard recall no lower and false positives no higher than the
pinned version, then inspect every flipped decision and every guard error manually. Check the dev split while
refining wording, and make the release decision from held-out results.
