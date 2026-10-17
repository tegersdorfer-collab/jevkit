# Upgrading Jev

Jevkit pins an explicit model version. A new Jev release does not change existing jevkit
behavior; adopting it requires a jevkit minor release.

## Release checklist

1. Run `jevkit-check-latest` with each provider key configured. Record the resolved id and
   confirm that the provider serves the candidate release.
2. Add a frozen `JevModel` constant in `jevkit/models.py` with the exact TypeSafe and OpenRouter
   ids, and include it in `KNOWN`.
3. Record a representative request and response from each provider in `tests/fixtures/`, using
   the `{ "request", "status", "response" }` wire shape. The offline tests must parse Noul,
   Choice, and Score through the real HTTP backends.
4. Configure the live compatibility suite to use the candidate pin, then run
   `JEVKIT_LIVE=1 pytest -m live`. Check the pinned model round trip and every guard case on each
   provider. The live suite reports all guard mismatches together.
5. Collect real labeled outcomes for the new model and use `calibrate.suggest_bands` to derive
   new bands. Review the measurements before adopting them.
6. Run the upgrade benchmark against the current pin before switching `PINNED`. Include decisions and the
   held-out guard test split; compare the candidate with the pinned Jev model and, when evaluating guard wording,
   pass a ref to the previous `jevkit/state.py` as the baseline:

   ```sh
   python3 -m bench.ab_decisions bench/local/decisions.json --models JEV_1_14,JEV_1_15 \
     --out bench/results/decisions-candidate.jsonl
   python3 -m bench.ab_guard --models JEV_1_14,JEV_1_15 --split test --baseline-ref v0.2.0 \
     --out bench/results/guard-candidate.jsonl
   python3 -m bench.analyze decisions bench/results/decisions-candidate.jsonl
   python3 -m bench.analyze guard bench/results/guard-candidate.jsonl
   ```

   Review accuracy and calibration, guard recall and false positives, and every flipped case. See
   [`bench/README.md`](../bench/README.md) for result interpretation and private-dataset handling.
7. Set `PINNED` to the candidate model after its live behavior, calibration, and benchmark results have been
   reviewed.
   Keep the previous model in `KNOWN` so callers can continue selecting it with `jev=`.
8. Bump the jevkit minor version, update `CHANGELOG.md` and README, then tag the release.

Do not use a provider's `latest` alias as the default model. It can resolve to different Jev
versions on TypeSafe and OpenRouter.
