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
6. Set `PINNED` to the candidate model after its live behavior and calibration have been reviewed.
   Keep the previous model in `KNOWN` so callers can continue selecting it with `jev=`.
7. Bump the jevkit minor version, update `CHANGELOG.md` and README, then tag the release.

Do not use a provider's `latest` alias as the default model. It can resolve to different Jev
versions on TypeSafe and OpenRouter.
